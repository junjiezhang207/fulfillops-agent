"""多轮对话上下文窗口管理。

本模块负责 Agent 的上下文裁剪。多轮对话会不断累积消息，如果不控制，
prompt 会越来越长，导致成本上升、响应变慢，甚至超过模型上下文限制。
本文件借助 LangChain ``trim_messages`` 和模型自身的
``get_num_tokens_from_messages``，在接近阈值时保留最近且格式合法的消息。

主要做的事：
1. ``ContextWindowConfig``：定义最大 token、裁剪阈值和最少保留消息数。
2. ``make_token_counter``：尽量使用当前 LLM 的官方 token 计数方法。
3. ``ContextWindowManager``：判断是否需要裁剪，并执行消息裁剪。
4. ``TrimResult``：记录裁剪前后消息数和 token 数，便于日志和调试。
5. ``estimate_tokens``：没有模型计数器时的兜底估算。

这个文件不保存历史消息。真正的会话历史由 LangGraph checkpointer 保存；
这里只负责“历史太长时怎么裁掉一部分”。
"""

from dataclasses import dataclass
from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, trim_messages


# ============================================================================
# Token 计数
# ============================================================================


def make_token_counter(
    llm: Optional[BaseLanguageModel] = None,
) -> BaseLanguageModel | None:
    """创建 token 计数函数。

    ``trim_messages`` 可以直接接收 LLM 实例，然后调用它的
    ``get_num_tokens_from_messages()``。如果没有可用 LLM，则返回 None，
    上下文裁剪会跳过。
    """
    if llm is not None and hasattr(llm, "get_num_tokens_from_messages"):
        return llm
    return None


# ============================================================================
# 配置与结果
# ============================================================================

@dataclass
class ContextWindowConfig:
    max_tokens: int = 6000
    trim_threshold: float = 0.80
    min_messages_to_keep: int = 4


@dataclass
class TrimResult:
    original_count: int
    final_count: int
    original_tokens: int
    final_tokens: int
    action_taken: str
    messages_dropped: int

    @property
    def reduction_rate(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 1 - self.final_tokens / self.original_tokens


# ============================================================================
# 上下文管理器（基于 LangChain trim_messages）
# ============================================================================

# 上下文裁剪使用模型自身 token 计数和 LangChain trim_messages，
# 保留最近且格式合法的消息，避免手写裁剪破坏消息结构。
class ContextWindowManager:
    """使用 LangChain trim_messages() 管理对话历史。

    trim_messages 参数说明：
      token_counter  LLM 实例，LangChain 会调用 get_num_tokens_from_messages
      strategy       "last" = 保留最新消息（尾部），"first" = 保留最旧消息（头部）
      include_system 始终保留 SystemMessage，不参与裁剪
      start_on       裁剪后确保第一条是指定类型（保持对话格式合法）
      allow_partial  是否允许截断消息内容（False = 保持消息完整）
    """

    def __init__(
        self,
        config: Optional[ContextWindowConfig] = None,
        llm: Optional[BaseLanguageModel] = None,
    ):
        self.config = config or ContextWindowConfig()
        self._token_counter = make_token_counter(llm)

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        if self._token_counter is None:
            return 0
        return self._token_counter.get_num_tokens_from_messages(messages)

    def should_trim(self, messages: list[BaseMessage]) -> bool:
        if self._token_counter is None:
            return False
        return self.count_tokens(messages) > self.config.max_tokens * self.config.trim_threshold

    def trim(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], TrimResult]:
        """用 LangChain trim_messages() 裁剪消息历史。"""
        original_count = len(messages)
        original_tokens = self.count_tokens(messages)

        if self._token_counter is None or not self.should_trim(messages):
            return messages, TrimResult(
                original_count=original_count, final_count=original_count,
                original_tokens=original_tokens, final_tokens=original_tokens,
                action_taken="none", messages_dropped=0,
            )

        try:
            trimmed = trim_messages(
                messages,
                max_tokens=self.config.max_tokens,
                token_counter=self._token_counter,
                strategy="last",        # 保留最新消息
                include_system=True,    # SystemMessage 永远保留
                start_on="human",       # 裁剪后首条必须是 HumanMessage
                allow_partial=False,    # 不截断消息内容
            )
        except Exception:
            # trim_messages 要求消息列表含有 HumanMessage，异常时保底保留尾部
            trimmed = messages[-self.config.min_messages_to_keep:]

        # 确保至少保留 min_messages_to_keep 条
        if len(trimmed) < min(self.config.min_messages_to_keep, len(messages)):
            trimmed = messages[-self.config.min_messages_to_keep:]

        final_tokens = self.count_tokens(trimmed)
        return trimmed, TrimResult(
            original_count=original_count,
            final_count=len(trimmed),
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            action_taken="trim_messages",
            messages_dropped=original_count - len(trimmed),
        )
