"""文件作用摘要：把 Agent 调用链路接入 Langfuse 可观测平台。

这个文件负责 Langfuse 集成。Langfuse 用来观察一次大模型调用的完整链路：
模型输入输出、工具调用树、trace_id、人工或自动评分等。它不是业务逻辑，
也不影响 Agent 决策；配置了就上报，没配置就自动跳过。

主要做的事：
1. ``LangfuseTracer``：封装 Langfuse client 和 LangChain CallbackHandler。
2. ``callback``：提供给 LangChain / LangGraph config，用于自动采集 trace。
3. ``score``：向某个 trace 写入质量分数，例如工具支撑、答案完整性、反思结果。
4. ``create_langfuse_tracer``：根据 settings 创建 tracer；未配置 key 时返回 None。
5. ``score_after_run``：Agent 一轮结束后，把质量信号写回 Langfuse。

上报的质量信号：
- ``tool_grounding``：回答是否有工具调用支撑。
- ``answer_completeness``：是否生成了非空答案。
- ``overall_quality``：反思质量门分数，启用反思时才有。
- ``reflection_passed``：反思质量门是否通过。

环境变量：
- ``LANGFUSE_PUBLIC_KEY``：Langfuse 项目 public key。
- ``LANGFUSE_SECRET_KEY``：Langfuse 项目 secret key。
- ``LANGFUSE_HOST``：自托管地址，留空时使用云端地址。

学习时先看：
1. ``create_langfuse_tracer``：未配置时如何不影响主流程。
2. ``LangfuseTracer.callback``：如何接入 LangChain callbacks。
3. ``score_after_run``：一轮 Agent 结束后写哪些评分。
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any

logger = logging.getLogger(__name__)


# ── Langfuse 可观测链路 ──────────────────────────────────────────────────────

# 面试官可能问：为什么用了 Langfuse 还要自己封一层 LangfuseTracer？
# 回答：封装后业务代码不用直接依赖 Langfuse SDK 细节。没配置 key 时返回 None，
# 主链路不受影响；配置后只把 callback 注入 LangChain，并在结束后写质量分。
class LangfuseTracer:
    """封装 Langfuse Client + LangChain CallbackHandler。

    正常路径：
        tracer = LangfuseTracer(settings)
        # 在 agent.invoke() 的 callbacks 里注入 tracer.callback
        result = agent.invoke(input, config={"callbacks": [tracer.callback]})
        # 拿到 trace_id 后写质量信号
        tracer.score(tracer.get_trace_id(), "tool_grounding", 1.0)
        tracer.flush()
    """

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler  # 3.x+ 路径

        langfuse_host = host or "https://cloud.langfuse.com"
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_HOST"] = langfuse_host
        os.environ["LANGFUSE_BASE_URL"] = langfuse_host

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=langfuse_host,
        )
        try:
            self._handler = CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=langfuse_host,
            )
        except TypeError:
            # langfuse>=3 的 LangChain handler 从环境变量读取 secret/host，
            # 构造函数只保留 public_key 等少量参数。
            self._handler = CallbackHandler(public_key=public_key)

    @property
    def callback(self):
        """注入到 LangChain config["callbacks"] 的 handler。"""
        return self._handler

    @contextmanager
    def start_span(self, name: str, input: Any = None, metadata: Any = None):
        """显式创建 Langfuse span，确保非流式 Hybrid/Agent 调用也有 trace。"""
        try:
            span_context = self._client.start_as_current_span(
                name=name,
                input=input,
                metadata=metadata,
            )
            with span_context as span:
                yield span
        except Exception as exc:
            logger.debug("Langfuse start_span failed (non-critical): %s", exc)
            with nullcontext() as span:
                yield span
        finally:
            self.flush()

    def get_trace_id(self) -> str | None:
        try:
            return self._handler.get_trace_id()
        except Exception:
            return None

    def get_trace_url(self) -> str | None:
        try:
            return self._handler.get_trace_url()
        except Exception:
            return None

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str = "",
    ) -> None:
        """向 Langfuse trace 写入数值评分（0-1），失败静默。"""
        try:
            if hasattr(self._client, "create_score"):
                self._client.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=comment,
                )
            else:
                self._client.score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=comment,
                )
        except Exception as exc:
            logger.debug("Langfuse score failed (non-critical): %s", exc)

    def flush(self) -> None:
        """批量上报挂起的 trace 事件（服务关闭时调用）。"""
        try:
            self._client.flush()
        except Exception:
            pass


# ── 工厂函数（优雅降级）──────────────────────────────────────────────────────

# 面试官可能问：为什么未配置 Langfuse 时不能直接报错？
# 回答：可观测是增强能力，不应该阻断本地开发和面试演示。没有 Langfuse 时
# Agent 仍然能跑，只是少了 trace 平台；生产环境再要求必须配置。
def create_langfuse_tracer(settings) -> LangfuseTracer | None:
    """按配置构建 LangfuseTracer，缺少 key 时返回 None（静默 no-op）。

    Args:
        settings: app.core.config.Settings 实例
    """
    if not getattr(settings, "langfuse_public_key", "") or \
       not getattr(settings, "langfuse_secret_key", ""):
        return None
    try:
        host = getattr(settings, "langfuse_host", "") or getattr(settings, "langfuse_base_url", "")
        return LangfuseTracer(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=host,
        )
    except ImportError:
        logger.warning("langfuse 未安装，跳过可观测性集成。运行 `uv add langfuse` 安装。")
        return None
    except Exception as exc:
        logger.warning("Langfuse 初始化失败（非致命）：%s", exc)
        return None


# ── 评分辅助 ─────────────────────────────────────────────────────────────────

# 面试官可能问：score_after_run 写的这些分数有什么意义？
# 回答：它把“有没有工具支撑、答案是否完整、反思是否通过”写成可查询指标。
# 后续可以在 Langfuse 里筛选低分样本，反向优化 prompt、工具描述和 RAG 策略。
def score_after_run(
    tracer: LangfuseTracer | None,
    trace_id: str | None,
    reply: str,
    tools_called: list[str],
    reflection_score: float | None = None,
    reflection_passed: bool | None = None,
    reflection_reason: str = "",
) -> None:
    """根据已有信息向 Langfuse 写入质量信号，不依赖额外 LLM 调用。

    信号维度：
      tool_grounding   — 是否调用了至少一个工具（1 = 有，0 = 无）
      answer_completeness — 是否生成了非空答案（1 = 有，0 = 无）
      overall_quality  — 只在 reflection_score 存在时写入，避免把启发式估算当成真实质量分
      reflection_passed — 反思质量门是否通过（1 = 通过，0 = 未通过）

    设计说明：
      这里不上 LLM-as-judge，避免增加延迟和成本。
      没有反思分时，只写客观信号；真正的 LLM-as-judge 建议走 Langfuse
      Eval Template，对历史 trace 做异步评分。
    """
    if tracer is None:
        return
    if trace_id is None:
        tracer.flush()
        return

    tool_grounding = 1.0 if tools_called else 0.0
    answer_completeness = 1.0 if reply and reply.strip() else 0.0

    tracer.score(trace_id, "tool_grounding", tool_grounding,
                 comment=f"tools={tools_called}")
    tracer.score(trace_id, "answer_completeness", answer_completeness,
                 comment=f"reply_chars={len(reply or '')}")

    if reflection_score is not None:
        comment = f"reflection_reason={reflection_reason}" if reflection_reason else "reflection_score"
        tracer.score(trace_id, "overall_quality", round(reflection_score, 3), comment=comment)

    if reflection_passed is not None:
        tracer.score(
            trace_id,
            "reflection_passed",
            1.0 if reflection_passed else 0.0,
            comment=reflection_reason,
        )

    tracer.flush()
