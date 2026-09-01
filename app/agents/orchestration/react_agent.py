"""ReAct Agent 创建入口与轻量质量门。

本模块负责把模型、工具、系统提示词、checkpointer 和长期记忆 store
交给 LangChain，创建可多轮对话、可自主调用工具的 ReAct Agent。
HTTP 请求、订单、库存、知识库等业务访问由服务层和工具层处理。

主要组成：
1. ``build_agent``：创建标准 ReAct Agent。
2. ``_run_config``：生成 LangGraph 运行配置。
3. ``ReflectiveAgentRunner``：提供可选的回答质量门与重试包装。
4. ``_reflect_on_answer``：基于工具、相关性和证据支撑进行规则评分。
"""

import re
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool


def _run_config(session_id: str, callbacks: list[Any] | None = None) -> dict:
    """生成 LangGraph 运行配置。

    ``thread_id`` 是 LangGraph 记忆系统的会话 key。
    同一个 thread_id 会读写同一份消息历史，所以多轮对话能接上上下文。
    callbacks 会在 LLM 开始/结束、工具开始/结束等事件触发时被 LangChain 调用。

    注意这里不直接传用户消息，只传“运行时配置”。真正的用户消息会在
    ``agent.invoke({"messages": [...]}, config=config)`` 里传入。
    """
    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
    if callbacks:
        config["callbacks"] = callbacks
    return config


# ReAct 循环交给 LangChain 维护，项目侧只负责业务工具、记忆、RAG、安全与结果整理。
def build_agent(
    chat_model: BaseChatModel,
    tools: list[BaseTool],
    checkpointer=None,
    store=None,
):
    """构建带记忆的标准 ReAct Agent。

    该函数作为 LangChain Agent 的工厂入口：
    - ``chat_model``：负责生成文本和决定工具调用。
    - ``tools``：Agent 能使用的外部能力，比如查订单、查库存、检索知识库。
    - ``system_prompt``：给 Agent 定角色、边界和回答风格。
    - ``checkpointer``：保存每个 session 的消息历史。

    System Prompt 从模型网关读取（降级链：YAML → 内置）：
      - 本地开发时：从 prompts/fulfillment_agent_v1.yaml 读取
      - 兜底：内置 hardcoded（服务不中断）

    Args:
        chat_model:   LLM 模型
        tools:        工具列表
        checkpointer: PostgreSQL Checkpointer（生产模式必传）
        store:        长期记忆 Store（预留接口）
    """
    from app.infrastructure.llm.model_gateway import get_model_gateway
    system_prompt = get_model_gateway().prompt_system(use_case="agent")

    if checkpointer is None:
        raise RuntimeError("ReAct Agent 必须显式传入 PostgreSQL checkpointer，生产模式不允许使用 MemorySaver。")

    # create_agent 返回的是一个可 invoke/ainvoke/stream/astream 的 LangGraph 编译图。
    # 之后 AgentService 调用它时，只需要传 {"messages": [...]} 和 config。
    kwargs = {
        "model": chat_model,
        "tools": tools,
        "system_prompt": system_prompt,
        "checkpointer": checkpointer,
    }
    if store is not None:
        kwargs["store"] = store
    return create_agent(**kwargs)


# ============================================================================
# 回答质量门
# ============================================================================

# 质量门使用规则分数而不是额外 LLM 评审，降低成本并保证重试原因可解释。
@dataclass
class ReflectionResult:
    """单次反思评估结果。

    这个对象不是给模型看的，而是给程序看的：它告诉外层循环这次回答
    是否应该接受，还是要带着反馈再问 Agent 一次。

    三个子分数不要理解成“真实业务正确率”，它们只是质量门的信号：
    - tool_grounding_score：工具有没有调对、结果是否可用。
    - relevance_score：答案有没有回应用户问题。
    - specificity_score：答案里的业务事实是否能被工具结果支撑。
    """

    tool_grounding_score: float   # 工具路由与工具成功分（0-0.35）
    relevance_score: float        # 问答相关性分（0-0.25）
    specificity_score: float      # 证据支撑与可执行性分（0-0.4）
    total_score: float            # 综合分（0-1）
    passed: bool                  # 是否通过质量门
    reason: str                   # 评估理由


@dataclass(frozen=True)
class ToolObservation:
    """本轮工具调用结果，用于判断答案里的事实是否有证据支撑。

    保存工具结果用于证据比对。仅记录工具是否被调用不够，答案仍可能编造
    工具结果里不存在的 SKU、仓库或数量。这里把 ToolMessage 文本保留下来，
    供后续质量门做轻量校验。
    """

    tool_name: str
    content: str
    ok: bool = True


# 下面这些正则只服务于“质量门”，不是业务解析的唯一来源。
# 它们故意做得轻量：能识别订单号、SKU、仓库、数字这些关键证据即可。
_ORDER_PAT = re.compile(r"\bSO\d{8,}\b", re.IGNORECASE)
_SKU_PAT = re.compile(r"\bSKU[-_A-Z0-9]+\b", re.IGNORECASE)
_WAREHOUSE_PAT = re.compile(r"\bWH[-_A-Z0-9]*\b|[\u4e00-\u9fff]{1,8}仓")
_NUMBER_PAT = re.compile(r"(?<![\w-])\d+(?:\.\d+)?%?(?![\w-])")
_TOKEN_PAT = re.compile(r"[\w\u4e00-\u9fff]+")


def _expected_tools_for_question(question: str) -> set[str]:
    """根据问题意图推断至少应该出现的工具。

    这里故意保守：只在业务词很明确时要求对应工具，避免把闲聊类问题误判成低质量。

    例子：
    - “能不能发货 / 有没有缺货”通常必须查库存，所以期望 ``check_inventory``。
    - “缺货怎么处理 / 有什么规则”更偏知识库，所以期望 ``retrieve_knowledge``。
    - “哪个仓能发”需要仓库库存分布，所以期望 ``search_warehouse_inventory``。

    这个函数不是要替代 LLM 的路由能力，而是在事后检查 Agent 是否明显漏调工具。
    """
    text = question.lower()
    expected: set[str] = set()

    if _ORDER_PAT.search(question) and re.search(r"订单|履约|发货|风险|分析|情况|优先级|客户|区域", question):
        expected.add("analyze_order")
    if re.search(r"库存|缺货|现货|可发|能不能发|能否发|全量履约|履约风险|发货风险", question):
        expected.add("check_inventory")
    if re.search(r"哪个仓|仓库|分仓|调拨|就近|区域仓|库存分布|全国仓", question):
        expected.add("search_warehouse_inventory")
    if re.search(r"替代|替换|兼容|平替|备选sku|替代品", text):
        expected.add("find_substitute_sku")
    if re.search(r"规则|政策|流程|知识库|怎么处理|处理建议|SOP|标准", question, re.IGNORECASE):
        expected.add("retrieve_knowledge")
    if re.search(r"方案|计划|履约方案|执行动作|怎么发|如何发", question):
        expected.add("generate_fulfillment_plan")

    return expected


def _question_needs_tool(question: str) -> bool:
    """判断问题是否依赖业务事实。

    有些问题虽然没有命中上面的强规则，但只要出现订单号或 SKU，通常就不该凭空回答。
    例如“SO202502140001 怎么样”虽然问得模糊，也应该至少查一下订单或库存。
    """
    if _expected_tools_for_question(question):
        return True
    return bool(_ORDER_PAT.search(question) or _SKU_PAT.search(question))


def _normalize_entity(value: str) -> str:
    """统一实体大小写，避免 ``sku-xxx`` 和 ``SKU-XXX`` 被当成不同实体。"""
    return value.strip().upper()


def _extract_business_entities(text: str) -> set[str]:
    """从文本中抽取可验证的业务实体。

    质量门最关心这类事实：订单号、SKU、仓库。
    如果答案里出现了这些实体，但工具结果和用户问题里都没有，就很可能是幻觉。
    """
    entities: set[str] = set()
    for pattern in (_ORDER_PAT, _SKU_PAT, _WAREHOUSE_PAT):
        entities.update(_normalize_entity(match.group(0)) for match in pattern.finditer(text))
    return entities


def _extract_meaningful_numbers(text: str) -> set[str]:
    """抽取业务数字，尽量忽略列表序号这类格式噪声。

    Agent 经常用列表回答，如果把列表序号当作业务数量，就会误判“答案有数字”。
    这里保留库存数量、件数、百分比这类更可能有业务含义的数字。
    """
    numbers: set[str] = set()
    for match in _NUMBER_PAT.finditer(text):
        value = match.group(0)
        line_start = text.rfind("\n", 0, match.start()) + 1
        prefix = text[line_start:match.start()]
        if re.fullmatch(r"\s*[-*（(]?\s*", prefix) and value.rstrip("%") in {"1", "2", "3", "4", "5"}:
            continue
        numbers.add(value)
    return numbers


def _evidence_text(tool_observations: list[ToolObservation]) -> str:
    """把本轮所有工具结果拼成一段证据文本，供实体和数字抽取使用。"""
    return "\n".join(obs.content for obs in tool_observations if obs.content)


def _ok_tool_names(tool_observations: list[ToolObservation]) -> set[str]:
    """取出成功返回的工具名。

    只看 ``tools_called`` 不够，因为工具可能被调用了但返回失败。
    质量门需要区分“调了且成功”和“调了但不可用”。
    """
    return {obs.tool_name for obs in tool_observations if obs.ok}


def _relevance_score(question: str, answer: str) -> float:
    """计算轻量相关性分。

    这里不用 embedding，也不调用 LLM judge，原因是反思质量门要便宜、稳定。
    这个分数只作为辅助信号：真正决定是否重试的主要是工具和证据。
    """
    question_tokens = {
        token for token in _TOKEN_PAT.findall(question)
        if len(token) > 1 and token not in {"请", "一下", "这个", "那个", "帮我"}
    }
    answer_tokens = set(_TOKEN_PAT.findall(answer))
    if not question_tokens:
        return 0.12
    overlap_ratio = len(question_tokens & answer_tokens) / len(question_tokens)
    return min(overlap_ratio * 0.5, 0.25)


def _looks_actionable(answer: str) -> bool:
    """判断答案是否包含可执行建议或明确结论。

    供应链 Agent 不应该只复述事实，还应该告诉用户下一步怎么处理。
    这个判断很轻量，只看一些常见动作词。
    """
    return bool(re.search(r"建议|优先|需要|可以|应|处理|方案|风险|结论|下一步|发货|调拨|替代", answer))


# 质量门检查必要工具调用、工具证据、关键实体和答案可执行性。
# 它只作为低质量输出兜底，不替代人工审核或完整评测。
def _reflect_on_answer(
    question: str,
    answer: str,
    tools_called: list[str],
    tool_observations: list[ToolObservation] | None = None,
    threshold: float = 0.6,
) -> ReflectionResult:
    """评估 Agent 答案质量。

    这是一个规则型评分器，不再调用 LLM，所以很快、很便宜、也稳定。
    它不是为了“完美判断答案对错”，而是更精确地触发反思：
    明确需要业务数据时没有调用正确工具、工具失败却给出肯定结论、答案里的关键实体
    无法从工具结果中找到支撑，这些情况才应该重试。

    评分结构：
    - 工具路由与工具成功：最高 0.35。检查该调的工具是否调了、是否成功。
    - 问答相关性：最高 0.25。检查答案有没有回应问题。
    - 证据支撑与可执行性：最高 0.40。检查答案事实是否来自工具结果，以及是否有动作建议。

    ``critical_failures`` 是硬失败：即使总分够，也不能通过。
    例如库存问题没查库存、答案编造了工具结果里没有的 SKU。
    """
    tool_observations = tool_observations or []
    answer = answer.strip()
    expected_tools = _expected_tools_for_question(question)
    called_tools = set(tools_called)
    successful_tools = _ok_tool_names(tool_observations)
    evidence = _evidence_text(tool_observations)

    parts: list[str] = []
    critical_failures: list[str] = []

    if not answer:
        critical_failures.append("答案为空")

    # 维度 1：工具路由与工具成功。
    # 这一步解决“该查库存却只查知识库”这类问题。只要问题意图明确，
    # 就要求至少命中一个期望工具，并且最好是成功返回。
    if expected_tools:
        matched_expected = expected_tools & called_tools
        matched_success = expected_tools & successful_tools
        if matched_success:
            tool_score = 0.35
        elif matched_expected:
            tool_score = 0.18
            parts.append("相关工具已调用但结果不可用")
        else:
            tool_score = 0.0
            critical_failures.append(f"未调用问题所需工具：{', '.join(sorted(expected_tools))}")
    elif called_tools:
        tool_score = 0.28 if successful_tools else 0.12
    elif _question_needs_tool(question):
        tool_score = 0.0
        critical_failures.append("问题需要业务数据，但未调用工具")
    else:
        tool_score = 0.22

    # 工具全失败时，Agent 仍然可以回答，但必须表达不确定性。
    # 如果工具失败还给出肯定结论，风险比“没有答案”更高。
    if tool_observations and not successful_tools and not re.search(r"无法|失败|未查到|不能确认|需要人工", answer):
        critical_failures.append("工具结果不可用，但答案没有说明不确定性")

    # 维度 2：相关性。
    # 这个分数故意权重较低，因为关键词重合不能证明答案正确；
    # 它只是帮助识别明显答非所问。
    relevance_score = _relevance_score(question, answer)
    if relevance_score < 0.08:
        parts.append("与问题相关性低")

    # 维度 3：证据支撑与可执行性。
    # 先分别从“答案”和“工具证据”里抽取实体/数字，再做集合差：
    # 出现在答案里，但不在工具结果、也不在用户问题里的实体，就是高风险事实。
    answer_entities = _extract_business_entities(answer)
    evidence_entities = _extract_business_entities(evidence)
    question_entities = _extract_business_entities(question)
    unsupported_entities = answer_entities - evidence_entities - question_entities

    answer_numbers = _extract_meaningful_numbers(answer)
    evidence_numbers = _extract_meaningful_numbers(evidence)
    unsupported_numbers = answer_numbers - evidence_numbers

    # specificity_score 不再等于“答案里有数字就加分”。
    # 现在更强调：具体事实是否可验证、是否和工具结果一致、是否能形成动作建议。
    specificity_score = 0.0
    if answer_entities or answer_numbers:
        specificity_score += 0.12
    if evidence:
        if answer_entities and not unsupported_entities:
            specificity_score += 0.14
        elif not answer_entities:
            specificity_score += 0.06
        if answer_numbers and not unsupported_numbers:
            specificity_score += 0.08
        elif not answer_numbers:
            specificity_score += 0.04
    if _looks_actionable(answer):
        specificity_score += 0.06
    specificity_score = min(specificity_score, 0.4)

    # 编造业务实体是硬失败，因为订单号、SKU、仓库名会直接影响业务决策。
    if unsupported_entities:
        critical_failures.append(
            "答案包含工具结果未支撑的关键实体：" + ", ".join(sorted(unsupported_entities)[:5])
        )
    # 数字更复杂：答案中可能有“第一、第二”或自然语言编号。
    # 所以这里不是一出现未支撑数字就硬失败，而是多个数字不在证据里时降低分数。
    if evidence and len(unsupported_numbers) >= 2:
        parts.append("答案包含多个工具结果未支撑的数字")
        specificity_score = min(specificity_score, 0.18)
    if not (answer_entities or answer_numbers or _looks_actionable(answer)):
        parts.append("答案缺乏具体事实或可执行动作")

    total = round(tool_score + relevance_score + specificity_score, 4)
    # 最终通过条件：分数达标，并且没有硬失败。
    # 这能避免“总分看起来够，但关键工具没调”这种误放行。
    passed = total >= threshold and not critical_failures
    reason_parts = critical_failures + parts
    reason = "；".join(reason_parts) if reason_parts else "质量达标"

    return ReflectionResult(
        tool_grounding_score=tool_score,
        relevance_score=relevance_score,
        specificity_score=specificity_score,
        total_score=total,
        passed=passed,
        reason=reason,
    )


_REFLECTION_RETRY_TEMPLATE = """\
你之前对以下问题的回答质量不够好（评分：{score:.2f}/1.0，问题：{reason}）。

原始问题：{question}
你之前的回答：{previous_answer}

请重新回答，这次务必：
1. 调用与问题意图匹配的工具获取数据，不要凭记忆回答
2. 只引用工具结果中真实出现的订单号、SKU、仓库、数量和规则
3. 如果工具没有查到或返回失败，要明确说明不确定性和下一步处理方式
4. 直接针对问题核心给出结论和可执行建议
"""


# 反思包装器保持标准 Agent 图不变，仅在外层根据规则评分决定是否带反馈重试。
class ReflectiveAgentRunner:
    """带自反思循环的 Agent 运行器。

    它使用装饰器思路：不改 LangChain Agent 内部图，只在外面包一层
    “执行 -> 评分 -> 必要时重试”的循环。

    设计取舍：
      - 写进 prompt 不容易测试，也不容易知道具体哪里失败。
      - 外层评分器能返回结构化分数和 reason，方便日志、评测和调试。
      - 开关更简单：生产环境想省 token 时可以不用反思。

    使用方式：
        runner = ReflectiveAgentRunner(agent, max_retries=2, threshold=0.6)
        result = runner.run(session_id, message)
    """

    def __init__(
        self,
        agent,           # build_agent() 返回的 CompiledGraph
        max_retries: int = 2,
        threshold: float = 0.6,
    ):
        self._agent = agent
        self.max_retries = max_retries
        self.threshold = threshold

    def run(self, session_id: str, message: str, callbacks: list[Any] | None = None) -> dict:
        """执行带反思的 Agent 对话。

        Returns:
            {
              "reply": str,
              "tools_called": list[str],
              "reflection": ReflectionResult,  # 最终轮的反思结果
              "retry_count": int,              # 实际重试次数
            }
        """
        config = _run_config(session_id, callbacks)
        current_message = message

        for attempt in range(self.max_retries + 1):
            # 第一次传用户原始问题；如果评分没过，下一轮传“带反馈的重试问题”。
            result = self._agent.invoke(
                {"messages": [HumanMessage(content=current_message)]},
                config=config,
            )
            # result 是完整消息历史，这里压缩成：
            # - 最终回复
            # - 本轮调用过的工具名
            # - 本轮工具返回的证据文本
            reply, tools_called, tool_observations = self._extract_reply(result, current_message)

            # 评分始终用原始问题做参照，避免第二轮的“重试提示”污染相关性判断。
            reflection = _reflect_on_answer(
                question=message,  # 始终针对原始问题评估
                answer=reply,
                tools_called=tools_called,
                tool_observations=tool_observations,
                threshold=self.threshold,
            )

            # 通过质量门就停止。注意 retry_count=attempt，因此第一次成功时是 0。
            if reflection.passed:
                return {
                    "reply": reply,
                    "tools_called": tools_called,
                    "reflection": reflection,
                    "retry_count": attempt,
                }

            # 最后一次重试仍未通过 → 返回当前结果（附警告）
            if attempt >= self.max_retries:
                return {
                    "reply": reply,
                    "tools_called": tools_called,
                    "reflection": reflection,
                    "retry_count": attempt,
                }

            # 把失败原因显式告诉 Agent，比只说“再试一次”更容易得到更扎实的答案。
            # 注意 previous_answer 只截前 300 字，避免把错误答案塞太长，污染下一轮上下文。
            current_message = _REFLECTION_RETRY_TEMPLATE.format(
                score=reflection.total_score,
                reason=reflection.reason,
                question=message,
                previous_answer=reply[:300],
            )

        # 不应到达此处
        return {"reply": "", "tools_called": [], "reflection": None, "retry_count": self.max_retries}

    async def arun(self, session_id: str, message: str, callbacks: list[Any] | None = None) -> dict:
        """异步版本的 run()。

        FastAPI 路由和流式接口通常运行在 asyncio 事件循环里。
        在这些地方应该用 ``ainvoke``，不要用同步 ``invoke`` 阻塞事件循环。
        """
        config = _run_config(session_id, callbacks)
        current_message = message

        for attempt in range(self.max_retries + 1):
            # 异步版本和同步版本逻辑一致，只是把 invoke 换成 ainvoke。
            # FastAPI 路由里应该走这里，避免阻塞事件循环。
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=current_message)]},
                config=config,
            )
            reply, tools_called, tool_observations = self._extract_reply(result, current_message)
            reflection = _reflect_on_answer(
                question=message,
                answer=reply,
                tools_called=tools_called,
                tool_observations=tool_observations,
                threshold=self.threshold,
            )
            if reflection.passed or attempt >= self.max_retries:
                return {"reply": reply, "tools_called": tools_called, "reflection": reflection, "retry_count": attempt}

            current_message = _REFLECTION_RETRY_TEMPLATE.format(
                score=reflection.total_score,
                reason=reflection.reason,
                question=message,
                previous_answer=reply[:300],
            )

        return {"reply": "", "tools_called": [], "reflection": None, "retry_count": self.max_retries}

    @staticmethod
    def _extract_reply(result: dict, user_message: str = "") -> tuple[str, list[str], list[ToolObservation]]:
        """从 Agent invoke 结果中提取回复和工具调用列表。

        LangChain Agent 返回的是完整消息历史，不是单个字符串。
        这里通过 ``filter_messages`` 只取 AIMessage：
        - 带 ``tool_calls`` 的 AIMessage：模型请求调用工具。
        - 不带 ``tool_calls`` 的 AIMessage：通常是最终回答。

        为什么要传 ``user_message``？
        同一个 session 会保存多轮历史。如果不截取“当前用户消息之后”的部分，
        反思评分可能会把上一轮的工具调用也算进来，导致误判。
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, filter_messages

        messages = result.get("messages", [])
        if user_message:
            # 找到当前轮用户消息的位置，只评估它之后产生的 AIMessage / ToolMessage。
            user_idx = next(
                (
                    i for i, msg in enumerate(messages)
                    if isinstance(msg, HumanMessage) and msg.content == user_message
                ),
                -1,
            )
            current_turn = messages[user_idx + 1:] if user_idx >= 0 else messages
        else:
            current_turn = messages

        ai_msgs = filter_messages(current_turn, include_types=[AIMessage])
        tool_msgs = filter_messages(current_turn, include_types=[ToolMessage])

        reply = ""
        tools_called: list[str] = []
        observations: list[ToolObservation] = []

        for msg in ai_msgs:
            # AIMessage.tool_calls 记录“模型想调用哪些工具”。
            # 这里收集工具名，用来判断路由是否正确。
            for tc in msg.tool_calls or []:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                if name:
                    tools_called.append(name)
            # 没有 tool_calls 的 AIMessage 通常是最终自然语言答案。
            # 如果有多个，以最后一个为准。
            if not msg.tool_calls:
                reply = str(msg.content)

        for msg in tool_msgs:
            # ToolMessage 是工具真正返回给模型的内容。
            # 这里把它转成 ToolObservation，后续用来做证据比对。
            content = str(msg.content)
            observations.append(ToolObservation(
                tool_name=str(getattr(msg, "name", "") or ""),
                content=content,
                ok=not re.search(r'"status"\s*:\s*"error"|失败|Traceback|Exception', content, re.IGNORECASE),
            ))

        return reply, tools_called, observations


def build_reflective_agent(
    chat_model: BaseChatModel,
    tools: list[BaseTool],
    checkpointer,
    store=None,
    max_retries: int = 2,
    threshold: float = 0.6,
) -> ReflectiveAgentRunner:
    """构建带自反思循环的 Agent。

    与 build_agent 的区别：
      - 答案质量低于阈值时自动重试（最多 max_retries 次）
      - 每次重试附带上一次的答案和评估原因
      - 适合高准确率要求的场景（代价：更多 LLM 调用）

    Args:
        chat_model: LLM 模型
        tools:      Agent 可用工具列表
        checkpointer: PostgreSQL Checkpointer（生产模式必传）
        store:      长期记忆 Store
        max_retries: 最大重试次数（默认 2）
        threshold:   质量门阈值 0-1（默认 0.6）

    Returns:
        ReflectiveAgentRunner，调用 .run(session_id, message) 使用
    """
    agent = build_agent(chat_model, tools, checkpointer=checkpointer, store=store)
    return ReflectiveAgentRunner(agent, max_retries=max_retries, threshold=threshold)
