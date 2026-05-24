"""LangGraph 工作流编排模块。

这个目录负责“固定 workflow”这条链路：用代码明确规定节点顺序、条件分支、
人工审批中断和最终汇总。它和 ReAct Agent 不同：Workflow 追求稳定、可解释，
Agent 追求灵活工具调用。

推荐阅读顺序：
    1. state.py              —— 先看整张图共享哪些字段。
    2. workflow.py           —— 再看节点如何连成串行主图。
    3. nodes.py              —— 最后看每个节点具体做什么。
    4. router.py             —— 理解条件边只负责读 state 决定下一跳。
    5. trace.py              —— 理解可观察性数据如何随 state 累加。
    6. risk_evaluator.py     —— 理解 HITL 为什么会被触发。
    7. parallel_workflow.py  —— 在读懂串行图后，再看并行 fan-out/fan-in。
    8. embed_adapter.py      —— RAG / 长期记忆需要向量化时再看。
    9. llm_adapter.py        —— 模型网关到 LangChain ChatModel 的兼容入口。
    10. ports.py             —— 已废弃迁移说明。

设计约束：
    1. 本目录只依赖 app/services 与 app/schemas，不反向依赖 api/ 层。
    2. 节点不直接访问数据源，一律通过已有 service。
    3. 节点只返回 state 增量，不返回整份 state。
    4. trace / errors 作为显式可观察数据随流程返回，便于调试和面试讲解。
"""
