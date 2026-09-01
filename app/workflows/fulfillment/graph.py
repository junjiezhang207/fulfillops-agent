"""构建并编译 LangGraph StateGraph。

这个文件只负责“搭图”，不写业务逻辑。业务逻辑都在 ``nodes.py`` 里。
学习 LangGraph 时可以先看这里，因为图结构决定了整个 workflow 的执行顺序。

拓扑：
    START
      → dispatch
      → order_analysis
      → inventory_analysis
      ├── [条件边 route_after_inventory]
      │     ├─ "fulfillable" → finalize
      │     └─ "stockout"    → knowledge_retrieval → finalize
      └── END

参考：https://docs.langchain.com/oss/python/langgraph/overview

这条 workflow 和 Agent 的区别：
  - Workflow 是固定路径：节点顺序和分支规则由代码写死，稳定、可解释。
  - Agent 是动态路径：LLM 自己决定下一步调哪个工具，更灵活但更难预测。
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.workflows.fulfillment.nodes import WorkflowNodes
from app.workflows.fulfillment.router import route_after_inventory
from app.workflows.fulfillment.state import GraphState


def build_workflow(
    nodes: WorkflowNodes,
    checkpointer=None,
    store=None,
) -> CompiledStateGraph:
    """构建并编译工作流图。

    LangGraph 的核心概念：
      - StateGraph(GraphState)：声明整张图共享的状态结构。
      - add_node：注册节点函数，节点输入完整 state，返回 state 增量。
      - add_edge：固定边，表示某节点结束后一定去下一个节点。
      - add_conditional_edges：条件边，根据 state 决定下一跳。
      - compile：把声明式图编译成可 invoke/ainvoke/stream 的运行对象。

    Args:
        nodes:        已注入 service 依赖的节点集合。
        checkpointer: 短期记忆 Checkpointer。生产模式必须由服务层传入 PostgreSQL checkpointer。
        store:        长期记忆 Store（默认 None）。
                      可由 create_long_term_memory_store() 创建；当前默认 PostgreSQL + PGVector。

    接入示例：
        from app.memory import create_postgres_checkpointer, create_long_term_memory_store
        graph = build_workflow(
            nodes,
            checkpointer=create_postgres_checkpointer("postgresql+psycopg://..."),
            store=create_long_term_memory_store(database_url="postgresql+psycopg://..."),
        )
    """
    if checkpointer is None:
        raise RuntimeError("Workflow 必须显式传入 PostgreSQL checkpointer，生产模式不允许使用 MemorySaver。")

    # GraphState 是 TypedDict。LangGraph 不要求节点返回完整 state，
    # 每个节点只返回自己新增/更新的字段，框架会自动合并到共享 state。
    graph: StateGraph = StateGraph(GraphState)

    # 注册节点：key 是图里的节点名，value 是真正执行的 Python callable。
    # 这些 callable 都是 WorkflowNodes 的方法，已经在 service 层注入了业务依赖。
    graph.add_node("dispatch", nodes.dispatch)
    graph.add_node("order_analysis", nodes.order_analysis)
    graph.add_node("inventory_analysis", nodes.inventory_analysis)
    graph.add_node("proposal_generation", nodes.proposal_generation)
    graph.add_node("human_approval", nodes.human_approval)
    graph.add_node("knowledge_retrieval", nodes.knowledge_retrieval)
    graph.add_node("finalize", nodes.finalize)

    # 主干顺序边：每次 workflow 都先校验入参，再查订单，再查库存。
    graph.add_edge(START, "dispatch")
    graph.add_edge("dispatch", "order_analysis")
    graph.add_edge("order_analysis", "inventory_analysis")
    graph.add_edge("inventory_analysis", "proposal_generation")
    graph.add_edge("proposal_generation", "human_approval")

    # 条件边：审批/二次校验后会写 fulfillment_branch。
    # route_after_inventory 读取这个字段并返回 "fulfillable" 或 "stockout"。
    graph.add_conditional_edges(
        "human_approval",
        route_after_inventory,
        {
            "fulfillable": "finalize",
            "stockout": "knowledge_retrieval",
        },
    )

    # 缺货路径需要先查规则，再进入 finalize 汇总。
    graph.add_edge("knowledge_retrieval", "finalize")

    # finalize 是唯一出口：无论快路径还是缺货路径，最终都在这里形成 FinalAnswer。
    graph.add_edge("finalize", END)

    # checkpointer 让 LangGraph 能保存 thread_id 对应的中间状态。
    # 这对 HITL interrupt/resume 很重要：中断后恢复执行要找回之前的 state。
    # store 是 LangGraph 的长期记忆接口，和 checkpointer 不同：
    #   checkpointer 记“这次会话/这张图跑到哪了”
    #   store        记“跨会话仍有价值的业务记忆”
    compile_kwargs = {"checkpointer": checkpointer}
    if store is not None:
        compile_kwargs["store"] = store
    return graph.compile(**compile_kwargs)
