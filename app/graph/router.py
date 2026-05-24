"""条件边路由函数。

LangGraph 1.0+ 的条件边是"函数式"的：
    add_conditional_edges(source, routing_fn, {return_value: target_node, ...})

路由函数的契约：
    - 入参：当前 state
    - 出参：一个字符串，必须是 mapping 里注册过的 key

本回合只有一条条件边：
    inventory_analysis → (fulfillable | stockout)

为什么把路由字符串保持稳定？
    mapping 中的 key 是"契约"，后续扩展分支时应该加新值，
    而不是修改已有值，避免破坏图结构。
"""

from app.graph.state import GraphState


def route_after_inventory(state: GraphState) -> str:
    """库存判断完成后的分支决策。

    规则非常简单：读取 inventory_analysis_node 已经写好的 fulfillment_branch 字段。
    不在这里做业务判断，保持路由函数"薄"且纯。

    若字段意外缺失（例如上游异常），兜底走 stockout，
    因为 stockout 路径更完整（会经过知识检索），对异常订单更稳妥。
    """
    branch = state.get("fulfillment_branch")
    if branch == "fulfillable":
        return "fulfillable"
    return "stockout"
