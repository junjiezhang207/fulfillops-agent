#!/usr/bin/env python3
"""验证三个 Agent/Workflow 亮点特性。

特性 1: 并行工作流（LangGraph fan-out/fan-in + Send API）
特性 2: 多 Agent 编排（Supervisor 模式）
特性 3: 自反思循环（ReflectiveAgentRunner）
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.graph.llm_adapter import LLMFactory
from app.repositories.file_system_knowledge_repository import FileSystemKnowledgeRepository
from app.repositories.in_memory_inventory_repository import InMemoryInventoryRepository
from app.repositories.in_memory_order_repository import InMemoryOrderRepository
from app.services.fulfillment_plan_service import FulfillmentPlanService
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.multi_agent_service import MultiAgentService
from app.services.order_analysis_service import OrderAnalysisService
from app.services.substitute_sku_service import SubstituteSkuService
from app.services.warehouse_service import WarehouseService
from app.graph.parallel_workflow import create_parallel_workflow_nodes, build_parallel_workflow
from app.agent.agent import _reflect_on_answer


def setup_services():
    """组装所有服务依赖。"""
    settings = get_settings()
    order_repo = InMemoryOrderRepository()
    order_svc = OrderAnalysisService(order_repo)
    inv_repo = InMemoryInventoryRepository()
    inv_svc = InventoryAnalysisService(
        inventory_repository=inv_repo,
        order_analysis_service=order_svc,
    )
    knowledge_repo = FileSystemKnowledgeRepository(settings.knowledge_dir)
    knowledge_svc = KnowledgeRetrievalService(
        knowledge_repository=knowledge_repo,
        inventory_analysis_service=inv_svc,
    )
    warehouse_svc = WarehouseService()
    substitute_svc = SubstituteSkuService()
    fulfillment_svc = FulfillmentPlanService(
        inventory_service=inv_svc,
        warehouse_service=warehouse_svc,
        substitute_service=substitute_svc,
    )
    llm = LLMFactory.create(settings)
    chat_model = LLMFactory.create_chat_model(settings)

    return {
        "order_svc": order_svc,
        "inv_svc": inv_svc,
        "knowledge_svc": knowledge_svc,
        "warehouse_svc": warehouse_svc,
        "substitute_svc": substitute_svc,
        "fulfillment_svc": fulfillment_svc,
        "llm": llm,
        "chat_model": chat_model,
    }


def demo_parallel_workflow(svcs: dict):
    """特性 1：并行工作流验证。"""
    print()
    print("=" * 70)
    print("特性 1: 并行工作流 (LangGraph fan-out/fan-in + Send API)")
    print("=" * 70)

    # 构建并行工作流
    nodes = create_parallel_workflow_nodes(
        order_service=svcs["order_svc"],
        inventory_service=svcs["inv_svc"],
        knowledge_service=svcs["knowledge_svc"],
        llm=svcs["llm"],
    )
    graph = build_parallel_workflow(nodes)

    order_id = "SO202502140001"
    print(f"\n[执行] 订单 {order_id}，并行工作流")

    start = time.time()
    final_state = graph.invoke({
        "order_id": order_id,
        "question": "该订单能否按时履约？",
        "filter_categories": [],
        "trace": [],
        "errors": [],
    })
    elapsed = (time.time() - start) * 1000

    # 打印执行轨迹
    trace = final_state.get("trace", [])
    print(f"\n执行轨迹（{len(trace)} 个节点）：")
    for t in trace:
        print(f"  [{t.status.upper():7s}] {t.node:25s}  {t.elapsed_ms}ms  {t.note or ''}")

    # 分析并行效果
    node_names = [t.node for t in trace]
    print(f"\n节点执行顺序：{' -> '.join(node_names)}")

    # 检查是否有并行节点
    parallel_stage1 = {"order_analysis", "inventory_analysis"}
    parallel_stage2 = {"knowledge_retrieval", "warehouse_search"}
    executed_nodes = set(node_names)

    if parallel_stage1.issubset(executed_nodes):
        print("[OK] Stage-1 并行：order_analysis + inventory_analysis 均执行")
    if parallel_stage2.issubset(executed_nodes):
        print("[OK] Stage-2 并行：knowledge_retrieval + warehouse_search 均执行（Send API）")
    elif "knowledge_retrieval" in executed_nodes:
        print("[OK] Stage-2 快速路径：库存充足，直接 finalize")

    warehouse_result = final_state.get("warehouse_search_result")
    if warehouse_result:
        print(f"[OK] warehouse_search 结果：{str(warehouse_result)[:80]}...")

    final_answer = final_state.get("final_answer")
    if final_answer:
        print(f"\n最终答案：{final_answer.conclusion[:150]}")

    errors = final_state.get("errors", [])
    if errors:
        print(f"[WARN] 错误：{errors}")

    print(f"\n总耗时：{elapsed:.0f}ms")
    print("\n亮点说明：")
    print("  - dispatch 后 order_analysis + inventory_analysis 同时触发（fan-out）")
    print("  - stage1_barrier 等两者都完成后才继续（fan-in）")
    print("  - 缺货时 Send API 同时启动 knowledge_retrieval + warehouse_search")
    print("  - 理论延迟降低 22-50%（取决于实际 I/O 耗时）")


def demo_multi_agent(svcs: dict):
    """特性 2：多 Agent 编排验证。"""
    print()
    print("=" * 70)
    print("特性 2: 多 Agent 编排 (Supervisor 模式)")
    print("=" * 70)

    service = MultiAgentService(
        order_service=svcs["order_svc"],
        inventory_service=svcs["inv_svc"],
        knowledge_service=svcs["knowledge_svc"],
        warehouse_service=svcs["warehouse_svc"],
        substitute_service=svcs["substitute_svc"],
        fulfillment_service=svcs["fulfillment_svc"],
        llm=svcs["chat_model"],
    )

    order_id = "SO202502140001"
    question = "这个订单的库存是否充足？如有缺货请给出替代方案，并评估履约风险。"

    print(f"\n[执行] 订单 {order_id}")
    print(f"问题：{question}")
    print()

    start = time.time()
    result = service.run(order_id=order_id, question=question)
    elapsed = (time.time() - start) * 1000

    print(f"调用了 {result.total_agents_called} 个 Agent：{result.agents_called}")
    print()

    print("Supervisor 推理过程：")
    for r in result.supervisor_reasoning:
        print(f"  {r}")

    print("\n各 Agent 输出摘要：")
    for record in result.agent_records:
        print(f"  [{record.agent_name}] {record.summary[:150]}...")

    print(f"\n最终综合答案：\n{result.final_answer[:400]}")

    print(f"\n总耗时：{elapsed:.0f}ms")
    print("\n亮点说明：")
    print("  - Supervisor 按问题复杂度动态决定调哪些专业 Agent")
    print("  - 每个专业 Agent 工具集聚焦，减少 LLM 推理干扰")
    print("  - Supervisor 循环：Agent 完成 -> Supervisor 判断是否足够 -> 继续或汇总")
    print("  - 无 LLM 时自动降级为规则驱动（Graceful Degradation）")


def demo_self_reflection(svcs: dict):
    """特性 3：自反思循环验证。"""
    print()
    print("=" * 70)
    print("特性 3: 自反思循环 (ReflectiveAgentRunner)")
    print("=" * 70)

    print("\n[场景 A] 有工具调用 + 具体数据 -> 应通过质量门")
    result_a = _reflect_on_answer(
        question="订单 SO202502140001 的 SKU-A001 库存是否充足？",
        answer="根据库存查询，SKU-A001 在 WH-SH 仓库有 500 件可用，订单需求 100 件，库存充足，可以正常履约。",
        tools_called=["check_inventory"],
    )
    print(f"  工具调用分: {result_a.tool_grounding_score:.2f} / 0.40")
    print(f"  相关性分:   {result_a.relevance_score:.2f} / 0.30")
    print(f"  具体性分:   {result_a.specificity_score:.2f} / 0.30")
    print(f"  综合分:     {result_a.total_score:.2f} / 1.00")
    print(f"  通过质量门: {'[PASS]' if result_a.passed else '[FAIL]'}  ({result_a.reason})")

    print("\n[场景 B] 无工具调用 + 模糊回答 -> 应触发重试")
    result_b = _reflect_on_answer(
        question="订单 SO202502140001 的 SKU-A001 库存是否充足？",
        answer="建议您查看一下库存系统，了解具体情况后再做决定。",
        tools_called=[],
    )
    print(f"  工具调用分: {result_b.tool_grounding_score:.2f} / 0.40  <- 未调用工具")
    print(f"  相关性分:   {result_b.relevance_score:.2f} / 0.30")
    print(f"  具体性分:   {result_b.specificity_score:.2f} / 0.30  <- 无具体数据")
    print(f"  综合分:     {result_b.total_score:.2f} / 1.00")
    print(f"  通过质量门: {'[PASS]' if result_b.passed else '[FAIL]'}  ({result_b.reason})")
    if not result_b.passed:
        print("  -> 将触发重试，提示 Agent 必须调用工具获取真实数据")

    print("\n[场景 C] 有工具调用但答案不相关 -> 部分通过")
    result_c = _reflect_on_answer(
        question="订单 SO202502140001 的 SKU-A001 库存是否充足？",
        answer="根据系统查询，整体仓库运营情况良好，各项指标正常，请放心。",
        tools_called=["check_inventory"],
    )
    print(f"  工具调用分: {result_c.tool_grounding_score:.2f} / 0.40  <- 有工具调用")
    print(f"  相关性分:   {result_c.relevance_score:.2f} / 0.30  <- 答非所问")
    print(f"  具体性分:   {result_c.specificity_score:.2f} / 0.30")
    print(f"  综合分:     {result_c.total_score:.2f} / 1.00")
    print(f"  通过质量门: {'[PASS]' if result_c.passed else '[FAIL]'}  ({result_c.reason})")

    print("\n亮点说明：")
    print("  - 三维评估：工具调用(40%) + 相关性(30%) + 具体性(30%)")
    print("  - 低于 0.6 分触发重试，提示 Agent 修正方向")
    print("  - 最多重试 2 次，防止无限循环")
    print("  - 避免'自信但错误'的答案直接返回给用户")


def main():
    """运行所有演示。"""
    print()
    print("=" * 70)
    print("Agent + Workflow 三大亮点特性验证")
    print("=" * 70)

    print("\n[初始化] 装配服务依赖...")
    svcs = setup_services()
    llm_status = "有 LLM" if svcs["chat_model"] else "无 LLM (规则降级模式)"
    print(f"[OK] 服务初始化完成，LLM 状态：{llm_status}")

    # 特性 1：并行工作流
    demo_parallel_workflow(svcs)

    # 特性 2：多 Agent 编排
    demo_multi_agent(svcs)

    # 特性 3：自反思循环（无需 LLM，纯逻辑验证）
    demo_self_reflection(svcs)

    print()
    print("=" * 70)
    print("全部验证完成")
    print("=" * 70)
    print()
    print("新增文件清单：")
    print("  app/graph/parallel_workflow.py  — 并行工作流")
    print("  app/agent/multi_agent.py        — 多 Agent 编排")
    print("  app/services/multi_agent_service.py — 多 Agent 服务")
    print()
    print("修改文件清单：")
    print("  app/graph/state.py              — 新增 warehouse_search_result 字段")
    print("  app/agent/agent.py              — 新增自反思循环（ReflectiveAgentRunner）")
    print("  app/api/routes/hybrid.py        — 新增 /parallel/run 和 /multi-agent/run 端点")
    print()


if __name__ == "__main__":
    main()
