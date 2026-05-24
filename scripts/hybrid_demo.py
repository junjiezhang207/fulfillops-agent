#!/usr/bin/env python3
"""混合策略演示脚本。

展示混合路由如何根据问题复杂度自动选择 Agent 或 Workflow。

运行方式：
  python -m scripts.hybrid_demo
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.infrastructure.llm.chat_adapter import LLMFactory
from app.repositories.file_system_knowledge_repository import (
    FileSystemKnowledgeRepository,
)
from app.repositories.in_memory_inventory_repository import (
    InMemoryInventoryRepository,
)
from app.repositories.in_memory_order_repository import InMemoryOrderRepository
from app.agents.runtime.agent_service import AgentService
from app.domain.fulfillment.plan_service import FulfillmentPlanService
from app.application.routing.hybrid_service import HybridService
from app.application.routing.intent_classifier import get_classifier
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.orders.analysis import OrderAnalysisService
from app.application.memory.session_memory_service import get_session_service
from app.domain.fulfillment.substitute_sku import SubstituteSkuService
from app.domain.inventory.warehouse_service import WarehouseService
from app.application.workflow.workflow_service import WorkflowService

from app.agents.tools.factory import (
    make_fulfillment_plan_tool,
    make_substitute_tool,
    make_warehouse_tool,
)


def create_test_cases():
    """创建测试用例。"""
    return [
        {
            "order_id": "SO202502140001",
            "question": "订单 SO202502140001 是什么？",
            "expected_path": "workflow",
            "description": "Simple query - should use Workflow",
        },
        {
            "order_id": "SO202502140001",
            "question": "订单 SO202502140001 的库存充足吗？",
            "expected_path": "workflow",
            "description": "Inventory check - should use Workflow",
        },
        {
            "order_id": "SO202502140002",
            "question": "订单 SO202502140002 的库存够吗？如果不够有替代方案吗？",
            "expected_path": "agent",
            "description": "Multi-step analysis - should use Agent",
        },
        {
            "order_id": "SO202502140003",
            "question": "请为订单 SO202502140003 生成一个完整的履约方案。",
            "expected_path": "agent",
            "description": "Complex decision - should use Agent",
        },
    ]


def main():
    """运行混合策略演示。"""
    print("=" * 80)
    print("Hybrid Strategy Demo - Agent + Workflow Dynamic Routing")
    print("=" * 80)
    print()

    # 初始化分类器
    print("[*] Initializing intent classifier...")
    classifier = get_classifier()
    print("[OK] Classifier ready\n")

    # 初始化服务
    print("[*] Initializing services...")
    settings = get_settings()

    order_repo = InMemoryOrderRepository()
    order_service = OrderAnalysisService(order_repo)

    inventory_repo = InMemoryInventoryRepository()
    inventory_service = InventoryAnalysisService(
        inventory_repository=inventory_repo,
        order_analysis_service=order_service,
    )

    knowledge_repo = FileSystemKnowledgeRepository(settings.knowledge_dir)
    knowledge_service = KnowledgeRetrievalService(
        knowledge_repository=knowledge_repo,
        inventory_analysis_service=inventory_service,
    )

    workflow_service = WorkflowService(
        order_service=order_service,
        inventory_service=inventory_service,
        knowledge_service=knowledge_service,
    )

    # 初始化 Agent
    chat_model = LLMFactory.create_chat_model(settings)
    if chat_model is None:
        print("[FAIL] Agent not configured. Please set LLM in .env")
        return

    warehouse_service = WarehouseService()
    substitute_service = SubstituteSkuService()
    fulfillment_service = FulfillmentPlanService(
        inventory_service=inventory_service,
        warehouse_service=warehouse_service,
        substitute_service=substitute_service,
    )

    extra_tools = [
        make_warehouse_tool(warehouse_service),
        make_substitute_tool(substitute_service),
        make_fulfillment_plan_tool(fulfillment_service),
    ]

    agent_service = AgentService(
        order_service=order_service,
        inventory_service=inventory_service,
        knowledge_service=knowledge_service,
        chat_model=chat_model,
        extra_tools=extra_tools,
    )

    # 创建会话缓存和混合服务
    session_cache = get_session_service()
    hybrid_service = HybridService(
        workflow_service=workflow_service,
        agent_service=agent_service,
        session_cache=session_cache,
    )
    print("[OK] Hybrid service ready")
    print("[OK] Session memory ready\n")

    # 运行测试用例
    test_cases = create_test_cases()
    print(f"[*] Running {len(test_cases)} test cases...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {test_case['description']}")
        print(f"{'=' * 80}")
        print(f"Order ID: {test_case['order_id']}")
        print(f"Question: {test_case['question']}")

        # 第一步：意图分类
        classification = classifier.classify(test_case["question"])
        print(f"\n[Intent Classification]")
        print(f"  Level: {classification.level.value}")
        print(f"  Confidence: {classification.score:.1%}")
        print(f"  Keywords: {', '.join(classification.primary_keywords)}")
        print(f"  Reasoning: {classification.reasoning}")

        # 第二步：混合处理
        print(f"\n[Hybrid Processing]")
        try:
            result = hybrid_service.process(
                order_id=test_case["order_id"],
                question=test_case["question"],
                thread_id=f"session-{test_case['order_id']}",  # 每个订单一个会话
            )

            path_match = "[OK]" if result.path_used == test_case["expected_path"] else "[MISMATCH]"
            print(f"  Path Used: {result.path_used} {path_match}")
            print(f"  Expected: {test_case['expected_path']}")
            print(f"  Execution Time: {result.execution_time_ms:.0f}ms")
            print(f"  Conversation Turns: {result.conversation_turns}")
            print(f"  Tools Called: {', '.join(result.tools_called) if result.tools_called else 'None'}")

            print(f"\n[Answer Preview]")
            answer_preview = result.final_answer[:150] + ("..." if len(result.final_answer) > 150 else "")
            print(f"  {answer_preview}")

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            import traceback
            traceback.print_exc()

    # 演示短期记忆 — 多轮对话
    print("\n\n" + "=" * 80)
    print("Session Memory Demo - Multi-turn Conversation")
    print("=" * 80)
    print()

    session_id = "demo-session-001"
    print(f"[*] Creating session: {session_id}\n")

    demo_turns = [
        "订单 SO202502140001 是什么？",
        "库存充足吗？",
        "如果库存不足有替代方案吗？",
    ]

    for turn_num, question in enumerate(demo_turns, 1):
        print(f"[Turn {turn_num}] Question: {question}")
        try:
            result = hybrid_service.process(
                order_id="SO202502140001",
                question=question,
                thread_id=session_id,
            )
            print(f"  Response: {result.final_answer[:100]}...")
            print(f"  Turns in session: {result.conversation_turns}")
            print()
        except Exception as exc:
            print(f"  [ERROR] {exc}\n")

    # 显示会话统计
    session_stats = session_cache.get_stats()
    print(f"[Session Statistics]")
    print(f"  Active sessions: {session_stats['active_sessions']}")
    print(f"  Session TTL: {session_stats['ttl_hours']} hours")
    if session_stats['sessions']:
        for sess in session_stats['sessions']:
            print(f"  - {sess['thread_id']}: {sess['turns']} turns")

    # 生成报告
    print("\n\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Hybrid routing strategy successfully demonstrated:")
    print("  • Simple questions detected and routed to Workflow (fast)")
    print("  • Complex questions detected and routed to Agent (intelligent)")
    print("  • Unified response format for both paths")
    print("  • Session memory tracking multi-turn conversations")
    print()
    print("Next Steps:")
    print("  1. Monitor actual traffic patterns")
    print("  2. Adjust classification thresholds if needed")
    print("  3. A/B test different threshold combinations")
    print("  4. Add metrics collection for cost/performance analysis")
    print("  5. Migrate to Redis for scalability")


if __name__ == "__main__":
    main()
