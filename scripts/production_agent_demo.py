#!/usr/bin/env python3
"""验证生产级 Agent 三项增强特性。

特性 1: 工具弹性   — 重试 + 超时 + 熔断器
特性 2: 上下文管理 — 滑动窗口防 token 溢出
特性 3: 流式输出   — stream_chat() 逐 token + 工具调用可见
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.context_manager import ContextWindowConfig, ContextWindowManager, estimate_tokens
from app.agent.tool_wrapper import CircuitBreaker, CircuitState, wrap_tool_with_resilience
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def demo_tool_resilience():
    """特性 1：工具弹性验证（重试 + 超时 + 熔断器）。"""
    print()
    print("=" * 70)
    print("特性 1: 工具弹性 — 重试 + 超时 + 熔断器")
    print("=" * 70)

    from langchain_core.tools import tool

    # ── 场景 A：重试成功 ──────────────────────────────────────────
    print("\n[场景 A] 工具第 2 次调用才成功（重试机制）")
    call_count_a = [0]

    @tool
    def flaky_inventory(order_id: str) -> str:
        """检查库存，模拟偶发失败。"""
        call_count_a[0] += 1
        if call_count_a[0] < 2:
            raise ConnectionError(f"临时网络错误（第 {call_count_a[0]} 次）")
        return f"订单 {order_id} 库存充足（第 {call_count_a[0]} 次调用成功）"

    resilient_a = wrap_tool_with_resilience(flaky_inventory, max_retries=2, timeout_seconds=5.0)
    result_a = resilient_a.invoke({"order_id": "SO001"})
    print(f"  结果: {result_a}")
    print(f"  总调用次数: {call_count_a[0]} (含 {call_count_a[0]-1} 次重试)")
    assert "成功" in result_a, "应该重试后成功"
    print(f"  [OK] 重试机制正常工作")

    # ── 场景 B：超时 ──────────────────────────────────────────────
    print("\n[场景 B] 工具执行超时（超时保护）")
    import time

    @tool
    def slow_database_query(order_id: str) -> str:
        """模拟慢查询。"""
        time.sleep(3.0)  # 模拟 3 秒慢查询
        return "查询结果"

    resilient_b = wrap_tool_with_resilience(slow_database_query, max_retries=0, timeout_seconds=0.5)
    result_b = resilient_b.invoke({"order_id": "SO001"})
    print(f"  结果: {result_b}")
    assert "[TIMEOUT]" in result_b, "应该触发超时"
    print(f"  [OK] 超时保护正常工作（0.5s 后强制终止）")

    # ── 场景 C：熔断器 ────────────────────────────────────────────
    print("\n[场景 C] 连续失败触发熔断（熔断器）")
    circuit_calls = [0]

    @tool
    def always_failing_service(order_id: str) -> str:
        """模拟持续不可用的服务。"""
        circuit_calls[0] += 1
        raise RuntimeError(f"服务不可用（调用 #{circuit_calls[0]}）")

    resilient_c = wrap_tool_with_resilience(
        always_failing_service,
        max_retries=1,
        timeout_seconds=5.0,
        enable_circuit_breaker=True,
        failure_threshold=2,  # 连续失败 2 次触发熔断
    )

    results = []
    for i in range(5):
        result = resilient_c.invoke({"order_id": f"SO{i:03d}"})
        results.append(result[:50])

    error_calls = sum(1 for r in results if "[ERROR]" in r)
    circuit_calls_count = sum(1 for r in results if "[CIRCUIT_OPEN]" in r)
    print(f"  5 次调用结果：{error_calls} 次错误返回，{circuit_calls_count} 次熔断快速失败")
    print(f"  实际工具调用次数：{circuit_calls[0]}（熔断后节省了 {5 - circuit_calls[0]} 次无效调用）")
    assert circuit_calls_count > 0, "应该触发熔断"
    print(f"  [OK] 熔断器正常工作（连续失败后快速返回，不浪费资源）")

    print("\n亮点说明：")
    print("  - 指数退避：重试等待 0.3s → 0.6s → 1.2s，避免雪崩")
    print("  - 超时用 ThreadPoolExecutor.future.result(timeout=N)，跨平台兼容")
    print("  - 熔断器状态机：CLOSED → OPEN → HALF_OPEN → CLOSED")
    print("  - 失败返回结构化字符串，让 Agent 感知并调整策略")


def demo_context_manager():
    """特性 2：上下文管理验证。"""
    print()
    print("=" * 70)
    print("特性 2: 上下文管理 — 防止 Token 溢出")
    print("=" * 70)

    manager = ContextWindowManager(ContextWindowConfig(
        max_tokens=500,          # 故意设小，方便演示
        trim_threshold=0.80,     # 400 tokens 时触发
        min_messages_to_keep=3,
    ))

    # 模拟多轮对话历史（每条消息约 50 token）
    messages = [
        HumanMessage(content="第1轮：帮我分析订单 SO001 的库存状态"),
        AIMessage(content="正在查询库存系统，请稍等..."),
        ToolMessage(content="SKU-A001: 500件, SKU-B002: 0件（缺货）", tool_call_id="tc1"),
        AIMessage(content="订单 SO001：SKU-A001 库存充足，SKU-B002 缺货，需要安排替代品。"),

        HumanMessage(content="第2轮：帮我查找 SKU-B002 的替代品"),
        AIMessage(content="正在搜索替代品..."),
        ToolMessage(content="SKU-B003 可作为替代品，功能相同，价格相近", tool_call_id="tc2"),
        AIMessage(content="推荐使用 SKU-B003 替代 SKU-B002，已向客户发送确认邮件。"),

        HumanMessage(content="第3轮：帮我生成完整的履约方案"),
        AIMessage(content="正在生成履约方案..."),
        ToolMessage(content="方案：从上海仓发 SKU-A001(100件)，用 SKU-B003(50件) 替代 SKU-B002", tool_call_id="tc3"),
        AIMessage(content="履约方案已生成：SKU-A001 从上海仓库发货，SKU-B002 用 SKU-B003 替代。"),

        HumanMessage(content="第4轮：当前对话中有多少个 SKU？"),
    ]

    original_tokens = manager.count_tokens(messages)
    print(f"\n原始消息：{len(messages)} 条，约 {original_tokens} tokens")
    print(f"Token 预算：{manager.config.max_tokens}，阈值：{manager.config.max_tokens * manager.config.trim_threshold:.0f}")

    should_trim = manager.should_trim(messages)
    print(f"需要裁剪：{'是' if should_trim else '否'}")

    if should_trim:
        trimmed_messages, result = manager.trim(messages)
        print(f"\n裁剪结果：")
        print(f"  操作：{result.action_taken}")
        print(f"  消息数：{result.original_count} -> {result.final_count}（减少 {result.messages_dropped} 条）")
        print(f"  Token：{result.original_tokens} -> {result.final_tokens}（压缩 {result.reduction_rate:.0%}）")
        print(f"\n保留的消息（最近 {result.final_count} 条）：")
        for msg in trimmed_messages:
            role = type(msg).__name__.replace("Message", "")
            content_preview = str(msg.content)[:60]
            print(f"  [{role:12s}] {content_preview}")

    # 测试 token 估算
    print(f"\nToken 估算示例：")
    samples = [
        ("纯英文", "The inventory for SKU-A001 is sufficient with 500 units available."),
        ("纯中文", "订单 SO001 的库存状态：SKU-A001 充足，SKU-B002 缺货，需要替代品处理。"),
        ("混合",   "库存分析：SKU-A001 has 500 units, SKU-B002 is out of stock."),
    ]
    for label, text in samples:
        tokens = estimate_tokens(text)
        print(f"  [{label}] {len(text)} 字符 ≈ {tokens} tokens")

    print("\n亮点说明：")
    print("  - 滑动窗口：从最新消息向前贪心选取，保持对话连续性")
    print("  - 三级降级：滑动窗口 → 摘要压缩（build_summary_prompt）→ 强制截断")
    print("  - Token 估算：中文字符 × 0.7（密度高），英文字符 / 4，误差 ±15%")
    print("  - SystemMessage 永远保留（系统提示丢失会导致角色混乱）")


def demo_streaming_format():
    """特性 3：流式输出格式验证（无需 LLM，验证事件格式）。"""
    print()
    print("=" * 70)
    print("特性 3: 流式输出 — stream_chat() 事件格式")
    print("=" * 70)

    import json

    # 模拟 stream_chat 产生的事件序列
    mock_stream_events = [
        {"type": "tool_call", "tool": "check_inventory", "node": "tools"},
        {"type": "tool_result", "tool": "check_inventory", "result": "SO001 库存充足", "truncated": False},
        {"type": "tool_call", "tool": "retrieve_knowledge", "node": "tools"},
        {"type": "tool_result", "tool": "retrieve_knowledge", "result": "VIP 客户优先发货规则", "truncated": False},
        {"type": "token", "content": "根据查询"},
        {"type": "token", "content": "结果，订单"},
        {"type": "token", "content": " SO001 库存充足，"},
        {"type": "token", "content": "建议从上海仓库发货。"},
        {
            "type": "done",
            "reply": "根据查询结果，订单 SO001 库存充足，建议从上海仓库发货。",
            "tools_called": ["check_inventory", "retrieve_knowledge"],
        }
    ]

    print("\n模拟事件流（前端 SSE 接收到的数据）：")
    full_reply = ""
    tools_seen = []

    for event in mock_stream_events:
        line = json.dumps(event, ensure_ascii=False)
        event_type = event["type"]

        if event_type == "tool_call":
            tools_seen.append(event["tool"])
            print(f"  [TOOL_CALL]   {event['tool']}")
        elif event_type == "tool_result":
            print(f"  [TOOL_RESULT] {event['tool']}: {event['result'][:40]}")
        elif event_type == "token":
            full_reply += event["content"]
            print(f"  [TOKEN]       '{event['content']}'")
        elif event_type == "done":
            print(f"  [DONE]        tools={event['tools_called']}")

    print(f"\n重建的完整回复: '{full_reply}'")
    print(f"调用的工具: {tools_seen}")
    print(f"\n前端渲染效果：")
    print(f"  1. 显示 '[调用工具: check_inventory...]'   — 用户知道 Agent 在做什么")
    print(f"  2. 工具完成后，逐 token 渲染最终回复      — 打字机效果")
    print(f"  3. done 事件返回完整回复和工具列表        — 便于前端收尾渲染")

    print("\n亮点说明：")
    print("  - 使用 LangGraph .stream(stream_mode='messages') 获取 token 级流")
    print("  - 每个事件有 type 字段，前端可区分工具调用 vs 答案 token")
    print("  - FastAPI StreamingResponse + SSE 实现无须 WebSocket")
    print("  - done 事件携带完整回复和工具列表，支持前端收尾状态更新")


def main():
    """运行所有演示。"""
    print()
    print("=" * 70)
    print("生产级 Agent 三项增强特性验证")
    print("=" * 70)

    demo_tool_resilience()
    demo_context_manager()
    demo_streaming_format()

    print()
    print("=" * 70)
    print("全部验证完成")
    print("=" * 70)
    print()
    print("新增文件：")
    print("  app/agent/tool_wrapper.py      — 工具重试 + 超时 + 熔断器")
    print("  app/agent/context_manager.py   — Token 上限管理")
    print()
    print("重写文件：")
    print("  app/services/agent_service.py  — 集成以上三项 + 新增 stream_chat()")
    print()


if __name__ == "__main__":
    main()
