#!/usr/bin/env python3
"""实时知识库演示 — 展示静态知识库 + 动态数据的融合效果。

演示场景：
  1. 查询库存背景 -> 生成库存优化建议
  2. 查询定价背景 -> 生成动态定价因子
  3. 查询客户背景 -> 生成信用评估
  4. 查询市场背景 -> 生成综合定价策略
  5. 综合分析 -> SKU 机会、客户订单匹配、方案推荐
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.realtime_knowledge_service import (
    create_demo_realtime_knowledge_service,
)
from app.services.hybrid_rag_service import HybridRAGService


def print_section(title: str):
    """打印章节标题"""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title: str):
    """打印小节标题"""
    print()
    print(f"[{title}]")
    print("-" * 80)


def demo_inventory_context():
    """演示 1：库存背景查询"""
    print_section("演示 1：库存背景查询 + 优化建议")

    realtime = create_demo_realtime_knowledge_service()
    rag = HybridRAGService(realtime)

    # 查询上海仓库的 SKU-A001（新库存）
    print_subsection("场景 A：新库存（15 天）")
    context_a = rag.query_inventory_context("WH-SH", "SKU-A001")
    print(context_a)

    # 查询广州仓库的 SKU-A001（积压库存）
    print_subsection("场景 B：积压库存（95 天）")
    context_b = rag.query_inventory_context("WH-GZ", "SKU-A001")
    print(context_b)

    # 对比分析
    print_subsection("对比分析")
    print("""
场景 A（15 天新库存）：
  -> 推荐方案：平衡或经济（库存充足）
  -> 定价：标准价格或轻微折扣
  -> 优先级：中等

场景 B（95 天积压库存）：
  -> 推荐方案：库存优化方案（强烈推荐）
  -> 定价：可给 5-10% 折扣加速清库
  -> 优先级：高（需要清库）
  -> 贬值已达 10%（继续积压成本更高）

启示：
  [OK] 同一个 SKU，在不同仓库、不同积压时间，推荐策略完全不同
  [OK] 实时知识库让系统能感知这些差异，做出最优决策
  [OK] 静态的"库存优化方案"规则 + 动态的库存数据 = 智能决策
""")


def demo_pricing_context():
    """演示 2：定价背景查询"""
    print_section("演示 2：定价背景 + 动态因子")

    realtime = create_demo_realtime_knowledge_service()
    rag = HybridRAGService(realtime)

    # 查询 SKU-A001 定价（标准价格）
    print_subsection("SKU-A001 定价分析（标准）")
    pricing_a = rag.query_pricing_context("SKU-A001")
    print(pricing_a)

    # 查询 SKU-D004 定价（库存积压，已降价）
    print_subsection("SKU-D004 定价分析（库存积压）")
    pricing_d = rag.query_pricing_context("SKU-D004")
    print(pricing_d)

    # 动态定价因子
    print_subsection("动态定价因子计算")
    factor = realtime.get_dynamic_pricing_factor()
    print(f"""
当前市场状况：
  - 库存使用率：78%（黄色/紧张）
  - 现金流状况：正常
  - 竞争级别：中等

动态因子计算：
  基础：1.0
  库存紧张（78%）：× 0.95（降价 5%）
  ─────────────────────
  最终因子：{factor:.2f}

应用示例：
  SKU-A001：$50 × {factor:.2f} = ${50 * factor:.2f}（降价 ${50 - 50 * factor:.2f}）
  SKU-D004：$3 × {factor:.2f} = ${3 * factor:.2f}（已包含库存因子）

启示：
  [OK] 定价不是固定的，而是根据市场实时调整
  [OK] 库存压力 + 现金流 + 竞争 等多个因素影响最终价格
  [OK] 实时知识库让定价自动响应市场变化
""")


def demo_customer_context():
    """演示 3：客户背景查询"""
    print_section("演示 3：客户背景 + 信用评估")

    realtime = create_demo_realtime_knowledge_service()
    rag = HybridRAGService(realtime)

    # 查询 VIP 客户
    print_subsection("VIP 客户（高信用）")
    profile_vip = rag.query_customer_context("CUST-VIP-001")
    print(profile_vip)

    # 查询新客户
    print_subsection("新客户（低信用）")
    profile_new = rag.query_customer_context("CUST-NEW-001")
    print(profile_new)

    # 综合分析
    print_subsection("订单处理建议对比")
    print("""
VIP 客户订单（$60,000）：
  [OK] 信用评分 99（极好）
  [OK] 120 个订单历史（稳定）
  [OK] 按时付款率 100%（最好）

  处理：
    - 接受标准信用条件（无预付）
    - 给予 VIP 折扣政策
    - 优先级最高，快速处理
    - 可考虑赊账 30-60 天

新客户订单（$5,000）：
  [WARN] 信用评分 65（较差）
  [WARN] 仅 2 个订单历史（新）
  [WARN] 按时付款率 90%（有风险）

  处理：
    - 要求 30% 预付（风险防控）
    - 可给 5% 折扣鼓励预付
    - 背景调查（可选）
    - 严密跟进收款状态

启示：
  [OK] 相同金额的订单，客户不同，处理方式完全不同
  [OK] 实时知识库让系统能根据客户信用做出差异化决策
  [OK] 自动化的信用评分 + 人工审批相结合
""")


def demo_market_context():
    """演示 4：市场背景查询"""
    print_section("演示 4：市场背景 + 综合定价策略")

    realtime = create_demo_realtime_knowledge_service()
    rag = HybridRAGService(realtime)

    market = rag.query_market_context()
    print(market)

    print_subsection("当前市场状况解读")
    print("""
库存情况：使用率 78%（黄色/紧张）
  -> 不紧急，但需要关注
  -> 库存优化方案可给 3-5% 折扣
  -> 推荐经济方案加速清库

现金流：正常
  -> 可以接受标准 30 天付款期
  -> 不需要特别激进的清库折扣
  -> 保持正常利润率

竞争：中等
  -> 有竞争压力，但不紧张
  -> 可以坚持标准价格
  -> 通过服务和质量竞争

季节：Q2（常规期）
  -> 销售稳定，不是旺季也不是淡季
  -> 推荐平衡方案（性价比最优）
  -> 标准定价策略

综合策略：
  1. 新订单默认推荐平衡方案
  2. 有库存积压的 SKU 推荐库存优化 + 3-5% 折扣
  3. 成本敏感客户推荐经济方案
  4. VIP 客户保持快速方案
  5. 定价：基础价格 × 0.95（因库存紧张）
""")


def demo_comprehensive_analysis():
    """演示 5：综合分析"""
    print_section("演示 5：综合分析（SKU + 客户 + 市场）")

    realtime = create_demo_realtime_knowledge_service()
    rag = HybridRAGService(realtime)

    # 分析 SKU 机会
    print_subsection("SKU-D004 机会分析")
    sku_analysis = rag.analyze_sku_opportunity("SKU-D004")
    print(sku_analysis)

    # 分析客户订单匹配
    print_subsection("新客户 $5,000 订单匹配度分析")
    order_analysis = rag.analyze_customer_order_fit("CUST-NEW-001", 5000)
    print(order_analysis)

    # 为方案推荐生成背景
    print_subsection("VIP 客户方案推荐背景")
    recommendation = rag.get_fulfillment_recommendation_context("vip")
    print(recommendation)


def demo_key_insights():
    """演示总结：关键启示"""
    print_section("总结：实时知识库的关键启示")

    print("""
传统方式（静态）：
  [NO] 所有 SKU 用统一的定价模型
  [NO] 所有客户用统一的信用政策
  [NO] 所有订单用统一的方案推荐
  [NO] 定价规则定死了，无法适应市场变化

  结果：低效、不灵活、容易错失机会

实时知识库方式（动态）：
  [OK] 同一个 SKU，根据积压情况、库存成本灵活定价
  [OK] 同一个客户等级，根据当前信用评分、付款历史做差异化处理
  [OK] 同一个订单，根据库存压力、现金流状况推荐不同方案
  [OK] 定价和策略实时响应市场，自动调整

  结果：高效、灵活、智能决策

三层融合的威力：
  Layer 1（实时数据）
    └─ 库存、价格、客户信息、市场条件

  Layer 2（静态知识）
    └─ 业务规则、决策逻辑、最佳实践

  Layer 3（AI 合成）
    └─ Agent/LLM 根据实时 + 静态，做出最优决策

具体例子：

  场景：新客户，$5,000 订单，SKU-D004（库存积压）

  仅静态规则：
    "新客户 -> 经济方案 -> 标准价格"

  有实时知识库：
    "新客户 + 成本敏感（score 3/10）
     + SKU-D004 库存 45 天（积压）
     + 仓库使用率 78%（紧张）
     + 现金流正常

     -> 推荐库存优化方案
     -> 基价 $3 × 动态因子 0.95 × 库存积压折扣 0.95 = $2.71
     -> 建议预付 30%
     -> 立即发货（优先清库）"

  效果：
    - 价格从 $3 -> $2.71（客户满意）
    - 及时清库存（节省仓储成本）
    - 获得现金流（改善财务）
    - 建立新客户关系（为后续升级铺路）

结论：
  [OK] 实时知识库 = 竞争优势
  [OK] 自动适应市场变化
  [OK] 每个决策都是最优的，而不是平均的
  [OK] 成本优化 + 收入增长 = 双赢
""")


def main():
    """运行演示"""
    print()
    print("=" * 80)
    print("实时知识库演示")
    print("=" * 80)
    print()

    demo_inventory_context()
    demo_pricing_context()
    demo_customer_context()
    demo_market_context()
    demo_comprehensive_analysis()
    demo_key_insights()

    print()
    print("=" * 80)
    print("演示完成")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
