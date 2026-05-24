# 知识库目录 — 供应链决策系统

> 为 RAG 系统提供业务规则、市场知识和决策上下文

---

## 📚 知识库结构

### 1️⃣ 业务规则 (Business Rules)

#### [customer_tiers.md](business_rules/customer_tiers.md) — 客户分级政策
**内容：** 4 个客户等级的完整政策

包括：
- **VIP 客户**：优先级 8、次日达、禁止合并、超级质量
- **一级客户**：优先级 7、快递、有条件合并、高级质量
- **二级客户**：优先级 5、标准物流、允许合并、标准质量
- **新客户**：优先级 3、经济物流、鼓励合并、标准质量

应用场景：
```
用户问：为什么 VIP 客户的订单不能合并？
RAG 检索：查询 customer_tiers.md → VIP 合并政策章节
回答：VIP 客户要求无延迟交期，合并会导致延迟，违反承诺
```

**相关代码：** [business_rule_engine.py](../app/services/business_rule_engine.py)

---

#### [fulfillment_strategies.md](business_rules/fulfillment_strategies.md) — 4 种履约方案
**内容：** 每种方案的特征、成本、质量、推荐场景

4 个方案：
- **快速方案 (FAST_TRACK)**：26h，$3,650，99% 质量
- **平衡方案 (MIXED)** ⭐：76h，$2,190，96% 质量
- **经济方案 (ECONOMY)**：174h，$1,445，92% 质量
- **库存优化 (INVENTORY_OPTIMIZATION)**：76h，节省仓储费用

应用场景：
```
用户问：为什么推荐平衡方案而不是快速方案？
RAG 检索：fulfillment_strategies.md → 方案对比章节
回答：这个订单是新客户成本敏感，成本权重 50%
      平衡方案性价比最高（比快速便宜 40%，只慢 2 天）
```

**相关代码：** [fulfillment_options_generator.py](../app/services/fulfillment_options_generator.py)

---

#### [risk_assessment.md](business_rules/risk_assessment.md) — 风险评估指南
**内容：** 7 种风险标签、风险等级、自动化缓释措施

风险标签：
- `high_value` (金额 > $100K)：强制检查、购买保险
- `critical_deadline` (交期 < 24h)：升至最高优先级、准备备用方案
- `fragile_handling_required`：专业包装、高质量仓库
- `out_of_stock_SKU`：协商替代或延期
- `urgent_high_value`：全套缓释（检查、保险、追踪、专人）
- `approaching_budget`：推荐经济方案
- `new_customer_credit_risk`：要求预付款、背景调查

应用场景：
```
订单特征：VIP 高价值急单 + 易碎品

系统识别风险标签：
  high_value, critical_deadline, fragile_handling_required, urgent_high_value

RAG 检索：risk_assessment.md → urgent_high_value 章节
回答：这个订单需要全套缓释措施：
      ✓ 使用最优仓库（质量 98.5%）
      ✓ 强制 OVERNIGHT 物流
      ✓ 100% 质量检查
      ✓ 购买 $2,000 保险
      ✓ 专业易碎品包装
      ✓ 指派专人监控
```

**相关代码：** [business_rule_engine.py](../app/services/business_rule_engine.py)

---

### 2️⃣ SKU 管理 (SKU Management)

#### [sku_properties.md](sku_management/sku_properties.md) — SKU 特性和处理指南
**内容：** 常见 SKU 的特性、存储要求、适用方案、替代品

SKU 分类：
- **电子产品**：SKU-A001（主板模块）、SKU-B002（电源）
  - 特性：易静电、防潮、需要防静电包装
  - 禁用方案：ECONOMY（质量风险太大）
  
- **日用消费品**：SKU-C003（清洁剂）
  - 特性：液体、防泄漏、周转快
  - 推荐方案：库存优化（周转快）
  
- **食品相关**：SKU-D004（食品添加剂）
  - 特性：需要冷链、保质期短（12 个月）
  - 推荐方案：快速方案（保证新鲜）
  
- **特殊品类**：易碎品、危险品、冷链产品
  - 处理规则：专业标准、额外成本、质量要求最高

SKU 替代品：
```
SKU-A001 的替代品：
  ✓ SKU-A002：完全替代（同功能）
  ✓ SKU-A003：高级替代（功能升级，+10% 价格）
```

库存老化模型：
```
30-60 天：5% 贬值
61-90 天：10% 贬值
91-180 天：20% 贬值
181-365 天：50% 贬值
365+ 天：80% 贬值

应对：超过 60 天立即推库存优化方案
```

应用场景：
```
用户问：SKU-D004 能用 ECONOMY 方案吗？
RAG 检索：sku_properties.md → SKU-D004 章节
回答：不建议。SKU-D004 是食品添加剂，需要冷链保护
      ECONOMY 物流太慢（7-14 天），会导致变质
      推荐快速方案（保证 26 小时内送达）
```

**相关代码：** [advanced_order.py](../app/schemas/advanced_order.py)

---

### 3️⃣ 市场上下文 (Market Context)

#### [shipping_partners.md](market_context/shipping_partners.md) — 配送合作商指南
**内容：** 5 家物流商的特性、成本、质量、适用场景

物流商对比：

| 物流商 | 类型 | 优势 | 劣势 | 成本 |
|-------|------|------|------|------|
| **顺丰** | 高端快递 | 质量最好（0.2% 破损） | 最贵（×2） | 次日达：基准×2 |
| **德邦** | 综合 | 覆盖最全、中等质量 | 覆盖面有限 | 次日达：基准×1.5 |
| **圆通** | 大众快递 | 成本低、覆盖广 | 质量一般、延误高 | 经济：基准×0.6 |
| **中通** | 大众快递 | 最便宜 | 质量不稳定 | 经济：基准×0.55 |
| **顺丰冷运** | 冷链 | 唯一冷链选择 | 最贵、覆盖有限 | 冷链：+40-60% |

自动选择逻辑：
```
VIP + 急单 → 顺丰（不可协商）
高价值 → 顺丰（质量最重要）
普通急单 → 德邦（成本vs质量平衡）
成本敏感 → 圆通/中通（最便宜）
冷链 → 顺丰冷运（唯一选择）
```

应用场景：
```
用户问：为什么这个订单选择顺丰而不是圆通？
RAG 检索：shipping_partners.md → 物流商对比章节
回答：订单金额 $80,000（高价值）且客户等级 VIP
      顺丰破损率 0.2%，圆通破损率 1%
      如果出现 1% 的破损风险，损失 $800
      顺丰加价 $300，但防止 $800 的风险，值得
```

**相关代码：** [fulfillment_options_generator.py](../app/services/fulfillment_options_generator.py)

---

#### [market_conditions.md](market_context/market_conditions.md) — 市场条件和定价策略
**内容：** 季节性、库存压力、现金流、竞争环境对决策的影响

季节性定价：

| 季节 | 销售额 | 库存 | 定价策略 | 方案推荐 |
|-----|-------|------|---------|---------|
| Q4（10-12月） | 35% | 充足 | 标准价格 | 经济 + 库存优化 |
| Q1（1-3月） | 20% | 紧张 | 现金折扣 5% | 快速（保时效） |
| Q2/Q3 | 22.5%×2 | 适中 | 标准价格 | 平衡（推荐） |

库存压力指标：
```
绿色（< 70%）：库存优化方案给 3-5% 折扣
黄色（70-85%）：推荐经济方案、库存优化
红色（85-100%）：强烈推荐经济方案、可降价 5-10%
黑色（100%）：无法履约、紧急调货或拒单
```

现金流影响：
```
现金充裕：坚持标准价格
现金困难：新客户预付 30% 给 5% 折扣
库存爆满：即使降价也要清库（换取现金周转）
```

应用场景：
```
用户问：为什么 Q1 新客户能得到 5% 折扣？
RAG 检索：market_conditions.md → Q1 淡季章节
回答：Q1 是全年最淡季，库存压力和现金流是主要问题
      新客户给 5% 折扣 + 要求预付，可以：
      ✓ 获得订单（销售额提升）
      ✓ 加快现金周转（急需）
      ✓ 建立客户关系（为后续服务）
```

**相关代码：** [advanced_hybrid_service.py](../app/services/advanced_hybrid_service.py)

---

## 🔄 RAG 系统如何使用这些文档

### 信息检索流程

```
用户问题
  ↓
意图理解（问的是什么？）
  ├─ 关于规则 → business_rules/*
  ├─ 关于 SKU → sku_management/*
  ├─ 关于物流 → market_context/shipping_partners.md
  └─ 关于定价 → market_context/market_conditions.md
  ↓
RAG 检索（相关知识库章节）
  ↓
上下文补充（加入订单信息、市场条件等）
  ↓
生成回答（解释为什么、如何处理）
```

### 实际案例

**案例 1：解释规则应用**

用户问："为什么这个订单需要 100% 质量检查？"

```
系统分析：
  订单特征 → high_value, fragile_handling_required, quality_requirement=SUPER

RAG 检索：
  1. customer_tiers.md → VIP 质量政策
  2. risk_assessment.md → high_value, fragile_handling_required 缓释措施
  3. sku_properties.md → 易碎品处理规则

生成回答：
"这个订单有 3 个理由需要 100% 检查：
  1. VIP 客户：要求 SUPER 质量，必须检查
  2. 高价值：订单 $100K+，风险大，必须检查
  3. 易碎品：破损风险高，必须检查
     
  任何一个理由就足够了，三个理由堆叠，检查必须是 100%"
```

**案例 2：推荐方案**

用户问："为什么推荐这个方案而不是那个方案？"

```
系统分析：
  订单：新客户、成本敏感、交期 7 天、$5,000

RAG 检索：
  1. customer_tiers.md → 新客户政策（优先成本）
  2. fulfillment_strategies.md → 方案对比和推荐逻辑
  3. market_conditions.md → 当前库存和现金流状况

生成回答：
"推荐平衡方案（$2,190）而不是快速（$3,650）的原因：
  1. 客户是新客户，成本权重 50%（费用最敏感）
  2. 成本对比：平衡省 $1,460（40% 省钱）
  3. 时间对比：多等 2 天（不影响 7 天交期）
  4. 质量对比：差异 < 1%（都很好）
  5. 综合评分：平衡方案 72.8 vs 快速 54.8
  
  综合来看，平衡方案性价比最优"
```

**案例 3：解释市场条件影响**

用户问："为什么同样的订单在 Q4 和 Q1 的处理不同？"

```
RAG 检索：
  market_conditions.md → Q4 vs Q1 对比章节

生成回答：
"Q4（旺季）vs Q1（淡季）的不同处理：

Q4（库存充足 + 现金充裕）：
  • 推荐经济方案（有库存空间）
  • 标准价格（有定价权）
  • 新客户 3% 折扣（建立关系）

Q1（库存紧张 + 现金困难）：
  • 推荐快速方案（保证时效，赢得好评）
  • 新客户 5% 折扣（需要现金周转）
  • 要求预付 30%（加快收款）

核心差异：Q4 看利润，Q1 看现金流"
```

---

## 🎯 常见 RAG 查询场景

| 查询场景 | 对应文档 | 预期回答内容 |
|---------|---------|-----------|
| 为什么这个客户不能合并订单？ | customer_tiers.md | 客户等级的合并政策 |
| 为什么推荐这个履约方案？ | fulfillment_strategies.md | 方案的适用场景和评分依据 |
| 这个 SKU 能用经济方案吗？ | sku_properties.md | SKU 特性和对方案的适用性 |
| 为什么需要购买保险？ | risk_assessment.md | 风险标签和缓释措施 |
| 为什么选择这个物流商？ | shipping_partners.md | 物流商特性和成本对比 |
| 为什么现在给折扣？ | market_conditions.md | 季节性、库存、现金流影响 |

---

## 📊 知识库与系统的映射

```
AdvancedOrderDetails 订单对象
  ├─ customer_level (VIP/一级/二级/新客)
  │   └─ RAG 查询：customer_tiers.md
  │
  ├─ shipping_method (OVERNIGHT/STANDARD/ECONOMY)
  │   └─ RAG 查询：shipping_partners.md
  │
  ├─ line_items (SKU 列表)
  │   └─ RAG 查询：sku_properties.md
  │
  ├─ risk_flags (high_value/critical_deadline/等)
  │   └─ RAG 查询：risk_assessment.md
  │
  └─ special_handling (易碎/防潮/等)
      └─ RAG 查询：sku_properties.md

FulfillmentOption 方案对象
  ├─ strategy (FAST_TRACK/MIXED/ECONOMY/等)
  │   └─ RAG 查询：fulfillment_strategies.md
  │
  ├─ shipping_method
  │   └─ RAG 查询：shipping_partners.md
  │
  └─ overall_score 评分依据
      └─ RAG 查询：fulfillment_strategies.md 的评分逻辑

当前市场状况
  └─ RAG 查询：market_conditions.md
     └─ 影响：季节性、定价、方案推荐
```

---

## 💡 如何扩展知识库

当系统需要支持新的业务场景时：

1. **新增客户等级**：在 customer_tiers.md 中添加新的等级定义
2. **新增 SKU 类型**：在 sku_properties.md 中添加新的特性
3. **新增物流商**：在 shipping_partners.md 中补充
4. **季节性调整**：在 market_conditions.md 中更新定价策略

---

## 📝 维护说明

**更新频率：**
- customer_tiers.md：年 1-2 次（客户政策变更）
- fulfillment_strategies.md：年 2-4 次（方案优化）
- risk_assessment.md：按需（新风险识别）
- sku_properties.md：月 1 次（SKU 变更）
- shipping_partners.md：月 2 次（成本、覆盖变化）
- market_conditions.md：实时（市场条件变化）

**更新责任人：**
- 业务规则：供应链经理
- SKU 特性：仓库管理员
- 物流信息：物流协调员
- 市场条件：财务和销售

---

**最后更新：** 2026-04-26  
**版本：** 1.0  
**状态：** 生产就绪
