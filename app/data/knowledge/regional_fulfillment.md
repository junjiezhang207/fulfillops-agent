---
category: regional_strategy
title: 区域仓配策略
owner: fulfillment-ops
version: v1
effective_date: 2026-01-01
business_scope: [区域仓配, 跨区域履约]
region: all
---

# 区域仓配策略

区域仓配策略用于辅助判断订单的最佳履约路径。

1. 华东订单优先匹配上海、杭州等华东仓。
2. 华南订单优先匹配广州、深圳等华南仓。
3. 华北订单优先匹配北京、天津等华北仓。
4. 如果本区域库存不足，再考虑邻近区域调拨。
5. 当跨区域履约发生时，需要在说明中明确提示时效和成本影响。
