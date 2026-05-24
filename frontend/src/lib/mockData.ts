import type { StatusTone } from "./status";

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type RouteMode = "Hybrid" | "Workflow" | "Agent" | "RAG" | "Multi-Agent";
export type InventoryState = "充足" | "部分缺货" | "缺货" | "锁库中";

export type MockOrder = {
  orderId: string;
  platform: "Shopify" | "Amazon" | "TikTok Shop" | "ERP";
  scenario: string;
  customerLevel: "普通" | "高价值" | "VIP" | "企业客户";
  region: "华东" | "华南" | "华北" | "海外";
  warehouse: string;
  inventoryState: InventoryState;
  fulfillmentPath: "单仓发货" | "跨仓调拨" | "拆单发货" | "替代 SKU" | "人工复核";
  riskLevel: RiskLevel;
  shortage: boolean;
  needsReview: boolean;
  confidence: number;
  slaDeadline: string;
  suggestion: string;
  riskReasons: string[];
  items: Array<{
    skuId: string;
    name: string;
    quantity: number;
    availableStock: number;
    lockedStock: number;
    shortage: number;
    recommendedWarehouse: string;
    substituteSku?: string;
  }>;
  warehouses: Array<{
    name: string;
    available: number;
    locked: number;
    shippable: boolean;
    outboundEta: string;
    transitEta: string;
  }>;
  timeline: Array<{
    step: string;
    status: "success" | "warn" | "pending";
    latency: string;
    strategy: RouteMode;
    note: string;
  }>;
};

export const mockOrders: MockOrder[] = [
  {
    orderId: "SO202605230001",
    platform: "TikTok Shop",
    scenario: "正常可履约",
    customerLevel: "VIP",
    region: "华东",
    warehouse: "上海仓",
    inventoryState: "充足",
    fulfillmentPath: "单仓发货",
    riskLevel: "LOW",
    shortage: false,
    needsReview: false,
    confidence: 0.93,
    slaDeadline: "今日 18:00",
    suggestion: "库存、SLA 和风险规则均通过，建议从上海仓单仓发货。",
    riskReasons: ["VIP 客户订单，优先释放可用库存"],
    items: [
      { skuId: "SKU-CASE-001", name: "磁吸手机壳", quantity: 2, availableStock: 46, lockedStock: 8, shortage: 0, recommendedWarehouse: "上海仓" },
      { skuId: "SKU-CABLE-002", name: "快充数据线", quantity: 1, availableStock: 88, lockedStock: 12, shortage: 0, recommendedWarehouse: "上海仓" },
    ],
    warehouses: [
      { name: "上海仓", available: 134, locked: 20, shippable: true, outboundEta: "2 小时", transitEta: "次日达" },
      { name: "深圳仓", available: 55, locked: 14, shippable: true, outboundEta: "4 小时", transitEta: "2 日达" },
      { name: "北京仓", available: 22, locked: 6, shippable: true, outboundEta: "6 小时", transitEta: "2 日达" },
    ],
    timeline: [
      { step: "读取订单", status: "success", latency: "42ms", strategy: "Workflow", note: "OMS 订单结构完整。" },
      { step: "检查库存", status: "success", latency: "86ms", strategy: "Workflow", note: "上海仓可满足全部 SKU。" },
      { step: "匹配规则", status: "success", latency: "31ms", strategy: "RAG", note: "命中 VIP 当日出库规则。" },
      { step: "选择路由", status: "success", latency: "64ms", strategy: "Workflow", note: "确定性链路即可处理。" },
      { step: "输出建议", status: "success", latency: "112ms", strategy: "Agent", note: "生成运营处理建议。" },
    ],
  },
  {
    orderId: "SO202605230002",
    platform: "Amazon",
    scenario: "部分缺货",
    customerLevel: "高价值",
    region: "华南",
    warehouse: "深圳仓",
    inventoryState: "部分缺货",
    fulfillmentPath: "跨仓调拨",
    riskLevel: "MEDIUM",
    shortage: true,
    needsReview: false,
    confidence: 0.78,
    slaDeadline: "明日 12:00",
    suggestion: "深圳仓库存不足，建议由上海仓调拨后拆分出库，并保留替代 SKU 方案。",
    riskReasons: ["SKU-ROUTER-009 缺口 4 件", "跨仓调拨可能压缩 SLA"],
    items: [
      { skuId: "SKU-ROUTER-009", name: "企业级路由器", quantity: 8, availableStock: 4, lockedStock: 3, shortage: 4, recommendedWarehouse: "上海仓", substituteSku: "SKU-ROUTER-009B" },
      { skuId: "SKU-HUB-021", name: "Type-C 扩展坞", quantity: 5, availableStock: 16, lockedStock: 2, shortage: 0, recommendedWarehouse: "深圳仓" },
    ],
    warehouses: [
      { name: "上海仓", available: 18, locked: 5, shippable: true, outboundEta: "3 小时", transitEta: "2 日达" },
      { name: "深圳仓", available: 27, locked: 6, shippable: true, outboundEta: "2 小时", transitEta: "次日达" },
      { name: "北京仓", available: 5, locked: 2, shippable: false, outboundEta: "8 小时", transitEta: "3 日达" },
    ],
    timeline: [
      { step: "读取订单", status: "success", latency: "49ms", strategy: "Workflow", note: "识别为高价值订单。" },
      { step: "检查库存", status: "warn", latency: "94ms", strategy: "Workflow", note: "深圳仓存在 SKU 缺口。" },
      { step: "匹配规则", status: "warn", latency: "46ms", strategy: "RAG", note: "跨仓调拨需评估 SLA。" },
      { step: "选择路由", status: "success", latency: "140ms", strategy: "Multi-Agent", note: "库存与履约专家协同给出路径。" },
      { step: "输出建议", status: "success", latency: "130ms", strategy: "Agent", note: "建议调拨并保留替代 SKU。" },
    ],
  },
  {
    orderId: "SO202605230003",
    platform: "Shopify",
    scenario: "高风险 HITL",
    customerLevel: "企业客户",
    region: "海外",
    warehouse: "洛杉矶仓",
    inventoryState: "缺货",
    fulfillmentPath: "人工复核",
    riskLevel: "HIGH",
    shortage: true,
    needsReview: true,
    confidence: 0.66,
    slaDeadline: "今日 16:00",
    suggestion: "海外仓缺货且 SLA 紧张，建议暂停自动履约，进入人工审查确认替代 SKU 和跨境发货方案。",
    riskReasons: ["海外仓缺货", "SLA 可能超时", "替代 SKU 需要确认"],
    items: [
      { skuId: "SKU-PRO-777", name: "旗舰套装", quantity: 10, availableStock: 3, lockedStock: 5, shortage: 7, recommendedWarehouse: "上海仓", substituteSku: "SKU-PRO-778" },
      { skuId: "SKU-BATT-310", name: "备用电池", quantity: 10, availableStock: 0, lockedStock: 2, shortage: 10, recommendedWarehouse: "深圳仓", substituteSku: "SKU-BATT-311" },
    ],
    warehouses: [
      { name: "上海仓", available: 16, locked: 8, shippable: true, outboundEta: "4 小时", transitEta: "5 日达" },
      { name: "深圳仓", available: 12, locked: 9, shippable: true, outboundEta: "5 小时", transitEta: "5 日达" },
      { name: "洛杉矶仓", available: 3, locked: 5, shippable: false, outboundEta: "库存不足", transitEta: "本地" },
    ],
    timeline: [
      { step: "读取订单", status: "success", latency: "54ms", strategy: "Workflow", note: "识别高价值企业客户订单。" },
      { step: "检查库存", status: "warn", latency: "112ms", strategy: "Workflow", note: "海外仓缺货，国内仓可部分补足。" },
      { step: "匹配规则", status: "warn", latency: "73ms", strategy: "RAG", note: "替代 SKU 需要人工确认。" },
      { step: "选择路由", status: "warn", latency: "126ms", strategy: "Hybrid", note: "触发 HITL 风险闸门。" },
      { step: "输出建议", status: "pending", latency: "等待", strategy: "Agent", note: "人工确认后恢复流程。" },
    ],
  },
  {
    orderId: "SO-ENT-001",
    platform: "ERP",
    scenario: "企业导入订单",
    customerLevel: "普通",
    region: "华东",
    warehouse: "上海仓",
    inventoryState: "锁库中",
    fulfillmentPath: "单仓发货",
    riskLevel: "LOW",
    shortage: false,
    needsReview: false,
    confidence: 0.89,
    slaDeadline: "明日 10:00",
    suggestion: "企业导入订单数据完整，建议释放上海仓锁定库存后发货。",
    riskReasons: ["ERP 订单需要同步锁库状态"],
    items: [
      { skuId: "SKU-ENT-001", name: "企业导入商品", quantity: 3, availableStock: 36, lockedStock: 6, shortage: 0, recommendedWarehouse: "上海仓" },
    ],
    warehouses: [
      { name: "上海仓", available: 36, locked: 6, shippable: true, outboundEta: "2 小时", transitEta: "次日达" },
      { name: "深圳仓", available: 8, locked: 1, shippable: true, outboundEta: "5 小时", transitEta: "2 日达" },
      { name: "北京仓", available: 4, locked: 0, shippable: true, outboundEta: "8 小时", transitEta: "2 日达" },
    ],
    timeline: [
      { step: "读取订单", status: "success", latency: "38ms", strategy: "Workflow", note: "读取企业导入订单。" },
      { step: "检查库存", status: "success", latency: "77ms", strategy: "Workflow", note: "上海仓库存充足。" },
      { step: "匹配规则", status: "success", latency: "26ms", strategy: "RAG", note: "满足标准 SLA。" },
      { step: "选择路由", status: "success", latency: "62ms", strategy: "Workflow", note: "推荐单仓发货。" },
      { step: "输出建议", status: "success", latency: "91ms", strategy: "Agent", note: "生成锁库释放建议。" },
    ],
  },
];

export const analysisTemplates = [
  "判断订单是否可正常履约，并给出风险与处理建议。",
  "检查库存是否充足，说明是否需要拆单或跨仓调拨。",
  "判断是否需要人工复核，并列出触发原因。",
  "推荐最优发货仓库，兼顾 SLA、库存和运输时效。",
];

export const decisionModes: RouteMode[] = ["Hybrid", "Workflow", "Agent", "RAG", "Multi-Agent"];

export const demoFlowSteps = [
  { title: "接入企业数据", detail: "OMS 订单 + WMS 库存" },
  { title: "选择待处理订单", detail: "正常、缺货、高风险" },
  { title: "Hybrid Routing", detail: "自动选择执行路径" },
  { title: "HITL 审查", detail: "高风险暂停并等待人工确认" },
];

export const mockReviewRows = [
  { orderId: "SO202605230003", riskLevel: "HIGH" as RiskLevel, reason: "海外仓缺货 / SLA 可能超时", platform: "Shopify", action: "人工确认替代 SKU", sla: "5 小时 12 分" },
  { orderId: "SO202605230008", riskLevel: "CRITICAL" as RiskLevel, reason: "高价值订单 / 地址异常", platform: "Amazon", action: "暂停履约并联系客户", sla: "2 小时 40 分" },
  { orderId: "SO202605230011", riskLevel: "MEDIUM" as RiskLevel, reason: "跨仓拆单风险", platform: "TikTok Shop", action: "确认拆单路径", sla: "8 小时 05 分" },
];

export const importHistory = [
  { batch: "IMP-20260523-001", file: "orders_20260523.csv", type: "订单", success: 1260, failed: 3, time: "2026-05-23 10:32", status: "完成" },
  { batch: "IMP-20260523-002", file: "stock_snapshot.json", type: "库存", success: 1860, failed: 0, time: "2026-05-23 08:30", status: "完成" },
  { batch: "IMP-20260522-009", file: "erp_orders.csv", type: "订单", success: 236, failed: 12, time: "2026-05-22 19:10", status: "部分失败" },
];

export const recentRuns = [
  { orderId: "SO202605230003", route: "Hybrid", latency: "1.2s", result: "进入人工审查", review: "是" },
  { orderId: "SO202605230002", route: "Multi-Agent", latency: "1.8s", result: "建议跨仓调拨", review: "否" },
  { orderId: "SO202605230001", route: "Workflow", latency: "420ms", result: "可履约", review: "否" },
  { orderId: "SO202605230008", route: "RAG", latency: "2.4s", result: "风险拦截", review: "是" },
];

export function riskToneFromLevel(level: RiskLevel | string): StatusTone {
  if (level === "CRITICAL" || level === "HIGH") return "danger";
  if (level === "MEDIUM") return "warn";
  if (level === "LOW") return "ok";
  return "neutral";
}
