import { strict as assert } from "node:assert";

import {
  buildMessageEvidence,
  buildExternalTaskBusinessResult,
  buildOrderActionPrompt,
  buildProfessionalQuestion,
  buildRecoveredCaseEvidence,
  caseStatusTone,
  extractActionCard,
  extractOrderId,
  finalAnswer,
  isAbnormalOrder,
  isPendingFulfillmentCase,
} from "../src/lib/opsConsole.js";
import type {
  ExecutionProposal,
  FulfillmentCase,
  HybridRunResult,
  OrderRecord,
  RoutedExternalTask,
} from "../src/lib/api";

function proposal(overrides: Partial<ExecutionProposal> = {}): ExecutionProposal {
  return {
    proposal_id: "prop-1",
    order_id: "SO-FE-001",
    title: "跨仓拆单履约提案",
    summary: "上海仓发 SKU1，杭州仓发 SKU2。",
    status: "pending_approval",
    capabilities: ["拆单", "库存调拨/跨仓调货"],
    actions: [
      {
        action_id: "act-1",
        action_type: "split_order",
        sku_id: "SKU-X",
        quantity: 2,
        from_warehouse: "WH-HZ",
        to_warehouse: null,
        carrier: "顺丰标快",
        cost_delta: 18,
        eta_hours: 24,
        reason: "杭州仓可覆盖剩余数量。",
        reversible: true,
      },
    ],
    decision_context: {
      data_loading_policy: {
        vector_store_backend: "chroma",
        immediate_inventory_judgement_field: "available_stock",
      },
    },
    inventory_snapshot: [],
    cost_breakdown: { shipping_cost: 18 },
    eta: { eta_hours: 24, eta_label: "24h" },
    rule_citations: ["缺货订单先检查同区域仓，再评估跨仓调拨。"],
    data_fingerprint: "fp-1",
    freshness: {},
    approval_required: true,
    preflight_checks: ["订单状态仍可履约", "SKU 可售库存仍覆盖动作数量", "物流渠道报价与时效仍有效"],
    ...overrides,
  };
}

function fulfillmentCase(overrides: Partial<FulfillmentCase> = {}): FulfillmentCase {
  const plan = proposal();
  return {
    case_id: "case-SO-FE-001",
    order_id: "SO-FE-001",
    proposal_id: plan.proposal_id,
    case_status: "WAITING_EXTERNAL_TASK",
    plan,
    checkpoint: { thread_id: "thread-fe-001", node: "human_approval" },
    tasks: [
      {
        task_id: "task-1",
        case_id: "case-SO-FE-001",
        proposal_id: plan.proposal_id,
        action_id: "act-1",
        action_type: "split_order",
        target_system: "WMS",
        domain_service: "warehouse_task_service",
        external_task_type: "WAREHOUSE_OPERATION",
        payload: { order_id: "SO-FE-001", sku_id: "SKU-X", quantity: 2 },
        status: "RUNNING",
        created_at: "2026-08-28T10:00:00Z",
        updated_at: "2026-08-28T10:00:00Z",
        result: { external_ref: "WMS-1" },
      },
    ],
    created_at: "2026-08-28T10:00:00Z",
    updated_at: "2026-08-28T10:00:00Z",
    verification: {},
    ...overrides,
  };
}

const abnormalOrder: OrderRecord = {
  order_id: "SO-FE-001",
  platform: "京东",
  order_time: "2026-08-28T09:00:00Z",
  order_status: "stockout_exception",
  region: "上海",
  priority: "vip",
  current_warehouse_id: "WH-SH",
  shipping_region: "上海/浦东",
  items: [
    {
      sku_id: "SKU-X",
      product_name: "核心商品",
      quantity: 5,
      unit_price: 299,
      allocated_quantity: 3,
      shipped_quantity: 0,
      sku_status: "stockout",
      split_allowed: true,
    },
  ],
};

assert.equal(extractOrderId("请处理订单 SO-FE-001 缺货"), "SO-FE-001");
assert.equal(isAbnormalOrder(abnormalOrder), true);
assert.match(buildOrderActionPrompt("SO-FE-001"), /SO-FE-001/);
assert.match(buildOrderActionPrompt("SO-FE-001"), /Action Card/);
assert.match(buildProfessionalQuestion("这个订单当前仓缺货，应该怎么处理？"), /履约执行 Agent/);

const result: HybridRunResult = {
  order_id: "SO-FE-001",
  final_answer: "",
  path_used: "workflow",
  tools_called: ["order_analysis", "inventory_analysis", "knowledge_retrieval"],
  status: "interrupted",
  interrupt: {
    type: "fulfillment_action_approval",
    node: "human_approval",
    prompt: "请审批",
    context: { proposal: proposal() },
    options: ["approved", "rejected", "modify", "ask_followup"],
    thread_id: "thread-fe-001",
    risk_level: "HIGH",
    risk_signals: ["split_order_required"],
    timeout_seconds: 1800,
  },
  action_card: null,
  preflight_validation: null,
};

const actionCard = extractActionCard(result);
assert.equal(actionCard?.proposal_id, "prop-1");
assert.match(finalAnswer(result), /HITL/);

const evidence = buildMessageEvidence(result);
assert.equal(evidence.orderId, "SO-FE-001");
assert.equal(evidence.threadId, "thread-fe-001");
assert.equal(evidence.actionCard?.title, "跨仓拆单履约提案");
assert.deepEqual(evidence.sources, ["OMS 订单", "WMS 库存", "RAG 规则"]);

const waitingCase = fulfillmentCase();
assert.equal(isPendingFulfillmentCase(waitingCase), true);
assert.equal(caseStatusTone(waitingCase.case_status), "warn");
assert.equal(caseStatusTone("REPLAN_REQUIRED"), "danger");
assert.equal(caseStatusTone("COMPLETED"), "ok");

const recovered = buildRecoveredCaseEvidence(waitingCase);
assert.equal(recovered.route, "Fulfillment Case");
assert.equal(recovered.fulfillmentCase?.case_id, waitingCase.case_id);
assert.equal(recovered.tools.includes("fulfillment_case_store"), true);
assert.equal(recovered.notes.some((note) => note.includes("不重新运行 Agent")), true);

const wmsFulfillmentResult = buildExternalTaskBusinessResult(waitingCase.tasks[0], "COMPLETED");
assert.equal(wmsFulfillmentResult.business_state_verified, true);
assert.equal(wmsFulfillmentResult.fulfillment_task_created, true);
assert.ok(wmsFulfillmentResult.fulfillment_task_id);

const wmsTransferTask: RoutedExternalTask = {
  ...waitingCase.tasks[0],
  task_id: "task-wms-transfer-1",
  action_type: "inventory_transfer",
  external_task_type: "INVENTORY_TRANSFER",
};
const wmsTransferResult = buildExternalTaskBusinessResult(wmsTransferTask, "COMPLETED");
assert.equal(wmsTransferResult.business_state_verified, true);
assert.equal(wmsTransferResult.transfer_status, "CONFIRMED");
assert.ok(wmsTransferResult.transfer_order_id);

const tmsTask: RoutedExternalTask = {
  ...waitingCase.tasks[0],
  task_id: "task-tms-1",
  target_system: "TMS",
  action_type: "change_carrier",
  external_task_type: "CHANGE_CARRIER",
};
const tmsResult = buildExternalTaskBusinessResult(tmsTask, "COMPLETED");
assert.equal(tmsResult.quote_confirmed, true);
assert.equal(tmsResult.serviceable, true);
assert.equal(tmsResult.channel_status, "confirmed");

const erpTask: RoutedExternalTask = {
  ...waitingCase.tasks[0],
  task_id: "task-erp-1",
  target_system: "ERP",
  action_type: "replenishment",
  external_task_type: "REPLENISHMENT_REQUEST",
};
const erpResult = buildExternalTaskBusinessResult(erpTask, "COMPLETED");
assert.equal(erpResult.replenishment_status, "CONFIRMED");
assert.ok(erpResult.replenishment_request_id);

const crmTask: RoutedExternalTask = {
  ...waitingCase.tasks[0],
  task_id: "task-crm-1",
  target_system: "CRM",
  action_type: "stockout_resolution",
  external_task_type: "CUSTOMER_STOCKOUT_CONFIRMATION",
};
const crmResult = buildExternalTaskBusinessResult(crmTask, "COMPLETED");
assert.ok(crmResult.crm_case_id);
assert.equal(crmResult.customer_confirmation_status, "contacted");

const failedResult = buildExternalTaskBusinessResult(tmsTask, "FAILED");
assert.equal(failedResult.business_state_verified, false);
assert.match(String(failedResult.failure_reason), /重新规划/);

console.log("ops console contract passed");
