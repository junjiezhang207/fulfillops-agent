import type {
  BusinessTrace,
  ExecutionProposal,
  FulfillmentCase,
  HybridRunResult,
  OrderRecord,
  PlannerRAGContext,
  PreflightValidation,
  RoutedExternalTask,
  SessionMemorySnapshot,
} from "./api";
import type { StatusTone } from "./status";

export type OpsMessageEvidence = {
  orderId?: string;
  threadId?: string;
  route?: string;
  status?: string;
  confidence?: number;
  durationMs?: number;
  fromCache?: boolean;
  tools: string[];
  sources: string[];
  notes: string[];
  trace?: BusinessTrace | null;
  actionCard?: ExecutionProposal | null;
  preflight?: PreflightValidation | null;
  sessionMemory?: SessionMemorySnapshot | null;
  ragContext?: PlannerRAGContext | null;
  fulfillmentCase?: FulfillmentCase | null;
  clientInputAt?: string;
  requestStartedAt?: string;
  requestEndedAt?: string;
  apiFailed?: boolean;
};

export const analysisTemplates = [
  "请为订单 SO202605230003 生成履约执行提案：是否换仓、拆单/合单、物流渠道变更、库存调拨或缺货处置，并展示库存、成本、时效和规则依据。",
  "请判断订单 SO202605230003 是否需要拆单或合单履约，给出可审批 Action Card。",
  "请为订单 SO202605230003 评估跨仓调货和物流渠道变更方案，批准前说明二次校验项。",
  "请复盘订单 SO202605230003 的异常原因，并把可沉淀的自动规则与仍需人工审批的动作分开。",
];

export const opsCopilotInstruction = [
  "请以“履约执行 Agent”的方式回答。",
  "必须输出可审批执行提案，但不要替 OMS/WMS/TMS/ERP 直接落库执行。",
  "输出重点放在：换仓履约、拆单/合单、物流渠道变更、库存调拨/跨仓调货、缺货处置，以及库存、成本、时效、规则依据和二次校验项。",
].join("\n");

export function extractOrderId(text: string) {
  const matched = text.match(/\b(?:SO|so)[-\w]*\d[\w-]*/);
  return matched?.[0] || "";
}

export function buildProfessionalQuestion(question: string) {
  return `${question.trim()}\n\n${opsCopilotInstruction}`;
}

export function cleanAiText(text: string) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/^\s*[-*]\s+/gm, "• ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function finalAnswer(result: HybridRunResult | null) {
  if (typeof result?.final_answer === "string" && result.final_answer.trim()) return cleanAiText(result.final_answer);
  if (result?.status === "interrupted") {
    return "系统已生成可审批执行提案，并在 HITL 节点暂停。请在 Action Card 中批准、拒绝、修改方案或继续追问。";
  }
  if (result?.status === "completed") return "分析已完成。";
  return "分析完成，但未返回可展示的结论。";
}

export function formatDuration(ms?: number) {
  if (!ms) return "--";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

export function formatConfidence(value?: number) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "--";
}

export function extractActionCard(result: HybridRunResult): ExecutionProposal | null {
  if (result.action_card) return result.action_card;
  const proposal = result.interrupt?.context?.proposal;
  if (proposal && typeof proposal === "object") return proposal as ExecutionProposal;
  return null;
}

export function buildMessageEvidence(
  result: HybridRunResult,
  ux?: Pick<OpsMessageEvidence, "clientInputAt" | "requestStartedAt" | "requestEndedAt" | "apiFailed">,
): OpsMessageEvidence {
  const tools = result.tools_called?.length ? result.tools_called : [];
  const sources = new Set<string>();
  const toolText = tools.join(" ").toLowerCase();
  if (/order|oms|订单|get_order_detail/.test(toolText) || (result.order_id && result.order_id !== "ADHOC-KNOWLEDGE")) sources.add("OMS 订单");
  if (/inventory|stock|wms|库存|get_inventory_warehouse_detail/.test(toolText)) sources.add("WMS 库存");
  if (/shipping|carrier|tms|物流|get_shipping_detail/.test(toolText)) sources.add("TMS 物流");
  if (/supply|erp|补货|采购|get_supply_chain_detail/.test(toolText)) sources.add("ERP 供应链");
  if (/product|pim|商品|get_product_constraints/.test(toolText)) sources.add("PIM 商品限制");
  if (/customer|crm|客服|客诉|get_customer_case_context/.test(toolText)) sources.add("CRM 客户上下文");
  if (/rag|knowledge|rule|规则/.test(toolText) || result.path_used?.toLowerCase().includes("rag")) sources.add("RAG 规则");
  if (result.rag_context?.sop_evidence?.evidence?.length) sources.add("SOP 知识库");
  if (result.rag_context?.similar_cases?.evidence?.length) sources.add("优秀案例库");
  if (result.session_memory?.structured?.current_topic || result.session_memory?.recent_messages?.length) sources.add("短期会话记忆");
  if (sources.size === 0) {
    sources.add("Hybrid Router");
  }

  const notes = [
    result.status === "interrupted" ? "触发人工审核，建议运营接管案件。" : "本次未返回需审批中断。",
    result.from_cache ? "本次结果来自缓存。" : "本次结果由当前请求实时生成。",
    result.session_memory?.structured?.current_topic ? `记忆主题：${result.session_memory.structured.current_topic}` : "",
    result.rag_context ? `知识召回：SOP ${result.rag_context.sop_evidence?.evidence?.length || 0} 条 / 优秀案例 ${result.rag_context.similar_cases?.evidence?.length || 0} 条。` : "",
  ].filter(Boolean);

  return {
    orderId: result.order_id === "ADHOC-KNOWLEDGE" ? undefined : result.order_id,
    threadId: result.interrupt?.thread_id || result.thread_id,
    route: result.path_used || "Hybrid Routing",
    status: result.status,
    confidence: result.confidence,
    durationMs: result.execution_time_ms,
    fromCache: result.from_cache,
    tools,
    sources: [...sources],
    notes,
    trace: result.business_trace,
    actionCard: extractActionCard(result),
    preflight: result.preflight_validation || undefined,
    sessionMemory: result.session_memory || undefined,
    ragContext: result.rag_context || undefined,
    ...ux,
  };
}

export function buildRecoveredCaseEvidence(fulfillmentCase: FulfillmentCase): OpsMessageEvidence {
  return {
    orderId: fulfillmentCase.order_id,
    route: "Fulfillment Case",
    status: fulfillmentCase.case_status,
    tools: ["fulfillment_case_store"],
    sources: ["Fulfillment Case Store", "WMS/TMS/ERP/CRM 任务回调"],
    notes: [
      "从外部任务等待区恢复案件，不重新运行 Agent。",
      "外部系统处理完成后，请执行真实业务状态验证，再决定关闭或重新规划。",
    ],
    fulfillmentCase,
  };
}

export function isAbnormalOrder(order: OrderRecord) {
  const status = (order.order_status || "").toLowerCase();
  if (order.active_fulfillment_tasks?.length) return true;
  if (order.already_split || order.waybill_created || order.package_created) return true;
  return /(exception|stockout|hold|risk|异常|缺货|待处理|待履约|待发货)/.test(status);
}

export function isPendingFulfillmentCase(caseItem: FulfillmentCase) {
  return ["WAITING_EXTERNAL_TASK", "VERIFYING", "FAILED", "REJECTED", "REPLAN_REQUIRED"].includes(caseItem.case_status);
}

export function caseStatusTone(status: string): StatusTone {
  if (status === "COMPLETED") return "ok";
  if (status.includes("FAILED") || status.includes("REJECTED") || status.includes("REPLAN")) return "danger";
  if (status.includes("WAITING") || status.includes("VERIFY")) return "warn";
  return "neutral";
}

export function buildOrderActionPrompt(orderId: string) {
  return `请为订单 ${orderId} 生成履约执行提案：先识别当前异常原因，再基于实时订单、库存、仓库、物流、在途补货、商品限制和客诉状态，给出可审批 Action Card。`;
}

export function buildExternalTaskBusinessResult(
  task: RoutedExternalTask,
  status: "COMPLETED" | "FAILED",
): Record<string, unknown> {
  const base = {
    simulated_from: "operations_dashboard",
    simulated_at: new Date().toISOString(),
    business_state_verified: status === "COMPLETED",
  };
  if (status !== "COMPLETED") {
    return { ...base, failure_reason: "外部系统返回失败，需重新规划。" };
  }
  if (task.target_system === "WMS" && task.action_type === "inventory_transfer") {
    return {
      ...base,
      transfer_order_id: `TR-${task.task_id.slice(-8)}`,
      transfer_status: "CONFIRMED",
    };
  }
  if (task.target_system === "WMS") {
    return {
      ...base,
      fulfillment_task_id: `WMS-${task.task_id.slice(-8)}`,
      fulfillment_task_created: true,
    };
  }
  if (task.target_system === "TMS") {
    return {
      ...base,
      quote_confirmed: true,
      serviceable: true,
      channel_status: "confirmed",
    };
  }
  if (task.target_system === "ERP") {
    return {
      ...base,
      replenishment_request_id: `ERP-${task.task_id.slice(-8)}`,
      replenishment_status: "CONFIRMED",
      inbound_stock_visible: true,
    };
  }
  if (task.target_system === "CRM") {
    return {
      ...base,
      crm_case_id: `CRM-${task.task_id.slice(-8)}`,
      customer_confirmation_status: "contacted",
    };
  }
  return base;
}
