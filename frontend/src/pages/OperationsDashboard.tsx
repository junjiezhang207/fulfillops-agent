import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  ClipboardList,
  Database,
  FileCheck2,
  PencilLine,
  RefreshCcw,
  Search,
  Sparkles,
  Split,
  Truck,
  Warehouse,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  confirmFulfillmentCase,
  listEnterpriseOrders,
  listFulfillmentCases,
  persistExcellentCase,
  resumeHybrid,
  runHybrid,
  updateFulfillmentTask,
  verifyFulfillmentCase,
  type ExecutionProposal,
  type FulfillmentCase,
  type HybridRunResult,
  type OrderRecord,
  type PreflightValidation,
  type RoutedExternalTask,
} from "../lib/api";
import {
  buildExternalTaskBusinessResult,
  buildMessageEvidence,
  buildOrderActionPrompt,
  buildProfessionalQuestion,
  buildRecoveredCaseEvidence,
  caseStatusTone,
  extractActionCard,
  finalAnswer,
  isAbnormalOrder,
  isPendingFulfillmentCase,
  type OpsMessageEvidence,
} from "../lib/opsConsole";
import type { StatusTone } from "../lib/status";

type AnalysisEvent = {
  id: string;
  role: "operator" | "agent" | "system";
  content: string;
  createdAt: string;
  status?: "typing" | "error" | "done";
  evidence?: OpsMessageEvidence;
};

type ReviewDecision = "approved" | "rejected" | "modify" | "ask_followup";

type ReviewState = {
  orderId: string;
  threadId?: string;
  riskLevel: string;
  reasons: string[];
  suggestion: string;
  proposal?: ExecutionProposal | null;
  preflight?: PreflightValidation | null;
};

const STORAGE_KEY = "fulfillops-agent-analysis-events";

const initialEvents: AnalysisEvent[] = [
  {
    id: "welcome",
    role: "system",
    content: "异常处置工作台已就绪。选择左侧异常订单后，可生成 AI 处置方案并进入人工确认。",
    createdAt: new Date().toISOString(),
    status: "done",
  },
];

function makeId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function makeWebhookEventId(taskId: string) {
  return `evt-${taskId}-${Date.now()}`;
}

function loadEvents() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? (JSON.parse(raw) as AnalysisEvent[]) : null;
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : initialEvents;
  } catch {
    return initialEvents;
  }
}

function compactValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "--";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === "string") return value;
  return JSON.stringify(value) || "--";
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asRecordList(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? (value.filter((item) => item && typeof item === "object") as Array<Record<string, unknown>>) : [];
}

function formatDateTime(value?: string | null) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatSla(value?: string | null) {
  if (!value) return "SLA 待确认";
  const dueAt = new Date(value);
  if (Number.isNaN(dueAt.getTime())) return "SLA 待确认";
  const deltaHours = (dueAt.getTime() - Date.now()) / 3_600_000;
  if (deltaHours < 0) return "已超时";
  if (deltaHours < 1) return "<1h";
  return `${Math.round(deltaHours)}h`;
}

function actionTypeLabel(type: string) {
  const labels: Record<string, string> = {
    switch_warehouse: "换仓",
    split_order: "拆单",
    merge_order: "合单",
    change_carrier: "物流变更",
    inventory_transfer: "库存调拨",
    stockout_resolution: "缺货处置",
    ship_from_warehouse: "仓库发货",
  };
  return labels[type] || type;
}

function actionIcon(type: string) {
  if (type.includes("carrier") || type.includes("ship")) return <Truck size={15} />;
  if (type.includes("warehouse") || type.includes("transfer")) return <Warehouse size={15} />;
  return <Split size={15} />;
}

function orderExceptionLabel(order: OrderRecord) {
  const status = (order.order_status || "").toLowerCase();
  if (/stockout|缺货/.test(status)) return "库存不足";
  if (/hold|risk|风险/.test(status)) return "人工接管";
  if (order.already_split) return "拆单异常";
  if (!order.current_warehouse_id) return "待分仓";
  if (order.active_fulfillment_tasks?.length) return "执行等待";
  return order.order_status || "履约异常";
}

function priorityTone(priority?: string): StatusTone {
  const value = (priority || "").toLowerCase();
  if (/vip|high|urgent|高/.test(value)) return "danger";
  if (/medium|中/.test(value)) return "warn";
  return "neutral";
}

function proposalStatusTone(status?: string): StatusTone {
  const value = (status || "").toLowerCase();
  if (/failed|rejected|invalid/.test(value)) return "danger";
  if (/waiting|pending|approval|confirm/.test(value)) return "warn";
  if (/running|execut/.test(value)) return "info";
  if (/complete|valid|approved/.test(value)) return "ok";
  return "neutral";
}

function confidenceLabel(value?: number) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "--";
}

function DotBadge({ tone = "neutral", children }: { tone?: StatusTone | "teal"; children: React.ReactNode }) {
  return (
    <span className={`ops-dot-badge ops-dot-badge--${tone}`}>
      <i />
      {children}
    </span>
  );
}

function SectionHeader({ title, meta }: { title: string; meta?: string }) {
  return (
    <div className="ops-section-head">
      <strong>{title}</strong>
      {meta && <span>{meta}</span>}
    </div>
  );
}

function CompactEmpty({ title, text, action }: { title: string; text: string; action?: React.ReactNode }) {
  return (
    <div className="ops-compact-empty">
      <strong>{title}</strong>
      <span>{text}</span>
      {action}
    </div>
  );
}

function OrderQueue({
  orders,
  cases,
  loadingOrders,
  loadingCases,
  selectedOrderId,
  onSelectOrder,
  onAnalyzeOrder,
  onSelectCase,
}: {
  orders: OrderRecord[];
  cases: FulfillmentCase[];
  loadingOrders: boolean;
  loadingCases: boolean;
  selectedOrderId?: string;
  onSelectOrder: (orderId: string) => void;
  onAnalyzeOrder: (orderId: string) => void;
  onSelectCase: (fulfillmentCase: FulfillmentCase) => void;
}) {
  const abnormalOrders = orders.filter((order) => isAbnormalOrder(order)).slice(0, 12);
  const waitingCases = cases.filter((caseItem) => isPendingFulfillmentCase(caseItem)).slice(0, 5);

  return (
    <aside className="ops-order-queue" aria-label="异常订单队列">
      <div className="ops-order-queue__top">
        <div>
          <h2>异常订单</h2>
          <p>{loadingOrders ? "同步中" : `${abnormalOrders.length} 个待处理`}</p>
        </div>
        <DotBadge tone={abnormalOrders.length ? "warn" : "neutral"}>{abnormalOrders.length}</DotBadge>
      </div>

      <div className="ops-queue-search" aria-hidden="true">
        <Search size={15} />
        <span>按订单号、仓库、区域筛选</span>
      </div>

      <div className="ops-order-list">
        {abnormalOrders.length ? (
          abnormalOrders.map((order) => (
            <button
              type="button"
              key={order.order_id}
              className={order.order_id === selectedOrderId ? "selected" : ""}
              onClick={() => onSelectOrder(order.order_id)}
            >
              <span className="ops-order-list__main">
                <strong>{order.order_id}</strong>
                <small>{order.current_warehouse_id || "待分仓"} → {order.shipping_region || order.region || "--"}</small>
              </span>
              <DotBadge tone={priorityTone(order.priority)}>{orderExceptionLabel(order)}</DotBadge>
              <span className="ops-order-list__meta">
                <small>{order.items.length} SKU</small>
                <small>SLA {formatSla(order.promise_delivery_time)}</small>
              </span>
              <span
                className="ops-order-list__action"
                onClick={(event) => {
                  event.stopPropagation();
                  onAnalyzeOrder(order.order_id);
                }}
              >
                分析
              </span>
            </button>
          ))
        ) : (
          <CompactEmpty
            title="暂无异常订单"
            text={loadingOrders ? "正在读取 OMS / WMS 数据。" : "导入 Mock OMS/WMS 数据后将在此显示。"}
          />
        )}
      </div>

      <div className="ops-external-mini">
        <SectionHeader title="外部任务" meta={loadingCases ? "同步中" : `${waitingCases.length} 个等待`} />
        {waitingCases.length ? (
          waitingCases.map((caseItem) => (
            <button type="button" key={caseItem.case_id} onClick={() => onSelectCase(caseItem)}>
              <span>
                <strong>{caseItem.order_id}</strong>
                <small>{caseItem.tasks.length} tasks · {caseItem.case_id}</small>
              </span>
              <DotBadge tone={caseStatusTone(caseItem.case_status)}>{caseItem.case_status}</DotBadge>
            </button>
          ))
        ) : (
          <p>暂无等待中的外部任务。</p>
        )}
      </div>
    </aside>
  );
}

function OrderSummary({
  order,
  proposal,
  caseItem,
  analyzing,
  onAnalyzeOrder,
}: {
  order?: OrderRecord;
  proposal?: ExecutionProposal | null;
  caseItem?: FulfillmentCase | null;
  analyzing: boolean;
  onAnalyzeOrder: (orderId: string) => void;
}) {
  if (!order) {
    return (
      <section className="ops-order-summary ops-order-summary--empty">
        <CompactEmpty title="请选择左侧异常订单" text="选中订单后会展示履约摘要、业务事实和 AI 处置建议。" />
      </section>
    );
  }

  return (
    <section className="ops-order-summary">
      <div className="ops-order-summary__identity">
        <span>ORDER</span>
        <h1>{order.order_id}</h1>
        <div>
          <DotBadge tone="warn">{orderExceptionLabel(order)}</DotBadge>
          <DotBadge tone={priorityTone(order.priority)}>{order.priority || "normal"}</DotBadge>
          {caseItem && <DotBadge tone={caseStatusTone(caseItem.case_status)}>{caseItem.case_status}</DotBadge>}
        </div>
      </div>
      <div className="ops-order-summary__facts">
        <article><span>当前仓</span><strong>{order.current_warehouse_id || "待分仓"}</strong></article>
        <article><span>目的地</span><strong>{order.shipping_region || order.region || "--"}</strong></article>
        <article><span>承诺送达</span><strong>{formatDateTime(order.promise_delivery_time)}</strong></article>
        <article><span>履约状态</span><strong>{proposal?.status || order.order_status || "待决策"}</strong></article>
      </div>
      <button className="ops-primary-action" type="button" disabled={analyzing} onClick={() => onAnalyzeOrder(order.order_id)}>
        {analyzing ? <Sparkles size={16} /> : <RefreshCcw size={16} />}
        {analyzing ? "分析中" : proposal ? "重新分析" : "开始分析"}
      </button>
    </section>
  );
}

function OrderContext({
  order,
  proposal,
  activeCase,
}: {
  order?: OrderRecord;
  proposal?: ExecutionProposal | null;
  activeCase?: FulfillmentCase | null;
}) {
  if (!order) {
    return (
      <section className="ops-context-panel">
        <CompactEmpty title="未选择订单" text="请选择左侧异常订单。" />
      </section>
    );
  }

  const context = proposal?.decision_context || {};
  const inventoryRows = (proposal?.inventory_snapshot || []).slice(0, 4);
  const logisticsRows = asRecordList(context.logistics_options).slice(0, 3);
  const candidateWarehouses = asRecordList(context.candidate_warehouses).slice(0, 3);
  const productRestrictions = asRecordList(context.product_restrictions).slice(0, 3);
  const customerRisk = asRecord(context.customer_risk);
  const sourceStatus = [
    ["OMS", Boolean(order)],
    ["WMS", Boolean(inventoryRows.length || context.sku_warehouse_inventory)],
    ["TMS", Boolean(logisticsRows.length)],
    ["ERP", Boolean(context.replenishment_options)],
    ["PIM", Boolean(productRestrictions.length)],
    ["CRM", Boolean(Object.keys(customerRisk).length)],
  ] as const;

  return (
    <section className="ops-context-panel">
      <div className="ops-data-sources">
        <SectionHeader title="Data Sources" meta="固定必查项" />
        <div>
          {sourceStatus.map(([source, ready]) => (
            <DotBadge tone={ready ? "ok" : "neutral"} key={source}>{source}</DotBadge>
          ))}
        </div>
      </div>

      <div className="ops-context-grid">
        <section className="ops-context-section">
          <SectionHeader title="履约状态" />
          <dl className="ops-kv-grid">
            <div><dt>当前仓</dt><dd>{order.current_warehouse_id || "待分仓"}</dd></div>
            <div><dt>库存占用</dt><dd>{order.inventory_reserved ? "已占用" : "待确认"}</dd></div>
            <div><dt>包裹</dt><dd>{order.package_created ? "已创建" : "未创建"}</dd></div>
            <div><dt>SLA</dt><dd>{formatSla(order.promise_delivery_time)}</dd></div>
          </dl>
        </section>

        <section className="ops-context-section">
          <SectionHeader title="库存与仓网" />
          {inventoryRows.length ? (
            <div className="ops-compact-table">
              {inventoryRows.map((item, index) => (
                <article key={`${String(item.sku_id || index)}-${index}`}>
                  <strong>{String(item.sku_id || "--")}</strong>
                  <span>需求 {compactValue(item.required_quantity)} · 可用 {compactValue(item.total_available_stock)}</span>
                </article>
              ))}
            </div>
          ) : (
            <p className="ops-muted-line">等待 WMS 库存矩阵。</p>
          )}
          {candidateWarehouses.length > 0 && (
            <div className="ops-chip-row">
              {candidateWarehouses.map((warehouse, index) => (
                <span key={`${String(warehouse.warehouse_id || index)}`}>{compactValue(warehouse.warehouse_id || warehouse.name)}</span>
              ))}
            </div>
          )}
        </section>

        <section className="ops-context-section">
          <SectionHeader title="物流" />
          {logisticsRows.length ? (
            <div className="ops-compact-table">
              {logisticsRows.map((item, index) => (
                <article key={`${String(item.carrier || index)}-${index}`}>
                  <strong>{compactValue(item.carrier || item.channel || item.name)}</strong>
                  <span>ETA {compactValue(item.eta_hours || item.eta)} · 成本 {compactValue(item.cost || item.cost_delta)}</span>
                </article>
              ))}
            </div>
          ) : (
            <p className="ops-muted-line">等待 TMS 渠道与 ETA。</p>
          )}
        </section>

        <section className="ops-context-section">
          <SectionHeader title="客户与商品约束" />
          <dl className="ops-kv-grid">
            <div><dt>平台</dt><dd>{order.platform || "--"}</dd></div>
            <div><dt>履约类型</dt><dd>{order.fulfillment_type || "--"}</dd></div>
            <div><dt>商品约束</dt><dd>{productRestrictions.length ? `${productRestrictions.length} 条` : "未命中"}</dd></div>
            <div><dt>客户风险</dt><dd>{compactValue(customerRisk.level || customerRisk.status)}</dd></div>
          </dl>
        </section>
      </div>

      <section className="ops-context-section ops-context-section--wide">
        <SectionHeader title="处理进度" />
        <div className="ops-progress-rail">
          {[
            ["分析中", Boolean(proposal)],
            ["等待人工确认", Boolean(proposal?.approval_required)],
            ["任务执行", Boolean(activeCase)],
            ["结果验证", Boolean(activeCase?.verification && Object.keys(activeCase.verification).length)],
            ["完成", activeCase?.case_status === "COMPLETED"],
          ].map(([label, active], index) => (
            <article className={active ? "active" : ""} key={String(label)}>
              <span>{index + 1}</span>
              <strong>{label}</strong>
            </article>
          ))}
        </div>
      </section>
    </section>
  );
}

function ActionPlanEditor({
  proposal,
  onProposalChange,
  readOnly,
}: {
  proposal: ExecutionProposal;
  onProposalChange?: (proposal: ExecutionProposal) => void;
  readOnly?: boolean;
}) {
  const [draft, setDraft] = useState(proposal);
  const [editingActionId, setEditingActionId] = useState<string | null>(null);
  const [manualEdited, setManualEdited] = useState(false);

  function patchAction(actionId: string, patch: Partial<ExecutionProposal["actions"][number]>) {
    const next = {
      ...draft,
      actions: draft.actions.map((action) => (action.action_id === actionId ? { ...action, ...patch } : action)),
    };
    setDraft(next);
    setManualEdited(true);
    onProposalChange?.(next);
  }

  return (
    <section className="ops-action-plan">
      <SectionHeader title="Action Plan" meta={manualEdited ? "人工已修改" : `${draft.actions.length} steps`} />
      <div className="ops-action-steps">
        {draft.actions.map((action, index) => (
          <article key={action.action_id}>
            <span className="ops-action-steps__index">{index + 1}</span>
            <div>
              <div className="ops-action-steps__title">
                <strong>{actionTypeLabel(action.action_type)}</strong>
                <small>{action.responsibility_domain || "WMS"}</small>
              </div>
              {editingActionId === action.action_id ? (
                <div className="ops-action-edit">
                  <label>
                    数量
                    <input
                      type="number"
                      min="0"
                      value={action.quantity || 0}
                      onChange={(event) => patchAction(action.action_id, { quantity: Number(event.target.value) })}
                    />
                  </label>
                  <label>
                    原仓
                    <input value={action.from_warehouse || ""} onChange={(event) => patchAction(action.action_id, { from_warehouse: event.target.value })} />
                  </label>
                  <label>
                    目标仓
                    <input value={action.to_warehouse || ""} onChange={(event) => patchAction(action.action_id, { to_warehouse: event.target.value })} />
                  </label>
                  <label>
                    物流
                    <input value={action.carrier || ""} onChange={(event) => patchAction(action.action_id, { carrier: event.target.value })} />
                  </label>
                  <label>
                    系统
                    <select
                      value={action.responsibility_domain || "WMS"}
                      onChange={(event) => patchAction(action.action_id, { responsibility_domain: event.target.value })}
                    >
                      <option value="OMS">OMS</option>
                      <option value="WMS">WMS</option>
                      <option value="TMS">TMS</option>
                      <option value="ERP">ERP</option>
                      <option value="PIM">PIM</option>
                      <option value="CRM">CRM</option>
                    </select>
                  </label>
                  <label className="ops-action-edit__reason">
                    说明
                    <textarea rows={2} value={action.reason} onChange={(event) => patchAction(action.action_id, { reason: event.target.value })} />
                  </label>
                </div>
              ) : (
                <>
                  <p>{action.reason}</p>
                  <div className="ops-action-steps__meta">
                    <span>{actionIcon(action.action_type)} {action.sku_id || draft.order_id}</span>
                    <span>{action.quantity ? `${action.quantity} 件` : "数量待定"}</span>
                    <span>{action.from_warehouse || action.to_warehouse || "仓库待定"}</span>
                    {action.carrier && <span>{action.carrier}</span>}
                  </div>
                </>
              )}
            </div>
            {!readOnly && (
              <button type="button" onClick={() => setEditingActionId(editingActionId === action.action_id ? null : action.action_id)}>
                <PencilLine size={14} />
                {editingActionId === action.action_id ? "完成" : "编辑"}
              </button>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

function AIDecisionPanel({
  order,
  evidence,
  proposal,
  activeCase,
  loading,
  review,
  reviewNotes,
  reviewError,
  onAnalyzeOrder,
  onOpenReview,
  onProposalChange,
  onNotesChange,
  onSubmitReview,
  onCaseUpdated,
}: {
  order?: OrderRecord;
  evidence?: OpsMessageEvidence | null;
  proposal?: ExecutionProposal | null;
  activeCase?: FulfillmentCase | null;
  loading: boolean;
  review: ReviewState | null;
  reviewNotes: string;
  reviewError: string | null;
  onAnalyzeOrder: (orderId: string) => void;
  onOpenReview: (evidence: OpsMessageEvidence, decision: ReviewDecision) => void;
  onProposalChange: (proposal: ExecutionProposal) => void;
  onNotesChange: (value: string) => void;
  onSubmitReview: (decision: ReviewDecision) => void;
  onCaseUpdated: (fulfillmentCase: FulfillmentCase) => void;
}) {
  const decisionEvidence = evidence && proposal ? { ...evidence, actionCard: proposal } : null;
  const costDelta = proposal ? compactValue(proposal.cost_breakdown.estimated_total_delta ?? proposal.cost_breakdown.shipping_cost) : "--";
  const etaLabel = proposal ? compactValue(proposal.eta.eta_label ?? proposal.eta.eta_hours) : "--";
  const reasons = proposal?.inventory_snapshot?.slice(0, 3).map((item) =>
    `${compactValue(item.sku_id)}：需求 ${compactValue(item.required_quantity)}，可用 ${compactValue(item.total_available_stock)}`,
  ) || [];
  const similarCases = evidence?.ragContext?.similar_cases?.evidence || [];

  return (
    <aside className="ops-ai-panel">
      <div className="ops-ai-panel__scroll">
        <div className="ops-ai-panel__head">
          <div>
            <h2>AI 异常分析</h2>
            <p>基于实时履约数据、SOP 与历史案例生成处置建议</p>
          </div>
          <DotBadge tone={loading ? "info" : proposal ? "warn" : "neutral"}>
            {loading ? "分析中" : proposal ? "待确认" : "待分析"}
          </DotBadge>
        </div>

        {!proposal && (
          <CompactEmpty
            title="尚未生成处置方案"
            text="选择订单后启动分析，AI 会输出根因、推荐方案和系统分发动作。"
            action={order?.order_id && (
              <button className="ops-primary-action" type="button" disabled={loading} onClick={() => onAnalyzeOrder(order.order_id)}>
                {loading ? <Sparkles size={16} /> : <ClipboardList size={16} />}
                {loading ? "分析中" : "开始异常分析"}
              </button>
            )}
          />
        )}

        {proposal && (
          <>
            <section className="ops-ai-judgement">
              <div>
                <span>异常判断</span>
                <strong>{order ? orderExceptionLabel(order) : proposal.goal_type || "履约异常"}</strong>
              </div>
              <DotBadge tone="ok">Confidence {confidenceLabel(evidence?.confidence)}</DotBadge>
              <ul>
                {(reasons.length ? reasons : evidence?.notes || ["已生成可审批执行提案。"]).slice(0, 3).map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </section>

            <section className="ops-ai-recommendation">
              <div className="ops-ai-recommendation__top">
                <div>
                  <span>方案 A · 推荐</span>
                  <strong>{proposal.title}</strong>
                </div>
                <DotBadge tone={proposalStatusTone(proposal.status)}>{proposal.status}</DotBadge>
              </div>
              <p>{proposal.summary}</p>
              <div className="ops-ai-metrics">
                <article><span>送达时间</span><strong>{etaLabel}</strong></article>
                <article><span>成本变化</span><strong>¥{costDelta}</strong></article>
                <article><span>SLA</span><strong>{proposal.preflight_checks.length ? "需校验" : "待确认"}</strong></article>
              </div>
            </section>

            <details className="ops-ai-details">
              <summary>SOP Evidence</summary>
              {(proposal.rule_citations.length ? proposal.rule_citations : evidence?.ragContext?.sop_evidence?.key_points || ["暂无 SOP 证据"]).slice(0, 6).map((rule) => (
                <p key={rule}>{rule}</p>
              ))}
            </details>

            <details className="ops-ai-details">
              <summary>Similar Case</summary>
              {similarCases.length ? similarCases.slice(0, 4).map((item, index) => (
                <p key={`${item.source_case_id || item.source_file || index}`}>
                  {item.source_case_id || item.source_file || "历史案例"} · score {compactValue(item.score)}
                </p>
              )) : <p>暂无相似案例。</p>}
            </details>

            <ActionPlanEditor
              key={proposal.proposal_id}
              proposal={proposal}
              readOnly={!decisionEvidence}
              onProposalChange={onProposalChange}
            />
          </>
        )}

        {activeCase && (
          <FulfillmentCasePanel
            key={activeCase.case_id}
            fulfillmentCase={activeCase}
            onCaseUpdated={onCaseUpdated}
            onReplanProposal={(nextProposal) => onAnalyzeOrder(nextProposal.order_id)}
          />
        )}
      </div>

      {proposal && (
        <section className="ops-hitl-bar">
          {review && (
            <div className="ops-hitl-review">
              <div>
                <strong>人工确认</strong>
                <span>{review.riskLevel} · {review.reasons.slice(0, 2).join(" / ")}</span>
              </div>
              <textarea
                value={reviewNotes}
                onChange={(event) => onNotesChange(event.target.value)}
                placeholder="填写修改点、审批备注或拒绝原因"
                rows={3}
              />
              {reviewError && <div className="error-box">{reviewError}</div>}
            </div>
          )}
          <div className="ops-hitl-actions">
            <button
              className="ops-hitl-actions__primary"
              type="button"
              onClick={() => review ? onSubmitReview("approved") : decisionEvidence && onOpenReview(decisionEvidence, "approved")}
            >
              <CheckCircle2 size={16} />
              确认执行
            </button>
            <button type="button" onClick={() => review ? onSubmitReview("modify") : decisionEvidence && onOpenReview(decisionEvidence, "modify")}>
              <PencilLine size={16} />
              修改方案
            </button>
            <button className="danger" type="button" onClick={() => review ? onSubmitReview("rejected") : decisionEvidence && onOpenReview(decisionEvidence, "rejected")}>
              <XCircle size={16} />
              拒绝
            </button>
          </div>
        </section>
      )}
    </aside>
  );
}

function FulfillmentCasePanel({
  fulfillmentCase,
  onCaseUpdated,
  onReplanProposal,
}: {
  fulfillmentCase: FulfillmentCase;
  onCaseUpdated: (fulfillmentCase: FulfillmentCase) => void;
  onReplanProposal: (proposal: ExecutionProposal) => void;
}) {
  const verifyMutation = useMutation({ mutationFn: verifyFulfillmentCase });
  const excellentMutation = useMutation({ mutationFn: persistExcellentCase });
  const webhookMutation = useMutation({ mutationFn: updateFulfillmentTask });
  const [message, setMessage] = useState<string | null>(null);
  const [caseState, setCaseState] = useState(fulfillmentCase);
  const [replanProposal, setReplanProposal] = useState<ExecutionProposal | null>(null);

  async function verifyCase() {
    setMessage(null);
    const response = await verifyMutation.mutateAsync(caseState.case_id);
    const verification = response.data.verification;
    const nextCase = {
      ...caseState,
      case_status: verification.status,
      verification: { checks: verification.checks, message: verification.message },
      plan: verification.replacement_proposal || caseState.plan,
    };
    setMessage(verification.message);
    setReplanProposal(verification.replacement_proposal || null);
    setCaseState(nextCase);
    onCaseUpdated(nextCase);
  }

  async function saveExcellentCase() {
    setMessage(null);
    const response = await excellentMutation.mutateAsync({
      caseId: caseState.case_id,
      operatorId: "operator",
      notes: "运营从异常订单工作台沉淀。",
    });
    setMessage(`优秀案例沉淀任务已提交：${response.data.ingestion_job.job_id}`);
    onCaseUpdated(caseState);
  }

  async function simulateTask(task: RoutedExternalTask, status: "COMPLETED" | "FAILED") {
    setMessage(null);
    const response = await webhookMutation.mutateAsync({
      taskId: task.task_id,
      status,
      eventId: makeWebhookEventId(task.task_id),
      caseVersion: caseState.case_version || task.case_version || 1,
      planVersion: task.plan_version || caseState.plan_version || 1,
      idempotencyKey: task.idempotency_key,
      message: status === "COMPLETED" ? "外部系统模拟处理完成。" : "外部系统模拟处理失败。",
      result: buildExternalTaskBusinessResult(task, status),
    });
    setCaseState(response.data.case);
    setReplanProposal(null);
    setMessage(`任务回调已接收，当前案件状态：${response.data.case.case_status}`);
    onCaseUpdated(response.data.case);
  }

  return (
    <section className="ops-execution-card">
      <div className="ops-execution-card__head">
        <div>
          <span>执行跟踪</span>
          <strong>{caseState.case_id}</strong>
        </div>
        <DotBadge tone={caseStatusTone(caseState.case_status)}>{caseState.case_status}</DotBadge>
      </div>
      <div className="ops-execution-card__meta">
        <span>Case v{caseState.case_version || 1}</span>
        <span>Plan v{caseState.plan_version || caseState.plan.plan_version || 1}</span>
        <span>Replan {caseState.replan_count || 0}/3</span>
      </div>
      <div className="ops-execution-tasks">
        {caseState.tasks.map((task) => (
          <article key={task.task_id}>
            <div>
              <strong>{task.target_system} · {actionTypeLabel(task.action_type)}</strong>
              <span>{task.external_task_type} · {task.status}</span>
              <small>{task.task_id}</small>
            </div>
            <div>
              <button type="button" disabled={webhookMutation.isPending || task.status === "COMPLETED"} onClick={() => void simulateTask(task, "COMPLETED")}>
                完成
              </button>
              <button className="danger" type="button" disabled={webhookMutation.isPending || task.status === "FAILED"} onClick={() => void simulateTask(task, "FAILED")}>
                失败
              </button>
            </div>
          </article>
        ))}
      </div>
      <div className="ops-execution-card__actions">
        <button type="button" disabled={verifyMutation.isPending} onClick={() => void verifyCase()}>
          <FileCheck2 size={15} />
          验证结果
        </button>
        <button type="button" disabled={excellentMutation.isPending} onClick={() => void saveExcellentCase()}>
          <Database size={15} />
          沉淀案例
        </button>
      </div>
      {message && <p>{message}</p>}
      {replanProposal && (
        <div className="ops-replan-box">
          <strong>需要重新规划</strong>
          <span>{replanProposal.summary}</span>
          <button type="button" onClick={() => onReplanProposal(replanProposal)}>
            <RefreshCcw size={15} />
            重新分析
          </button>
        </div>
      )}
    </section>
  );
}

function AnalysisLog({ events }: { events: AnalysisEvent[] }) {
  const visibleEvents = events.filter((event) => event.id !== "welcome").slice(-3);
  if (!visibleEvents.length) return null;

  return (
    <section className="ops-analysis-log">
      <SectionHeader title="最近分析记录" meta={`${visibleEvents.length} 条`} />
      {visibleEvents.map((event) => (
        <article className={`ops-analysis-log__item ops-analysis-log__item--${event.role}`} key={event.id}>
          <span>{event.role === "agent" ? "AI 分析" : event.role === "operator" ? "运营动作" : "系统事件"}</span>
          <p>{event.content}</p>
        </article>
      ))}
    </section>
  );
}

function OperationsWorkspace({
  order,
  proposal,
  activeCase,
  latestEvidence,
  events,
  analyzing,
  review,
  reviewNotes,
  reviewError,
  onAnalyzeOrder,
  onOpenReview,
  onProposalChange,
  onNotesChange,
  onSubmitReview,
  onCaseUpdated,
}: {
  order?: OrderRecord;
  proposal?: ExecutionProposal | null;
  activeCase?: FulfillmentCase | null;
  latestEvidence?: OpsMessageEvidence | null;
  events: AnalysisEvent[];
  analyzing: boolean;
  review: ReviewState | null;
  reviewNotes: string;
  reviewError: string | null;
  onAnalyzeOrder: (orderId: string) => void;
  onOpenReview: (evidence: OpsMessageEvidence, decision: ReviewDecision) => void;
  onProposalChange: (proposal: ExecutionProposal) => void;
  onNotesChange: (value: string) => void;
  onSubmitReview: (decision: ReviewDecision) => void;
  onCaseUpdated: (fulfillmentCase: FulfillmentCase) => void;
}) {
  return (
    <main className="ops-main-workspace">
      <OrderSummary
        order={order}
        proposal={proposal}
        caseItem={activeCase}
        analyzing={analyzing}
        onAnalyzeOrder={onAnalyzeOrder}
      />
      <div className="ops-workspace-content">
        <div className="ops-workspace-content__left">
          <OrderContext order={order} proposal={proposal} activeCase={activeCase} />
          <AnalysisLog events={events} />
        </div>
        <AIDecisionPanel
          order={order}
          evidence={latestEvidence}
          proposal={proposal}
          activeCase={activeCase}
          loading={analyzing}
          review={review}
          reviewNotes={reviewNotes}
          reviewError={reviewError}
          onAnalyzeOrder={onAnalyzeOrder}
          onOpenReview={onOpenReview}
          onProposalChange={onProposalChange}
          onNotesChange={onNotesChange}
          onSubmitReview={onSubmitReview}
          onCaseUpdated={onCaseUpdated}
        />
      </div>
    </main>
  );
}

export function OperationsDashboard() {
  const queryClient = useQueryClient();
  const [events, setEvents] = useState<AnalysisEvent[]>(loadEvents);
  const [selectedOrderId, setSelectedOrderId] = useState("");
  const [manualProposal, setManualProposal] = useState<ExecutionProposal | null>(null);
  const [review, setReview] = useState<ReviewState | null>(null);
  const [reviewNotes, setReviewNotes] = useState("");
  const [reviewError, setReviewError] = useState<string | null>(null);
  const typingTimerRef = useRef<number | null>(null);

  const ordersQuery = useQuery({ queryKey: ["enterprise-orders", 50], queryFn: () => listEnterpriseOrders(50) });
  const casesQuery = useQuery({
    queryKey: ["fulfillment-cases", 30],
    queryFn: () => listFulfillmentCases({ limit: 30 }),
  });
  const runMutation = useMutation({ mutationFn: runHybrid });
  const resumeMutation = useMutation({ mutationFn: resumeHybrid });
  const confirmCaseMutation = useMutation({ mutationFn: confirmFulfillmentCase });

  const knownOrderIds = useMemo(
    () => new Set((ordersQuery.data?.data.orders || []).map((order) => order.order_id)),
    [ordersQuery.data],
  );
  const latestEvidence = useMemo(
    () => [...events].reverse().find((event) => event.evidence)?.evidence || null,
    [events],
  );
  const abnormalOrders = useMemo(
    () => (ordersQuery.data?.data.orders || []).filter((order) => isAbnormalOrder(order)),
    [ordersQuery.data],
  );
  const selectedOrder = useMemo(() => {
    const orders = ordersQuery.data?.data.orders || [];
    const effectiveOrderId =
      selectedOrderId
      || latestEvidence?.actionCard?.order_id
      || latestEvidence?.fulfillmentCase?.order_id
      || latestEvidence?.orderId
      || abnormalOrders[0]?.order_id
      || orders[0]?.order_id
      || "";
    return orders.find((order) => order.order_id === effectiveOrderId) || abnormalOrders[0] || orders[0];
  }, [abnormalOrders, latestEvidence, ordersQuery.data, selectedOrderId]);
  const activeCase = useMemo(() => {
    const evidenceCase = latestEvidence?.fulfillmentCase;
    if (evidenceCase && selectedOrder?.order_id && evidenceCase.order_id === selectedOrder.order_id) return evidenceCase;
    return (casesQuery.data?.data.cases || []).find((caseItem) => caseItem.order_id === selectedOrder?.order_id && isPendingFulfillmentCase(caseItem)) || null;
  }, [casesQuery.data, latestEvidence, selectedOrder]);
  const activeProposal = useMemo(() => {
    const proposal = manualProposal || latestEvidence?.actionCard || activeCase?.plan || null;
    if (!proposal) return null;
    return proposal.order_id === selectedOrder?.order_id ? proposal : null;
  }, [activeCase, latestEvidence, manualProposal, selectedOrder]);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(events.slice(-40)));
  }, [events]);

  useEffect(() => {
    return () => {
      if (typingTimerRef.current) window.clearInterval(typingTimerRef.current);
    };
  }, []);

  function appendEvent(event: Omit<AnalysisEvent, "id" | "createdAt">) {
    const next: AnalysisEvent = { ...event, id: makeId(event.role), createdAt: new Date().toISOString() };
    setEvents((current) => [...current, next]);
    return next.id;
  }

  function typeAgentEvent(eventId: string, text: string) {
    if (typingTimerRef.current) window.clearInterval(typingTimerRef.current);
    const chars = Array.from(text);
    let index = 0;
    setEvents((current) => current.map((event) => (event.id === eventId ? { ...event, content: "", status: "typing" } : event)));
    typingTimerRef.current = window.setInterval(() => {
      index += 1;
      setEvents((current) =>
        current.map((event) =>
          event.id === eventId
            ? { ...event, content: chars.slice(0, index).join(""), status: index >= chars.length ? "done" : "typing" }
            : event,
        ),
      );
      if (index >= chars.length && typingTimerRef.current) {
        window.clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
      }
    }, 8);
  }

  function attachEvidence(eventId: string, evidence: OpsMessageEvidence) {
    setEvents((current) => current.map((event) => (event.id === eventId ? { ...event, evidence } : event)));
  }

  function handleCaseUpdated(fulfillmentCase: FulfillmentCase) {
    setEvents((current) =>
      current.map((event) => {
        if (event.evidence?.fulfillmentCase?.case_id !== fulfillmentCase.case_id) return event;
        return {
          ...event,
          evidence: {
            ...event.evidence,
            status: fulfillmentCase.case_status,
            fulfillmentCase,
          },
        };
      }),
    );
    void queryClient.invalidateQueries({ queryKey: ["fulfillment-cases"] });
  }

  function openCaseFromQueue(fulfillmentCase: FulfillmentCase) {
    setReview(null);
    setReviewError(null);
    setSelectedOrderId(fulfillmentCase.order_id);
    const evidence = buildRecoveredCaseEvidence(fulfillmentCase);
    appendEvent({
      role: "system",
      content: `已打开外部任务 ${fulfillmentCase.case_id}，当前状态 ${fulfillmentCase.case_status}。`,
      status: "done",
      evidence,
    });
  }

  async function runAnalysisRequest(orderId: string) {
    if (!orderId || runMutation.isPending) return;
    const content = buildOrderActionPrompt(orderId);
    const clientInputAt = new Date().toISOString();
    setSelectedOrderId(orderId);
    setManualProposal(null);
    setReview(null);
    setReviewNotes("");
    setReviewError(null);
    appendEvent({
      role: "operator",
      content: `开始分析订单 ${orderId}`,
      status: "done",
    });
    const agentEventId = appendEvent({
      role: "agent",
      content: "正在读取订单、库存、物流、规则和客户上下文，准备生成异常处置方案...",
      status: "typing",
    });

    try {
      const requestStartedAt = new Date().toISOString();
      const response = await runMutation.mutateAsync({
        orderId,
        question: buildProfessionalQuestion(content),
        clientInputAt,
      });
      const requestEndedAt = new Date().toISOString();
      const result = response.data;
      const answer = finalAnswer(result);
      const evidence = buildMessageEvidence(result, { clientInputAt, requestStartedAt, requestEndedAt });
      attachEvidence(agentEventId, evidence);
      setManualProposal(evidence.actionCard || null);
      typeAgentEvent(agentEventId, answer);

      if (result.status === "interrupted" && result.interrupt) {
        const interrupt = result.interrupt;
        const proposal = extractActionCard(result);
        setReview({
          orderId: result.order_id || orderId,
          threadId: result.interrupt?.thread_id || result.thread_id,
          riskLevel: interrupt.risk_level,
          reasons: interrupt.risk_signals?.length ? interrupt.risk_signals : [interrupt.prompt],
          suggestion: answer,
          proposal,
          preflight: result.preflight_validation,
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "智能分析失败，请检查后端服务、模型配置或网络连接。";
      setEvents((current) =>
        current.map((event) =>
          event.id === agentEventId
            ? {
              ...event,
              content: `分析失败：${message}`,
              status: "error",
              evidence: {
                orderId,
                route: "Hybrid Routing",
                status: "error",
                tools: [],
                sources: ["后端服务"],
                notes: [
                  "请求未完成，未能返回工具和规则依据。",
                  orderId && !knownOrderIds.has(orderId) ? "该订单号不在当前企业订单列表中，请先导入 OMS 订单。" : "",
                ].filter(Boolean),
              },
            }
            : event,
        ),
      );
    }
  }

  function openReviewFromEvidence(evidence: OpsMessageEvidence, decision: ReviewDecision) {
    if (!evidence.actionCard) return;
    setReviewError(null);
    setReviewNotes(
      decision === "approved"
        ? "已确认执行提案，批准进入执行前二次校验。"
        : decision === "modify"
          ? "需要调整仓库、物流渠道或缺货处置方案后重新规划。"
          : "风险未解除，暂不释放履约动作。",
    );
    setReview({
      orderId: evidence.actionCard.order_id || evidence.orderId || "",
      threadId: evidence.threadId,
      riskLevel: evidence.status === "interrupted" ? "HITL" : "INFO",
      reasons: evidence.actionCard.capabilities.length ? evidence.actionCard.capabilities : ["待人工确认"],
      suggestion: evidence.preflight?.message || evidence.actionCard.summary,
      proposal: evidence.actionCard,
      preflight: evidence.preflight,
    });
  }

  function handleProposalChange(proposal: ExecutionProposal) {
    setManualProposal(proposal);
    setReview((current) =>
      current?.proposal?.proposal_id === proposal.proposal_id
        ? { ...current, proposal, suggestion: proposal.summary }
        : current,
    );
  }

  async function submitReview(decision: ReviewDecision) {
    if (!review) return;
    const effectiveNotes = reviewNotes.trim()
      || (decision === "approved" ? "已确认执行提案，批准进入执行前二次校验。" : "未批准当前执行提案，等待人工调整。");
    setReviewError(null);
    try {
      let result: HybridRunResult | null = null;
      let fulfillmentCase: FulfillmentCase | null = null;
      if (review.threadId) {
        const response = await resumeMutation.mutateAsync({
          threadId: review.threadId,
          decision,
          notes: effectiveNotes,
        });
        result = response.data;
      }
      const proposalForCase = result ? extractActionCard(result) || review.proposal : review.proposal;
      const preflightForCase = result?.preflight_validation || review.preflight || null;
      if (decision === "approved" && proposalForCase && preflightForCase?.status === "valid") {
        const caseResponse = await confirmCaseMutation.mutateAsync({
          proposal: proposalForCase,
          preflightValidation: preflightForCase,
          approverId: "operator",
          notes: effectiveNotes,
          checkpoint: {
            thread_id: review.threadId,
            review_source: "operations_dashboard",
          },
        });
        fulfillmentCase = caseResponse.data.case;
        handleCaseUpdated(fulfillmentCase);
      }

      const summary =
        decision === "approved"
          ? fulfillmentCase
            ? `人工审核已批准：${review.orderId} 二次校验通过，已创建外部任务并进入等待状态。备注：${effectiveNotes}`
            : `人工审核已批准：${review.orderId} 已触发执行前二次校验。备注：${effectiveNotes}`
          : decision === "modify"
            ? `人工要求修改方案：${review.orderId} 暂不释放履约。备注：${effectiveNotes}`
            : decision === "ask_followup"
              ? `人工选择补充事实：${review.orderId} 暂不审批。备注：${effectiveNotes}`
              : `人工已拒绝：${review.orderId} 暂不释放履约。备注：${effectiveNotes}`;
      const eventId = appendEvent({ role: "system", content: summary, status: "done" });
      if (result) {
        attachEvidence(eventId, {
          ...buildMessageEvidence(result),
          fulfillmentCase,
        });
      }
      setReview(null);
      setReviewNotes("");
    } catch (error) {
      setReviewError(error instanceof Error ? error.message : "审核提交失败，请稍后重试。");
    }
  }

  return (
    <div className="ops-page">
      <OrderQueue
        orders={ordersQuery.data?.data.orders || []}
        cases={casesQuery.data?.data.cases || []}
        loadingOrders={ordersQuery.isLoading}
        loadingCases={casesQuery.isLoading}
        selectedOrderId={selectedOrder?.order_id}
        onSelectOrder={(orderId) => {
          setSelectedOrderId(orderId);
          setReview(null);
          setReviewNotes("");
          setReviewError(null);
        }}
        onAnalyzeOrder={(orderId) => void runAnalysisRequest(orderId)}
        onSelectCase={openCaseFromQueue}
      />
      <OperationsWorkspace
        order={selectedOrder}
        proposal={activeProposal}
        activeCase={activeCase}
        latestEvidence={latestEvidence}
        events={events}
        analyzing={runMutation.isPending}
        review={review}
        reviewNotes={reviewNotes}
        reviewError={reviewError}
        onAnalyzeOrder={(orderId) => void runAnalysisRequest(orderId)}
        onOpenReview={openReviewFromEvidence}
        onProposalChange={handleProposalChange}
        onNotesChange={setReviewNotes}
        onSubmitReview={(decision) => void submitReview(decision)}
        onCaseUpdated={handleCaseUpdated}
      />
    </div>
  );
}
