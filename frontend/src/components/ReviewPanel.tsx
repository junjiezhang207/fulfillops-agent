import { CheckCircle2, ExternalLink, History, ShieldAlert, Truck, X } from "lucide-react";
import { useMemo, useState } from "react";

import { resumeHybrid, type HybridRunResult } from "../lib/api";
import { riskTone } from "../lib/status";
import { useReviewStore, type PendingReview } from "../store/reviewStore";
import { StatusPill } from "./StatusPill";

type ReviewPanelProps = {
  review: PendingReview | null;
  onClose: () => void;
  onResolved?: (result: HybridRunResult) => void;
};

function formatValue(value: unknown): string {
  if (value == null || value === "") return "暂无";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value, null, 2);
}

function openReviewWindow(threadId: string) {
  const url = `${window.location.origin}${window.location.pathname}#/review/${encodeURIComponent(threadId)}`;
  window.open(url, "_blank", "noopener,noreferrer,width=1180,height=820");
}

const noteTemplates = [
  "库存、仓库和客服侧信息已确认，可继续执行。",
  "风险信号仍未消除，建议暂停履约并转人工跟进。",
  "需要补充客户或仓库确认后再恢复流程。",
];

export function ReviewPanel({ review, onClose, onResolved }: ReviewPanelProps) {
  const removePendingReview = useReviewStore((state) => state.removePendingReview);
  const [decision, setDecision] = useState<"approved" | "rejected">("approved");
  const [notes, setNotes] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setSubmitting] = useState(false);

  const contextEntries = useMemo(() => {
    if (!review?.interrupt.context) return [];
    return Object.entries(review.interrupt.context).slice(0, 12);
  }, [review]);

  if (!review) return null;

  const canSubmit = notes.trim().length > 0 && !isSubmitting;

  async function submitDecision() {
    if (!review || !canSubmit) return;
    const actionText = decision === "approved" ? "通过履约" : "暂停 / 驳回履约";
    if (!window.confirm(`确认提交“${actionText}”？该操作会恢复或终止当前履约流程。`)) return;
    setSubmitting(true);
    setError(null);
    try {
      const response = await resumeHybrid({
        threadId: review.threadId,
        decision,
        notes: notes.trim(),
      });
      removePendingReview(review.threadId);
      onResolved?.(response.data);
      onClose();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "审批提交失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="review-backdrop" role="dialog" aria-modal="true" aria-label="人工审查">
      <section className="review-panel">
        <header className="review-panel__header">
          <div>
            <p className="eyebrow">风险订单审核</p>
            <h2>订单 {review.orderId}</h2>
            <span>中断节点 {review.interrupt.node}</span>
          </div>
          <div className="review-actions">
            <button className="icon-button" type="button" onClick={() => openReviewWindow(review.threadId)} title="打开独立审查页">
              <ExternalLink size={18} />
            </button>
            <button className="icon-button" type="button" onClick={onClose} title="关闭">
              <X size={18} />
            </button>
          </div>
        </header>

        <div className="review-alert">
          <ShieldAlert size={22} />
          <div>
            <strong>{review.interrupt.prompt || "当前流程需要人工确认后继续。"}</strong>
            <p>
              <StatusPill tone={riskTone(review.interrupt.risk_level)}>
                {review.interrupt.risk_level}
              </StatusPill>
              <span>{review.interrupt.risk_signals?.length || 0} 个风险信号</span>
              <span>线程 {review.threadId}</span>
            </p>
          </div>
        </div>

        <div className="review-decision-strip">
          <div>
            <span>审查目标</span>
            <strong>{decision === "approved" ? "确认风险可控并恢复履约" : "暂停当前履约动作并进入人工跟进"}</strong>
          </div>
          <CheckCircle2 size={22} />
        </div>

        <div className="section-block review-summary-card">
          <h3>订单摘要</h3>
          <dl className="facts-list">
            <div>
              <dt>订单号</dt>
              <dd>{review.orderId}</dd>
            </div>
            <div>
              <dt>平台</dt>
              <dd>{formatValue(review.interrupt.context.platform || "OMS / ERP")}</dd>
            </div>
            <div>
              <dt>客户等级</dt>
              <dd>{formatValue(review.interrupt.context.customer_level || "高价值客户")}</dd>
            </div>
            <div>
              <dt>建议路径</dt>
              <dd>{formatValue(review.interrupt.context.fulfillment_path || "人工复核后恢复")}</dd>
            </div>
          </dl>
        </div>

        <div className="review-grid">
          <div className="section-block">
            <h3>风险信号</h3>
            <div className="tag-row">
              {(review.interrupt.risk_signals || []).length > 0 ? (
                review.interrupt.risk_signals.map((signal) => <span className="tag" key={signal}>{signal}</span>)
              ) : (
                <span className="muted">暂无结构化风险信号</span>
              )}
            </div>
          </div>

          <div className="section-block">
            <h3>审查依据</h3>
            <dl className="facts-list">
              {contextEntries.map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{formatValue(value)}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>

        <div className="review-grid">
          <div className="section-block">
            <h3>系统建议</h3>
            <div className="evidence-card">
              <Truck size={18} />
              <div>
                <strong>{review.sourceResult.final_answer || "建议审核员确认库存、SLA 和替代 SKU 后再释放履约。"}</strong>
                <span>建议动作：{decision === "approved" ? "通过履约" : "暂停 / 驳回履约"}</span>
              </div>
            </div>
          </div>
          <div className="section-block">
            <h3>操作记录</h3>
            <div className="audit-log">
              <span><History size={15} /> 系统触发风险拦截</span>
              <span><History size={15} /> 已进入 HITL 审核队列</span>
              <span><History size={15} /> 等待审核员备注并提交</span>
            </div>
          </div>
        </div>

        <div className="section-block">
          <h3>审查结论</h3>
          <div className="segmented-control" aria-label="审查决策">
            <button className={decision === "approved" ? "selected" : ""} type="button" onClick={() => setDecision("approved")}>
              通过履约
            </button>
            <button className={decision === "rejected" ? "selected danger" : ""} type="button" onClick={() => setDecision("rejected")}>
              暂停 / 驳回履约
            </button>
          </div>
          <p className={decision === "approved" ? "decision-hint decision-hint--ok" : "decision-hint decision-hint--danger"}>
            {decision === "approved"
              ? "批准后流程会继续执行，适合风险已确认可控的订单。"
              : "拒绝后流程会按失败处理，适合库存、规则或客户风险仍未解除的订单。"}
          </p>
          <label className="field-label" htmlFor="review-notes">审批理由</label>
          <div className="template-row template-row--notes">
            {noteTemplates.map((template) => (
              <button key={template} type="button" onClick={() => setNotes(template)}>
                {template}
              </button>
            ))}
          </div>
          <textarea
            id="review-notes"
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="例如：库存已由仓库主管确认，可继续跨仓调拨。"
            rows={4}
          />
          {error && <div className="error-box">{error}</div>}
        </div>

        <footer className="review-panel__footer">
          <button className="secondary-button" type="button" onClick={onClose}>稍后处理</button>
          <button className="primary-button" type="button" disabled={!canSubmit} onClick={submitDecision}>
            {isSubmitting ? "提交中..." : "提交审核结论"}
          </button>
        </footer>
      </section>
    </div>
  );
}
