import { useMutation } from "@tanstack/react-query";
import { ClipboardCheck, PauseCircle, ShieldAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { EmptyState } from "../components/EmptyState";
import { ReviewPanel } from "../components/ReviewPanel";
import { StatusPill } from "../components/StatusPill";
import { resumeHybrid } from "../lib/api";
import { mockReviewRows, riskToneFromLevel, type RiskLevel } from "../lib/mockData";
import { useReviewStore, type PendingReview } from "../store/reviewStore";

type ReviewRow = {
  orderId: string;
  riskLevel: RiskLevel | string;
  reason: string;
  platform: string;
  action: string;
  sla: string;
  pending?: PendingReview;
};

function rowsFromQueue(pendingReviews: PendingReview[]): ReviewRow[] {
  if (pendingReviews.length === 0) return mockReviewRows;

  return pendingReviews.map((review) => ({
    orderId: review.orderId,
    riskLevel: review.interrupt.risk_level,
    reason: review.interrupt.risk_signals?.join(" / ") || review.interrupt.prompt,
    platform: String(review.interrupt.context.platform || "OMS"),
    action: "人工确认后恢复履约",
    sla: `${Math.max(1, Math.floor((review.interrupt.timeout_seconds || 1800) / 3600))} 小时内`,
    pending: review,
  }));
}

function ReviewDetail({ row }: { row?: ReviewRow }) {
  const removePendingReview = useReviewStore((state) => state.removePendingReview);
  const [decision, setDecision] = useState<"approved" | "rejected">("approved");
  const [notes, setNotes] = useState("");

  const mutation = useMutation({
    mutationFn: async () => {
      if (!row?.pending) return null;
      if (!notes.trim()) throw new Error("请先填写审核备注，说明通过或暂停的原因。");
      const response = await resumeHybrid({
        threadId: row.pending.threadId,
        decision,
        notes: notes.trim(),
      });
      removePendingReview(row.pending.threadId);
      return response.data;
    },
  });

  if (!row) {
    return (
      <section className="workspace hitl-detail">
        <EmptyState icon={ClipboardCheck} title="选择一条审查任务" description="左侧选择待审任务，查看触发原因和人工处理动作。" />
      </section>
    );
  }

  return (
    <section className="workspace hitl-detail hitl-detail--focused">
      <div className="section-title">
        <div>
          <h2>审查详情</h2>
          <p>{row.orderId} · {row.platform} · SLA {row.sla}</p>
        </div>
        <StatusPill tone={riskToneFromLevel(row.riskLevel)}>{row.riskLevel}</StatusPill>
      </div>

      <div className="hitl-summary">
        <article>
          <span>触发原因</span>
          <strong>{row.reason}</strong>
        </article>
        <article>
          <span>系统建议</span>
          <strong>{row.action}</strong>
        </article>
      </div>

      {!row.pending && (
        <div className="info-box">
          当前为兜底展示任务。真实任务会从智能履约页触发 HITL 后进入这里，并在提交时调用 `/hybrid/resume`。
        </div>
      )}

      <label className="field-label" htmlFor="notes">审核备注</label>
      <textarea
        id="notes"
        value={notes}
        onChange={(event) => setNotes(event.target.value)}
        placeholder="例如：确认替代 SKU 可用，允许恢复履约；或暂停发货并转客服确认。"
        rows={4}
      />

      <div className="segmented-control">
        <button className={decision === "approved" ? "selected" : ""} type="button" onClick={() => setDecision("approved")}>
          通过履约
        </button>
        <button className={decision === "rejected" ? "selected danger" : ""} type="button" onClick={() => setDecision("rejected")}>
          暂停履约
        </button>
      </div>

      {mutation.error && <div className="error-box">{mutation.error instanceof Error ? mutation.error.message : "提交失败"}</div>}
      {mutation.data && <div className="success-box">审核已提交，Hybrid 流程已恢复处理。</div>}

      <button className="primary-button full-width" type="button" disabled={!row.pending || mutation.isPending} onClick={() => mutation.mutate()}>
        {mutation.isPending ? "提交中..." : row.pending ? "提交审核结论" : "仅真实待审任务可提交"}
      </button>
    </section>
  );
}

export function ReviewQueuePage() {
  const pendingReviews = useReviewStore((state) => state.pendingReviews);
  const rows = useMemo(() => rowsFromQueue(pendingReviews), [pendingReviews]);
  const [selectedOrderId, setSelectedOrderId] = useState(rows[0]?.orderId || "");
  const selectedRow = rows.find((row) => row.orderId === selectedOrderId) || rows[0];
  const highRiskCount = rows.filter((row) => ["HIGH", "CRITICAL"].includes(row.riskLevel.toUpperCase())).length;

  return (
    <div className="page page--demo">
      <header className="page-header demo-hero demo-hero--focused">
        <div>
          <p className="eyebrow">HITL 人工决策</p>
          <h1>人工审查队列</h1>
          <p>当系统遇到高风险、缺货、跨仓调拨或替代 SKU 时，会暂停自动履约，并等待人工确认后继续流程。</p>
        </div>
        <div className="demo-metric-row">
          <article className="demo-metric"><span>真实待审</span><strong>{pendingReviews.length}</strong><small>来自 Hybrid 中断</small></article>
          <article className="demo-metric"><span>高风险任务</span><strong>{highRiskCount}</strong><small>含兜底队列</small></article>
          <article className="demo-metric"><span>今日已处理</span><strong>42</strong><small>通过 / 暂停</small></article>
        </div>
      </header>

      {pendingReviews.length === 0 && (
        <section className="workspace empty-callout">
          <EmptyState
            icon={ClipboardCheck}
            title="当前没有真实待审任务"
            description="从智能履约页触发高风险订单后，真实 HITL 任务会进入这里。下方保留兜底队列，避免页面空白。"
            action={<Link className="secondary-button" to="/">去智能履约页</Link>}
          />
        </section>
      )}

      <div className="hitl-layout">
        <section className="workspace">
          <div className="section-title">
            <div>
              <h2>待审任务</h2>
              <p>集中处理需要人工确认的履约中断任务。</p>
            </div>
          </div>
          <div className="hitl-list">
            {rows.map((row) => (
              <button className={row.orderId === selectedRow?.orderId ? "hitl-card selected" : "hitl-card"} key={row.orderId} type="button" onClick={() => setSelectedOrderId(row.orderId)}>
                <div>
                  <strong>{row.orderId}</strong>
                  <span>{row.platform} · {row.reason}</span>
                </div>
                <StatusPill tone={riskToneFromLevel(row.riskLevel)}>{row.riskLevel}</StatusPill>
              </button>
            ))}
          </div>
        </section>

        <ReviewDetail row={selectedRow} />
      </div>

      <section className="demo-bottom-notes">
        <article><ShieldAlert size={18} /><span>风险闸门触发后，流程不会直接发货。</span></article>
        <article><PauseCircle size={18} /><span>人工备注会随 `/hybrid/resume` 回到后端继续执行。</span></article>
        <article><ClipboardCheck size={18} /><span>人工审查完成后，履约流程会继续执行或保持暂停。</span></article>
      </section>
    </div>
  );
}

export function StandaloneReviewPage() {
  const { threadId = "" } = useParams();
  const pendingReviews = useReviewStore((state) => state.pendingReviews);
  const review = pendingReviews.find((item) => item.threadId === threadId) ?? null;

  return (
    <div className="standalone-review">
      {review ? (
        <ReviewPanel review={review} onClose={() => window.close()} />
      ) : (
        <EmptyState
          icon={ShieldAlert}
          title="没有找到这条审查任务"
          description="任务可能已经提交完成，或当前浏览器没有本地待审记录。"
          action={<Link className="secondary-button" to="/reviews">返回审查队列</Link>}
        />
      )}
    </div>
  );
}
