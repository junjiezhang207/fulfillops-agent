import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ClipboardCheck, PauseCircle, ShieldAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { EmptyState } from "../components/EmptyState";
import { ReviewPanel } from "../components/ReviewPanel";
import { StatusPill } from "../components/StatusPill";
import { getHitlStats, listPendingApprovals, resumeHybrid, type ApprovalAuditEntry } from "../lib/api";
import { useReviewStore, type PendingReview } from "../store/reviewStore";

type ReviewRow = {
  orderId: string;
  riskLevel: string;
  reason: string;
  platform: string;
  action: string;
  sla: string;
  threadId: string;
};

function riskToneFromLevel(level: string) {
  if (level === "CRITICAL" || level === "HIGH") return "danger";
  if (level === "MEDIUM") return "warn";
  if (level === "LOW") return "ok";
  return "neutral";
}

function rowsFromApprovals(items: ApprovalAuditEntry[]): ReviewRow[] {
  return items.map((item) => ({
    orderId: item.order_id,
    riskLevel: item.risk_level,
    reason: item.risk_signals?.join(" / ") || item.reason || "等待人工确认",
    platform: "OMS/WMS",
    action: "人工确认后恢复履约",
    sla: item.requested_at ? `请求时间 ${new Date(item.requested_at).toLocaleString()}` : "待处理",
    threadId: item.thread_id,
  }));
}

function ReviewDetail({ row }: { row?: ReviewRow }) {
  const queryClient = useQueryClient();
  const [decision, setDecision] = useState<"approved" | "rejected">("approved");
  const [notes, setNotes] = useState("");

  const mutation = useMutation({
    mutationFn: async () => {
      if (!row) return null;
      if (!notes.trim()) throw new Error("请先填写审核备注，说明通过或暂停的原因。");
      const response = await resumeHybrid({
        threadId: row.threadId,
        decision,
        notes: notes.trim(),
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pending-approvals"] });
      queryClient.invalidateQueries({ queryKey: ["hitl-stats"] });
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

      <button className="primary-button full-width" type="button" disabled={mutation.isPending} onClick={() => mutation.mutate()}>
        {mutation.isPending ? "提交中..." : "提交审核结论"}
      </button>
    </section>
  );
}

export function ReviewQueuePage() {
  const pendingApprovals = useQuery({ queryKey: ["pending-approvals"], queryFn: () => listPendingApprovals() });
  const hitlStats = useQuery({ queryKey: ["hitl-stats"], queryFn: getHitlStats });
  const rows = useMemo(() => rowsFromApprovals(pendingApprovals.data?.data.items || []), [pendingApprovals.data]);
  const [selectedOrderId, setSelectedOrderId] = useState(rows[0]?.orderId || "");
  const selectedRow = rows.find((row) => row.orderId === selectedOrderId) || rows[0];
  const highRiskCount = rows.filter((row) => ["HIGH", "CRITICAL"].includes(row.riskLevel.toUpperCase())).length;
  const processedCount =
    (hitlStats.data?.data.approved_count || 0)
    + (hitlStats.data?.data.rejected_count || 0)
    + (hitlStats.data?.data.escalated_count || 0);

  return (
    <div className="page page--demo">
      <header className="page-header demo-hero demo-hero--focused">
        <div>
          <p className="eyebrow">HITL 人工决策</p>
          <h1>人工审查队列</h1>
          <p>当系统遇到高风险、缺货、跨仓调拨或替代 SKU 时，会暂停自动履约，并等待人工确认后继续流程。</p>
        </div>
        <div className="demo-metric-row">
          <article className="demo-metric"><span>真实待审</span><strong>{rows.length}</strong><small>来自 MySQL HITL</small></article>
          <article className="demo-metric"><span>高风险任务</span><strong>{highRiskCount}</strong><small>HIGH / CRITICAL</small></article>
          <article className="demo-metric"><span>累计已处理</span><strong>{processedCount}</strong><small>通过 / 暂停 / 升级</small></article>
        </div>
      </header>

      {rows.length === 0 && (
        <section className="workspace empty-callout">
          <EmptyState
            icon={ClipboardCheck}
            title="当前没有真实待审任务"
            description="从智能履约页触发高风险订单后，HITL 任务会写入 MySQL 并进入这里。"
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
            {pendingApprovals.isLoading && <div className="empty-mini">正在读取待审任务...</div>}
            {pendingApprovals.isError && <div className="error-box">待审任务加载失败，请检查后端和 MySQL 配置。</div>}
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
