import { useQuery } from "@tanstack/react-query";
import { Activity, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { StatusPill } from "../components/StatusPill";
import {
  getBusinessTrace,
  listBusinessTraces,
  type BusinessTrace,
  type BusinessTraceStep,
} from "../lib/api";
import type { StatusTone } from "../lib/status";

function formatDuration(ms?: number | null) {
  if (!ms) return "--";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function formatDate(value?: string | null) {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function statusTone(status?: string | null): StatusTone {
  if (status === "error") return "danger";
  if (status === "interrupted" || status === "pending_human") return "warn";
  if (status === "success" || status === "completed") return "ok";
  return "neutral";
}

function stepEvidence(step: BusinessTraceStep) {
  return Array.isArray(step.evidence) ? step.evidence.slice(0, 3) : [];
}

function hasStepDetails(step: BusinessTraceStep) {
  return Boolean(step.input_summary || step.output_summary || Object.keys(step.metadata || {}).length > 0);
}

function formatTraceValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "--";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function TraceStepDetails({ step }: { step: BusinessTraceStep }) {
  if (!hasStepDetails(step)) return null;
  return (
    <details className="trace-step-details">
      <summary>展开完整字段</summary>
      {step.input_summary && (
        <pre>
          <strong>input</strong>
          {step.input_summary}
        </pre>
      )}
      {step.output_summary && (
        <pre>
          <strong>output</strong>
          {step.output_summary}
        </pre>
      )}
      {Object.keys(step.metadata || {}).length > 0 && (
        <pre>
          <strong>metadata</strong>
          {formatTraceValue(step.metadata)}
        </pre>
      )}
    </details>
  );
}

type TraceTreeNode = {
  step: BusinessTraceStep;
  children: TraceTreeNode[];
};

function buildStepTree(steps: BusinessTraceStep[]): TraceTreeNode[] {
  const nodes = new Map<string, TraceTreeNode>();
  const roots: TraceTreeNode[] = [];
  steps.forEach((step) => nodes.set(step.id, { step, children: [] }));
  steps.forEach((step) => {
    const node = nodes.get(step.id);
    if (!node) return;
    const parent = step.parent_id ? nodes.get(step.parent_id) : null;
    if (parent) {
      parent.children.push(node);
    } else {
      roots.push(node);
    }
  });
  return roots;
}

function metadataValue(step: BusinessTraceStep, key: string) {
  return step.metadata ? step.metadata[key] : undefined;
}

function TraceStepCard({ node, depth = 0 }: { node: TraceTreeNode; depth?: number }) {
  const step = node.step;
  const modelId = metadataValue(step, "model_id");
  const promptId = metadataValue(step, "prompt_id");
  const promptVersion = metadataValue(step, "prompt_version");
  const hasModelMetadata = Boolean(modelId || promptId || promptVersion);
  return (
    <article
      className={`trace-center-step trace-center-step--${statusTone(step.status)}`}
      style={{ marginLeft: depth ? `${Math.min(depth * 18, 54)}px` : undefined }}
    >
      <div className="trace-center-step__head">
        <div>
          <strong>{step.name}</strong>
          <span>
            {step.type} / {formatDuration(step.duration_ms)} / {formatDate(step.started_at)}
          </span>
        </div>
        <StatusPill tone={statusTone(step.status)}>{step.status}</StatusPill>
      </div>
      {hasModelMetadata && (
        <div className="trace-model-strip">
          {Boolean(modelId) && <span>model: {String(modelId)}</span>}
          {Boolean(promptId) && <span>prompt: {String(promptId)}</span>}
          {Boolean(promptVersion) && <span>v{String(promptVersion)}</span>}
        </div>
      )}
      {step.summary && <p>{step.summary}</p>}
      {step.error_message && <p className="trace-error-text">{step.error_message}</p>}
      {stepEvidence(step).length > 0 && (
        <div className="trace-center-evidence">
          {stepEvidence(step).map((item, index) => (
            <span key={`${String(item.chunk_id || item.source_file || index)}`}>
              {String(item.source_file || "--")} / {String(item.chunk_id || "--")} / {String(item.score ?? "--")}
            </span>
          ))}
        </div>
      )}
      <TraceStepDetails step={step} />
      {node.children.map((child) => (
        <TraceStepCard key={child.step.id} node={child} depth={depth + 1} />
      ))}
    </article>
  );
}

function TraceDetail({ trace }: { trace: BusinessTrace }) {
  const steps = trace.steps || [];
  const stepTree = buildStepTree(steps);
  const llmSteps = steps.filter((step) => step.type === "llm");
  const evalSteps = steps.filter((step) => step.type === "evaluation");
  const agentSteps = steps.filter((step) => step.type === "agent" || step.type === "agent_iteration");

  return (
    <section className="workspace trace-detail">
      <div className="section-title">
        <div>
          <h2>Trace 详情</h2>
          <p>{trace.trace_id}</p>
        </div>
        <StatusPill tone={statusTone(trace.status)}>{trace.status || "--"}</StatusPill>
      </div>

      <div className="trace-summary-grid">
        <article>
          <span>订单</span>
          <strong>{trace.order_id || "--"}</strong>
        </article>
        <article>
          <span>路由</span>
          <strong>{trace.route || "--"}</strong>
        </article>
        <article>
          <span>总耗时</span>
          <strong>{formatDuration(trace.duration_ms)}</strong>
        </article>
        <article>
          <span>开始时间</span>
          <strong>{formatDate(trace.started_at)}</strong>
        </article>
      </div>

      <div className="trace-summary-grid trace-summary-grid--dense">
        <article>
          <span>LLM 调用</span>
          <strong>{llmSteps.length}</strong>
        </article>
        <article>
          <span>Agent 步骤</span>
          <strong>{agentSteps.length}</strong>
        </article>
        <article>
          <span>评估快照</span>
          <strong>{evalSteps.length}</strong>
        </article>
        <article>
          <span>Step 总数</span>
          <strong>{steps.length}</strong>
        </article>
      </div>

      <div className="trace-center-timeline">
        {stepTree.map((node) => <TraceStepCard key={node.step.id} node={node} />)}
        {steps.length === 0 && <p className="trace-empty">这条 Trace 暂时没有步骤明细。</p>}
      </div>

      {trace.audit_events?.length ? (
        <div className="trace-audit-panel">
          <strong>HITL / Audit</strong>
          {trace.audit_events.map((event) => (
            <p key={event.id}>
              {formatDate(event.created_at)} / {event.event_type} / {event.action || "--"} /{" "}
              {event.summary || "--"}
            </p>
          ))}
        </div>
      ) : null}

      <details className="trace-raw-json">
        <summary>查看原始 Trace JSON</summary>
        <pre>{JSON.stringify(trace, null, 2)}</pre>
      </details>
    </section>
  );
}

export function TraceCenterPage() {
  const navigate = useNavigate();
  const { traceId } = useParams<{ traceId: string }>();
  const [orderId, setOrderId] = useState("");
  const [status, setStatus] = useState("");
  const [toolName, setToolName] = useState("");
  const [route, setRoute] = useState("");
  const [stepType, setStepType] = useState("");
  const [modelId, setModelId] = useState("");
  const [promptId, setPromptId] = useState("");
  const [minDurationMs, setMinDurationMs] = useState("");
  const selectedTraceId = traceId ? decodeURIComponent(traceId) : null;

  const filters = useMemo(
    () => ({
      orderId: orderId.trim() || undefined,
      status: status.trim() || undefined,
      toolName: toolName.trim() || undefined,
      route: route.trim() || undefined,
      stepType: stepType.trim() || undefined,
      modelId: modelId.trim() || undefined,
      promptId: promptId.trim() || undefined,
      minDurationMs: minDurationMs.trim() ? Number(minDurationMs) : undefined,
      limit: 80,
    }),
    [minDurationMs, modelId, orderId, promptId, route, status, stepType, toolName],
  );

  const traces = useQuery({
    queryKey: ["business-traces", filters],
    queryFn: () => listBusinessTraces(filters),
  });
  const detail = useQuery({
    queryKey: ["business-trace", selectedTraceId],
    queryFn: () => getBusinessTrace(selectedTraceId || ""),
    enabled: Boolean(selectedTraceId),
  });

  const rows = traces.data?.data.traces || [];

  return (
    <div className="page page--wide trace-center-page">
      <section className="workspace trace-filter-bar">
        <div className="section-title">
          <div>
            <h2>Trace Center</h2>
            <p>按订单、路由、模型、Prompt、步骤类型和耗时定位一次业务决策链路。</p>
          </div>
          <Activity size={20} />
        </div>
        <div className="trace-filter-grid">
          <label>
            <span>订单号</span>
            <input value={orderId} onChange={(event) => setOrderId(event.target.value)} />
          </label>
          <label>
            <span>状态</span>
            <input
              value={status}
              onChange={(event) => setStatus(event.target.value)}
              placeholder="success / error"
            />
          </label>
          <label>
            <span>工具名</span>
            <input value={toolName} onChange={(event) => setToolName(event.target.value)} />
          </label>
          <label>
            <span>路由</span>
            <input value={route} onChange={(event) => setRoute(event.target.value)} placeholder="hybrid:workflow" />
          </label>
          <label>
            <span>步骤类型</span>
            <input value={stepType} onChange={(event) => setStepType(event.target.value)} placeholder="llm / agent / evaluation" />
          </label>
          <label>
            <span>模型 ID</span>
            <input value={modelId} onChange={(event) => setModelId(event.target.value)} placeholder="deepseek-v4-flash" />
          </label>
          <label>
            <span>Prompt ID</span>
            <input value={promptId} onChange={(event) => setPromptId(event.target.value)} placeholder="rag_rewrite" />
          </label>
          <label>
            <span>最小耗时 ms</span>
            <input value={minDurationMs} onChange={(event) => setMinDurationMs(event.target.value)} inputMode="numeric" />
          </label>
          <button className="primary-button" type="button" onClick={() => void traces.refetch()}>
            <Search size={18} />
            筛选
          </button>
        </div>
      </section>

      <div className="trace-center-layout">
        <section className="workspace trace-list-panel">
          <div className="section-title">
            <div>
              <h2>Trace 列表</h2>
              <p>{traces.isLoading ? "加载中" : `${rows.length} 条记录`}</p>
            </div>
            <StatusPill tone={traces.isError ? "danger" : "ok"}>
              {traces.isError ? "异常" : "可查询"}
            </StatusPill>
          </div>
          <div className="trace-list">
            {rows.map((trace) => (
              <button
                className={trace.trace_id === selectedTraceId ? "trace-row trace-row--active" : "trace-row"}
                key={trace.trace_id}
                type="button"
                onClick={() => navigate(`/traces/${encodeURIComponent(trace.trace_id)}`)}
              >
                <strong>{trace.order_id || trace.trace_id}</strong>
                <span>
                  {trace.route || "--"} / {formatDuration(trace.duration_ms)} /{" "}
                  {formatDate(trace.started_at)}
                </span>
                <StatusPill tone={statusTone(trace.status)}>{trace.status || "--"}</StatusPill>
              </button>
            ))}
            {!traces.isLoading && rows.length === 0 && <p className="trace-empty">没有匹配的 Trace。</p>}
          </div>
        </section>

        {detail.isLoading ? (
          <section className="workspace trace-detail trace-detail--empty">
            <strong>正在加载 Trace 详情</strong>
            <span>正在读取步骤时间线与审计记录。</span>
          </section>
        ) : detail.isError ? (
          <section className="workspace trace-detail trace-detail--empty">
            <strong>Trace 详情加载失败</strong>
            <span>{detail.error instanceof Error ? detail.error.message : "请刷新后重试。"}</span>
          </section>
        ) : detail.data?.data ? (
          <TraceDetail trace={detail.data.data} />
        ) : (
          <section className="workspace trace-detail trace-detail--empty">
            <strong>选择一条 Trace 查看时间线</strong>
            <span>这里会展示步骤状态、RAG evidence、错误和 HITL 审批记录。</span>
          </section>
        )}
      </div>
    </div>
  );
}
