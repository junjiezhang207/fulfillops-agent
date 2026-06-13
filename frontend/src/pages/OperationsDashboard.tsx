import { useMutation, useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Bot,
  Send,
  Sparkles,
  UserRound,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { StatusPill } from "../components/StatusPill";
import {
  getBusinessTrace,
  listEnterpriseOrders,
  resumeHybrid,
  runHybrid,
  type BusinessTrace,
  type BusinessTraceStep,
  type HybridRunResult,
} from "../lib/api";

type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
  status?: "typing" | "error" | "done";
  evidence?: MessageEvidence;
};

type ReviewState = {
  orderId: string;
  threadId?: string;
  riskLevel: string;
  reasons: string[];
  suggestion: string;
};

type MessageEvidence = {
  orderId?: string;
  route?: string;
  status?: string;
  confidence?: number;
  durationMs?: number;
  fromCache?: boolean;
  tools: string[];
  sources: string[];
  notes: string[];
  trace?: BusinessTrace | null;
  clientInputAt?: string;
  requestStartedAt?: string;
  requestEndedAt?: string;
  apiFailed?: boolean;
};

const STORAGE_KEY = "multiship-ai-chat-history";

const analysisTemplates = [
  "判断订单是否可正常履约，并给出风险与处理建议。",
  "检查库存是否充足，说明是否需要拆单或跨仓调拨。",
  "判断是否需要人工复核，并列出触发原因。",
  "推荐最优发货仓库，兼顾 SLA、库存和运输时效。",
];

const initialMessages: ChatMessage[] = [
  {
    id: "welcome",
    role: "assistant",
    content:
      "你好，我是 Multiship 智能履约助手。你可以直接问我某个订单是否可履约，我会结合 OMS 订单、WMS 库存、SLA、规则知识库和 Hybrid Routing 给出处理建议。",
    createdAt: new Date().toISOString(),
    status: "done",
  },
];

function makeId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function loadMessages() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? (JSON.parse(raw) as ChatMessage[]) : null;
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : initialMessages;
  } catch {
    return initialMessages;
  }
}

function extractOrderId(text: string) {
  const matched = text.match(/\b(?:SO|so)[-\w]*\d[\w-]*/);
  return matched?.[0] || "";
}

function buildProfessionalQuestion(question: string) {
  return question.trim();
}

function cleanAiText(text: string) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/^\s*[-*]\s+/gm, "• ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function finalAnswer(result: HybridRunResult | null) {
  if (typeof result?.final_answer === "string" && result.final_answer.trim()) return cleanAiText(result.final_answer);
  if (result?.status === "interrupted") {
    return "系统已暂停自动履约，请查看返回的人工审核信息，并由人工审核后恢复流程。";
  }
  if (result?.status === "completed") return "分析已完成。";
  return "分析完成，但未返回可展示的结论。";
}

function formatDuration(ms?: number) {
  if (!ms) return "--";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function formatConfidence(value?: number) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "--";
}

function buildMessageEvidence(
  result: HybridRunResult,
  ux?: Pick<MessageEvidence, "clientInputAt" | "requestStartedAt" | "requestEndedAt" | "apiFailed">,
): MessageEvidence {
  const tools = result.tools_called?.length ? result.tools_called : [];
  const sources = new Set<string>();
  const toolText = tools.join(" ").toLowerCase();
  if (/order|oms|订单/.test(toolText) || (result.order_id && result.order_id !== "ADHOC-KNOWLEDGE")) sources.add("OMS 订单");
  if (/inventory|stock|wms|库存/.test(toolText)) sources.add("WMS 库存");
  if (/rag|knowledge|rule|规则/.test(toolText) || result.path_used?.toLowerCase().includes("rag")) sources.add("RAG 规则");
  if (sources.size === 0) {
    sources.add("Hybrid Router");
  }

  const notes = [
    result.status === "interrupted" ? "触发人工审核，中断自动履约流程。" : "未触发人工审核中断。",
    result.from_cache ? "本次结果来自缓存。" : "本次结果由当前请求实时生成。",
  ];

  return {
    orderId: result.order_id === "ADHOC-KNOWLEDGE" ? undefined : result.order_id,
    route: result.path_used || "Hybrid Routing",
    status: result.status,
    confidence: result.confidence,
    durationMs: result.execution_time_ms,
    fromCache: result.from_cache,
    tools,
    sources: [...sources],
    notes,
    trace: result.business_trace,
    ...ux,
  };
}

function ChatWindow({
  messages,
  input,
  loading,
  onInputChange,
  onSend,
  onKeyDown,
}: {
  messages: ChatMessage[];
  input: string;
  loading: boolean;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
}) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  return (
    <section className="workspace chat-shell">
      <div className="section-title chat-title">
        <div>
          <h2>AI 履约对话</h2>
          <p>多轮对话会持续保留，新分析会追加到历史记录中。</p>
        </div>
        <StatusPill tone={loading ? "warn" : "ok"}>{loading ? "分析中" : "可对话"}</StatusPill>
      </div>

      <div className="chat-messages" aria-live="polite">
        {messages.map((message) => (
          <article className={`chat-message chat-message--${message.role}`} key={message.id}>
            <div className="chat-avatar">{message.role === "user" ? <UserRound size={18} /> : <Bot size={18} />}</div>
            <div className="chat-bubble">
              <span>{message.role === "user" ? "运营人员" : "Multiship AI"}</span>
              <MessageContent role={message.role} text={message.content} />
              {message.status === "typing" && <i className="typewriter-cursor" aria-hidden="true" />}
              {message.role === "assistant" && message.status !== "typing" && message.evidence && (
                <MessageEvidencePanel evidence={message.evidence} />
              )}
            </div>
          </article>
        ))}
        <div ref={endRef} />
      </div>

      <div className="chat-composer">
        <textarea
          value={input}
          onChange={(event) => onInputChange(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="输入履约问题，例如：分析订单 SO202605230003 是否需要人工审查"
          rows={3}
        />
        <button className="primary-button" type="button" disabled={loading || !input.trim()} onClick={onSend}>
          {loading ? <Sparkles size={18} /> : <Send size={18} />}
          {loading ? "分析中" : "发送"}
        </button>
      </div>
    </section>
  );
}

function MessageEvidencePanel({ evidence }: { evidence: MessageEvidence }) {
  const [traceExpandedAt, setTraceExpandedAt] = useState<string | null>(null);
  const [fullTrace, setFullTrace] = useState<BusinessTrace | null>(null);
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceError, setTraceError] = useState<string | null>(null);
  const statusTone = evidence.status === "interrupted" ? "warn" : evidence.status === "error" ? "danger" : "ok";
  const traceId = evidence.trace?.trace_id;

  useEffect(() => {
    if (!traceExpandedAt || !traceId || fullTrace || traceLoading) return;
    setTraceLoading(true);
    setTraceError(null);
    getBusinessTrace(traceId)
      .then((response) => setFullTrace(response.data))
      .catch((error: unknown) => setTraceError(error instanceof Error ? error.message : "Trace 详情加载失败"))
      .finally(() => setTraceLoading(false));
  }, [fullTrace, traceExpandedAt, traceId, traceLoading]);

  return (
    <div className="message-evidence">
      <div className="message-evidence__chips">
        <StatusPill tone="info">{evidence.route || "Hybrid Routing"}</StatusPill>
        <StatusPill tone={statusTone}>{evidence.status === "interrupted" ? "需人工审核" : "已完成"}</StatusPill>
        {evidence.sources.slice(0, 3).map((source) => (
          <StatusPill tone="neutral" key={source}>{source}</StatusPill>
        ))}
        <StatusPill tone="neutral">{formatDuration(evidence.durationMs)}</StatusPill>
      </div>

      <details className="message-evidence__details">
        <summary>查看依据</summary>
        <div className="message-evidence__grid">
          <article>
            <span>订单</span>
            <strong>{evidence.orderId || "--"}</strong>
          </article>
          <article>
            <span>置信度</span>
            <strong>{formatConfidence(evidence.confidence)}</strong>
          </article>
          <article>
            <span>缓存</span>
            <strong>{evidence.fromCache ? "命中" : "未命中"}</strong>
          </article>
        </div>
        <div className="message-evidence__list">
          <strong>数据来源</strong>
          <p>{evidence.sources.length ? evidence.sources.join(" / ") : "未返回来源信息"}</p>
        </div>
        <div className="message-evidence__list">
          <strong>工具调用</strong>
          <p>{evidence.tools.length ? evidence.tools.join(" / ") : "未返回工具调用明细"}</p>
        </div>
        <div className="message-evidence__list">
          <strong>处理说明</strong>
          {evidence.notes.map((note) => <p key={note}>{note}</p>)}
        </div>
        {evidence.trace && (
          <details
            className="trace-inline-panel"
            onToggle={(event) => {
              if (event.currentTarget.open && !traceExpandedAt) {
                setTraceExpandedAt(new Date().toISOString());
              }
            }}
          >
            <summary>查看决策链路</summary>
            {traceLoading && <p className="trace-empty">正在加载完整 Trace...</p>}
            {traceError && <p className="trace-error-text">{traceError}</p>}
            <TraceTimeline trace={fullTrace || evidence.trace} />
            <div className="trace-ux-grid">
              <span>输入时间：{formatTime(evidence.clientInputAt)}</span>
              <span>请求开始：{formatTime(evidence.requestStartedAt)}</span>
              <span>请求结束：{formatTime(evidence.requestEndedAt)}</span>
              <span>展开时间：{formatTime(traceExpandedAt)}</span>
            </div>
          </details>
        )}
      </details>
    </div>
  );
}

function formatTime(value?: string | null) {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString();
}

function stepTone(step: BusinessTraceStep) {
  if (step.status === "error") return "danger";
  if (step.status === "interrupted" || step.status === "pending_human") return "warn";
  if (step.status === "skipped") return "neutral";
  return "ok";
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

function TraceTimeline({ trace }: { trace: BusinessTrace }) {
  const steps = trace.steps || [];
  const evidence = steps.flatMap((step) => step.evidence || []).slice(0, 5);
  return (
    <div className="trace-timeline">
      <div className="trace-timeline__head">
        <strong>{trace.trace_id}</strong>
        <span>{formatDuration(trace.duration_ms ?? undefined)} / {trace.route || "route unknown"}</span>
      </div>
      <div className="trace-step-list">
        {steps.length === 0 && <p>本次响应没有返回步骤明细。</p>}
        {steps.map((step) => (
          <article className={`trace-step trace-step--${stepTone(step)}`} key={step.id}>
            <div>
              <strong>{step.name}</strong>
              <span>{step.type} / {step.status} / {formatDuration(step.duration_ms ?? undefined)}</span>
            </div>
            {step.summary && <p>{step.summary}</p>}
            {step.error_message && <p className="trace-step__error">{step.error_message}</p>}
            <TraceStepDetails step={step} />
          </article>
        ))}
      </div>
      {evidence.length > 0 && (
        <div className="trace-evidence-list">
          <strong>RAG evidence</strong>
          {evidence.map((item, index) => (
            <p key={`${String(item.chunk_id || item.source_file || index)}`}>
              {String(item.source_file || "--")} / {String(item.chunk_id || "--")} / score {String(item.score ?? "--")}
            </p>
          ))}
        </div>
      )}
      {trace.audit_events?.length ? (
        <div className="trace-evidence-list">
          <strong>HITL / Audit</strong>
          {trace.audit_events.slice(0, 5).map((event) => (
            <p key={event.id}>{event.event_type} / {event.action || "--"} / {event.summary || "--"}</p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function MessageContent({ role, text }: { role: ChatMessage["role"]; text: string }) {
  const sections = text
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .split(/\n{2,}/)
    .map((section) => section.trim())
    .filter(Boolean);

  if (sections.length === 0) return <p />;

  const content = (
    <div className="message-content">
      {sections.map((section, index) => {
        const [firstLine, ...rest] = section.split(/\n/);
        const titleMatch = firstLine.match(/^(结论|核心结论|关键依据|风险点|风险原因|建议动作|处理建议|人工审核|是否需要人工审核|排查建议|总结)[:：]\s*(.*)$/);
        const tone = sectionTone(titleMatch?.[1] || firstLine);
        if (titleMatch) {
          const body = [titleMatch[2], ...rest].filter(Boolean);
          return (
            <section className={`ai-section ai-section--${tone}`} key={`${firstLine}-${index}`}>
              <div className="ai-section__head">
                <span />
                <strong>{titleMatch[1]}</strong>
              </div>
              {body.length > 0 && <TextLines lines={body} />}
            </section>
          );
        }
        return <TextLines key={`${firstLine}-${index}`} lines={section.split(/\n/)} />;
      })}
    </div>
  );

  if (role !== "assistant" || text.length < 80) return content;

  return (
    <div className="ai-report">
      <div className="ai-report__top">
        <div>
          <strong>履约分析报告</strong>
          <span>Hybrid Routing · OMS / WMS / RAG</span>
        </div>
        <em>AI 建议</em>
      </div>
      {content}
    </div>
  );
}

function TextLines({ lines }: { lines: string[] }) {
  const normalized = lines.map((line) => line.trim()).filter(Boolean);
  if (normalized.length === 0) return null;

  return (
    <div className="message-lines">
      {normalized.map((line, index) => {
        const bullet = line.replace(/^\d+[.、]\s*/, "").replace(/^[-*•]\s*/, "");
        const isList = bullet !== line || line.startsWith("•");
        return isList ? (
          <p className="message-bullet" key={`${line}-${index}`}>
            <span />
            {bullet}
          </p>
        ) : (
          <p key={`${line}-${index}`}>{line}</p>
        );
      })}
    </div>
  );
}

function sectionTone(title: string) {
  if (/风险|审核|暂停|人工/.test(title)) return "warn";
  if (/建议|处理|动作/.test(title)) return "info";
  if (/结论|通过|可履约/.test(title)) return "ok";
  return "neutral";
}

function ReviewModal({
  review,
  notes,
  loading,
  error,
  onNotesChange,
  onClose,
  onSubmit,
}: {
  review: ReviewState;
  notes: string;
  loading: boolean;
  error: string | null;
  onNotesChange: (value: string) => void;
  onClose: () => void;
  onSubmit: (decision: "approved" | "rejected") => void;
}) {
  const approveNote = "已确认订单、库存和履约风险，可继续履约。";
  const rejectNote = "风险未解除，暂停自动履约，等待人工进一步处理。";

  function chooseDecision(decision: "approved" | "rejected") {
    if (!notes.trim()) onNotesChange(decision === "approved" ? approveNote : rejectNote);
    onSubmit(decision);
  }

  return (
    <div className="hitl-modal-backdrop" role="dialog" aria-modal="true" aria-label="人工审核">
      <section className="hitl-modal hitl-modal--decision">
        <div className="hitl-modal__hero">
          <div>
            <p className="eyebrow">HITL 人工确认</p>
            <h2>此订单需要人工审核</h2>
            <p>AI 已暂停自动履约。请选择处理动作，结果会回写当前对话。</p>
          </div>
          <div className="hitl-modal__risk">
            <AlertTriangle size={22} />
            <strong>{review.riskLevel}</strong>
          </div>
        </div>

        <div className="review-alert-mini">
          <strong>{review.orderId}</strong>
          <span>{review.reasons.join(" / ")}</span>
          <p>{review.suggestion}</p>
        </div>

        <div className="review-decision-cards">
          <button type="button" disabled={loading} onClick={() => chooseDecision("approved")}>
            <strong>通过履约</strong>
            <span>风险已确认可接受，继续释放履约流程。</span>
          </button>
          <button className="danger" type="button" disabled={loading} onClick={() => chooseDecision("rejected")}>
            <strong>暂停履约</strong>
            <span>风险未解除，暂停发货并进入人工处理。</span>
          </button>
        </div>

        <label className="field-label" htmlFor="review-note">审核备注</label>
        <textarea
          id="review-note"
          value={notes}
          onChange={(event) => onNotesChange(event.target.value)}
          placeholder="例如：已与仓库确认替代 SKU，可继续履约。"
          rows={4}
        />
        {error && <div className="error-box">{error}</div>}

        <div className="review-inline-actions">
          <button className="secondary-button" type="button" disabled={loading} onClick={onClose}>
            稍后处理
          </button>
          {loading && <span className="review-submit-status">提交中...</span>}
        </div>
      </section>
    </div>
  );
}

export function OperationsDashboard() {
  const [messages, setMessages] = useState<ChatMessage[]>(loadMessages);
  const [input, setInput] = useState(analysisTemplates[0]);
  const [review, setReview] = useState<ReviewState | null>(null);
  const [reviewNotes, setReviewNotes] = useState("");
  const [reviewError, setReviewError] = useState<string | null>(null);
  const typingTimerRef = useRef<number | null>(null);
  const ordersQuery = useQuery({ queryKey: ["enterprise-orders", 50], queryFn: () => listEnterpriseOrders(50) });
  const knownOrderIds = useMemo(
    () => new Set((ordersQuery.data?.data.orders || []).map((order) => order.order_id)),
    [ordersQuery.data],
  );

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-40)));
  }, [messages]);

  useEffect(() => {
    return () => {
      if (typingTimerRef.current) window.clearInterval(typingTimerRef.current);
    };
  }, []);

  const runMutation = useMutation({ mutationFn: runHybrid });
  const resumeMutation = useMutation({ mutationFn: resumeHybrid });

  function appendMessage(message: Omit<ChatMessage, "id" | "createdAt">) {
    const next: ChatMessage = { ...message, id: makeId(message.role), createdAt: new Date().toISOString() };
    setMessages((current) => [...current, next]);
    return next.id;
  }

  function typeAssistantMessage(messageId: string, text: string) {
    if (typingTimerRef.current) window.clearInterval(typingTimerRef.current);
    const chars = Array.from(text);
    let index = 0;
    setMessages((current) =>
      current.map((message) => (message.id === messageId ? { ...message, content: "", status: "typing" } : message)),
    );
    typingTimerRef.current = window.setInterval(() => {
      index += 1;
      setMessages((current) =>
        current.map((message) =>
          message.id === messageId
            ? { ...message, content: chars.slice(0, index).join(""), status: index >= chars.length ? "done" : "typing" }
            : message,
        ),
      );
      if (index >= chars.length && typingTimerRef.current) {
        window.clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
      }
    }, 16);
  }

  function attachEvidence(messageId: string, evidence: MessageEvidence) {
    setMessages((current) =>
      current.map((message) => (message.id === messageId ? { ...message, evidence } : message)),
    );
  }

  async function handleSend() {
    const content = input.trim();
    if (!content || runMutation.isPending) return;
    const orderId = extractOrderId(content);
    const clientInputAt = new Date().toISOString();
    setReview(null);
    setReviewError(null);
    appendMessage({ role: "user", content, status: "done" });
    const assistantId = appendMessage({
      role: "assistant",
      content: orderId
        ? "正在读取订单、库存和规则，准备执行 Hybrid Routing..."
        : "正在检索知识库文档，准备执行 Hybrid Routing...",
      status: "typing",
    });
    setInput("");

    try {
      const requestStartedAt = new Date().toISOString();
      const response = await runMutation.mutateAsync({
        orderId: orderId || undefined,
        question: buildProfessionalQuestion(content),
        clientInputAt,
      });
      const requestEndedAt = new Date().toISOString();
      const result = response.data;
      const answer = finalAnswer(result);
      attachEvidence(assistantId, buildMessageEvidence(result, { clientInputAt, requestStartedAt, requestEndedAt }));
      typeAssistantMessage(assistantId, answer);

      if (result.status === "interrupted" && result.interrupt) {
        const interrupt = result.interrupt;
        const threadId = result.interrupt?.thread_id || result.thread_id;
        setReview({
          orderId: result.order_id || orderId,
          threadId,
          riskLevel: interrupt.risk_level,
          reasons: interrupt.risk_signals?.length ? interrupt.risk_signals : [interrupt.prompt],
          suggestion: answer,
        });
      }
    } catch (error) {
      const requestEndedAt = new Date().toISOString();
      const message = error instanceof Error ? error.message : "智能分析失败，请检查后端服务、模型配置或网络连接。";
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId
            ? {
              ...item,
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
            : item,
        ),
      );
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  }

  async function submitReview(decision: "approved" | "rejected") {
    if (!review) return;
    const effectiveNotes = reviewNotes.trim()
      || (decision === "approved" ? "已确认订单、库存和履约风险，可继续履约。" : "风险未解除，暂停自动履约，等待人工进一步处理。");
    setReviewError(null);
    try {
      let result: HybridRunResult | null = null;
      if (review.threadId) {
        const response = await resumeMutation.mutateAsync({
          threadId: review.threadId,
          decision,
          notes: effectiveNotes,
        });
        result = response.data;
      }
      const summary =
        decision === "approved"
          ? `人工审核已通过：${review.orderId} 可继续履约。备注：${effectiveNotes}`
          : `人工审核已暂停：${review.orderId} 暂不释放履约。备注：${effectiveNotes}`;
      const assistantId = appendMessage({ role: "assistant", content: "", status: "typing" });
      if (result) attachEvidence(assistantId, buildMessageEvidence(result));
      typeAssistantMessage(assistantId, result?.final_answer ? `${summary}\n\n${result.final_answer}` : summary);
      setReview(null);
      setReviewNotes("");
    } catch (error) {
      setReviewError(error instanceof Error ? error.message : "审核提交失败，请稍后重试。");
    }
  }

  return (
    <div className="page page--agent-console">
      <div className="agent-console-grid agent-console-grid--single">
        <ChatWindow
          messages={messages}
          input={input}
          loading={runMutation.isPending}
          onInputChange={setInput}
          onSend={() => void handleSend()}
          onKeyDown={handleKeyDown}
        />
      </div>

      {review && (
        <ReviewModal
          review={review}
          notes={reviewNotes}
          loading={resumeMutation.isPending}
          error={reviewError}
          onNotesChange={setReviewNotes}
          onClose={() => setReview(null)}
          onSubmit={(decision) => void submitReview(decision)}
        />
      )}
    </div>
  );
}
