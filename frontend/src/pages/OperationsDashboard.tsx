import { useMutation } from "@tanstack/react-query";
import { Bot, CheckCircle2, ChevronDown, ChevronUp, Clock3, Play, ShieldAlert, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { StatusPill } from "../components/StatusPill";
import { runHybrid, type HybridRunResult, type InterruptEvent } from "../lib/api";
import {
  analysisTemplates,
  decisionModes,
  demoFlowSteps,
  mockOrders,
  riskToneFromLevel,
  type MockOrder,
  type RouteMode,
} from "../lib/mockData";
import { useReviewStore } from "../store/reviewStore";

function finalAnswer(result: HybridRunResult | null, order: MockOrder) {
  if (typeof result?.final_answer === "string" && result.final_answer.trim()) return result.final_answer;
  if (result?.status === "interrupted") return "系统已暂停自动履约，并将该订单送入人工审查队列。";
  if (result?.status === "completed") return order.suggestion;
  return "";
}

function formatDuration(ms?: number) {
  if (!ms) return "--";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function formatPercent(value?: number) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "--";
}

function useTypewriter(text: string, speedMs = 18) {
  const [displayedText, setDisplayedText] = useState("");
  const requestRef = useRef(0);

  useEffect(() => {
    requestRef.current += 1;
    const requestId = requestRef.current;
    const chars = Array.from(text);

    if (!text) return undefined;

    let index = 0;
    const resetTimer = window.setTimeout(() => setDisplayedText(""), 0);
    const timer = window.setInterval(() => {
      if (requestRef.current !== requestId) {
        window.clearInterval(timer);
        return;
      }
      index += 1;
      setDisplayedText(chars.slice(0, index).join(""));
      if (index >= chars.length) window.clearInterval(timer);
    }, speedMs);

    return () => {
      window.clearTimeout(resetTimer);
      window.clearInterval(timer);
    };
  }, [text, speedMs]);

  return displayedText;
}

function buildInterrupt(order: MockOrder): InterruptEvent {
  return {
    type: "human_review_required",
    node: "risk_gate",
    prompt: `${order.orderId} 触发 ${order.riskReasons.join("、")}，请人工确认履约动作。`,
    context: {
      platform: order.platform,
      customer_level: order.customerLevel,
      fulfillment_path: order.fulfillmentPath,
      sla_deadline: order.slaDeadline,
      warehouse: order.warehouse,
    },
    options: ["approved", "rejected"],
    thread_id: `manual-${order.orderId}`,
    risk_level: order.riskLevel,
    risk_signals: order.riskReasons,
    timeout_seconds: 1800,
  };
}

function DemoMetric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <article className="demo-metric demo-metric--clean">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </article>
  );
}

function FlowStrip() {
  return (
    <section className="demo-flow-strip" aria-label="履约处理链路">
      {demoFlowSteps.map((step, index) => (
        <article key={step.title}>
          <span>{index + 1}</span>
          <div>
            <strong>{step.title}</strong>
            <small>{step.detail}</small>
          </div>
        </article>
      ))}
    </section>
  );
}

function OrderScenarioCard({
  order,
  selected,
  onSelect,
  onAnalyze,
}: {
  order: MockOrder;
  selected: boolean;
  onSelect: () => void;
  onAnalyze: () => void;
}) {
  return (
    <article className={selected ? "scenario-card selected" : "scenario-card"}>
      <button type="button" onClick={onSelect}>
        <div className="scenario-card__top">
          <span>{order.scenario}</span>
          <StatusPill tone={riskToneFromLevel(order.riskLevel)}>{order.riskLevel}</StatusPill>
        </div>
        <strong>{order.orderId}</strong>
        <small>{order.platform} · {order.region} · {order.inventoryState}</small>
        <p>{order.suggestion}</p>
      </button>
      <div className="scenario-card__bottom">
        <span>{order.fulfillmentPath}</span>
        <button className="ghost-button" type="button" onClick={onAnalyze}>分析</button>
      </div>
    </article>
  );
}

function StepList({ active }: { active: boolean }) {
  const steps = ["读取订单", "检查库存", "匹配履约规则", "选择智能路由", "输出建议"];
  return (
    <div className="demo-steps demo-steps--strong">
      {steps.map((step, index) => (
        <div className={active ? "is-running" : ""} key={step}>
          <span>{index + 1}</span>
          <strong>{step}</strong>
        </div>
      ))}
    </div>
  );
}

function ResultPanel({
  result,
  pending,
  selectedOrder,
}: {
  result: HybridRunResult | null;
  pending: boolean;
  selectedOrder: MockOrder;
}) {
  const text = finalAnswer(result, selectedOrder);
  const typedText = useTypewriter(text);
  const typing = Boolean(result && text && typedText.length < text.length);

  return (
    <section className="workspace demo-result demo-result--spotlight">
      <div className="section-title">
        <div>
          <h2>AI 履约结论</h2>
          <p>展示本次分析的路由路径、风险判断、人工审查状态和最终处理建议。</p>
        </div>
        {result && <StatusPill tone={result.status === "interrupted" ? "warn" : "ok"}>{result.status}</StatusPill>}
      </div>

      {pending && (
        <div className="progress-panel progress-panel--active">
          <span>正在执行 Hybrid Routing</span>
          <strong>后端正在结合订单、库存、规则和模型输出处理建议</strong>
          <div><i /></div>
        </div>
      )}

      {result ? (
        <>
          <div className="demo-result-grid">
            <div><span>使用路径</span><strong>{result.path_used || "Hybrid"}</strong></div>
            <div><span>置信度</span><strong>{formatPercent(result.confidence ?? selectedOrder.confidence)}</strong></div>
            <div><span>耗时</span><strong>{formatDuration(result.execution_time_ms)}</strong></div>
            <div><span>进入 HITL</span><strong>{result.status === "interrupted" || selectedOrder.needsReview ? "是" : "否"}</strong></div>
          </div>
          <div className="answer-box answer-box--hero">
            {typedText}
            {typing && <span className="typewriter-cursor" aria-hidden="true" />}
          </div>
          <div className="demo-risk-summary">
            <ShieldAlert size={18} />
            <span>{selectedOrder.riskReasons.join(" / ")}</span>
          </div>
        </>
      ) : (
        <div className="empty-panel empty-panel--result">
          <Bot size={30} />
          <strong>选择订单后开始分析</strong>
          <span>选择订单并发起分析后，系统会返回履约结论、风险原因和建议动作。</span>
        </div>
      )}
    </section>
  );
}

function OrderDetail({ order }: { order: MockOrder }) {
  return (
    <section className="workspace demo-detail">
      <div className="section-title">
        <div>
          <h2>订单证据</h2>
          <p>{order.orderId} · SLA {order.slaDeadline} · 推荐 {order.fulfillmentPath}</p>
        </div>
        <StatusPill tone={riskToneFromLevel(order.riskLevel)}>{order.riskLevel}</StatusPill>
      </div>

      <div className="demo-detail-grid">
        <div>
          <h3>SKU 明细</h3>
          <div className="table-scroll">
            <table className="data-table data-table--compact">
              <thead>
                <tr>
                  <th>SKU</th>
                  <th>商品</th>
                  <th>订购</th>
                  <th>可用</th>
                  <th>缺口</th>
                  <th>推荐仓</th>
                </tr>
              </thead>
              <tbody>
                {order.items.map((item) => (
                  <tr key={item.skuId}>
                    <td>{item.skuId}</td>
                    <td>{item.name}</td>
                    <td>{item.quantity}</td>
                    <td>{item.availableStock}</td>
                    <td>{item.shortage}</td>
                    <td>{item.recommendedWarehouse}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h3>仓库库存</h3>
          <div className="compact-warehouse-list">
            {order.warehouses.map((warehouse) => (
              <article key={warehouse.name}>
                <strong>{warehouse.name}</strong>
                <span>可用 {warehouse.available} · 锁定 {warehouse.locked}</span>
                <StatusPill tone={warehouse.shippable ? "ok" : "warn"}>{warehouse.shippable ? "可发货" : "需协调"}</StatusPill>
                <small>出库 {warehouse.outboundEta} · 时效 {warehouse.transitEta}</small>
              </article>
            ))}
          </div>
        </div>
      </div>

      <div className="demo-timeline">
        {order.timeline.map((item) => (
          <article key={item.step}>
            <span className={`dot dot--${item.status}`} />
            <div>
              <strong>{item.step}</strong>
              <p>{item.note}</p>
              <small>{item.strategy} · {item.latency}</small>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export function OperationsDashboard() {
  const [selectedOrderId, setSelectedOrderId] = useState(mockOrders[0].orderId);
  const [question, setQuestion] = useState(analysisTemplates[0]);
  const [decisionMode, setDecisionMode] = useState<RouteMode>("Hybrid");
  const [detailOpen, setDetailOpen] = useState(false);
  const [lastResult, setLastResult] = useState<HybridRunResult | null>(null);
  const addPendingReview = useReviewStore((state) => state.addPendingReview);

  const selectedOrder = useMemo(
    () => mockOrders.find((order) => order.orderId === selectedOrderId) || mockOrders[0],
    [selectedOrderId],
  );

  const runMutation = useMutation({
    mutationFn: runHybrid,
    onSuccess: (response) => {
      const result = response.data;
      setLastResult(result);
      if (result.status === "interrupted" && result.interrupt) {
        const threadId = result.interrupt.thread_id || result.thread_id || `review-${Date.now()}`;
        addPendingReview({
          threadId,
          orderId: result.order_id || selectedOrder.orderId,
          question,
          interrupt: { ...result.interrupt, thread_id: threadId },
          sourceResult: result,
          createdAt: new Date().toISOString(),
        });
      }
    },
  });

  function analyze(order = selectedOrder) {
    setSelectedOrderId(order.orderId);
    runMutation.mutate({ orderId: order.orderId, question });
  }

  function addDemoReview(order: MockOrder) {
    const interrupt = buildInterrupt(order);
    addPendingReview({
      threadId: interrupt.thread_id,
      orderId: order.orderId,
      question,
      interrupt,
      sourceResult: {
        order_id: order.orderId,
        question,
        final_answer: order.suggestion,
        path_used: "Hybrid",
        intent_level: "manual_review",
        confidence: order.confidence,
        tools_called: ["risk_gate", "inventory_check"],
        execution_time_ms: 0,
        thread_id: interrupt.thread_id,
        conversation_turns: 1,
        status: "interrupted",
        interrupt,
      },
      createdAt: new Date().toISOString(),
    });
  }

  return (
    <div className="page page--demo">
      <header className="page-header demo-hero demo-hero--focused">
        <div>
          <p className="eyebrow">履约决策工作台</p>
          <h1>智能履约控制台</h1>
          <p>选择待处理订单，系统结合 OMS 订单、WMS 库存、SLA 规则和智能路由输出履约建议；高风险订单会自动进入人工审查。</p>
        </div>
        <div className="demo-metric-row">
          <DemoMetric label="今日订单" value="286" hint="OMS / ERP / 平台订单" />
          <DemoMetric label="风险订单" value="18" hint="缺货、SLA、替代 SKU" />
          <DemoMetric label="自动决策率" value="74.8%" hint="无需人工介入" />
        </div>
      </header>

      <FlowStrip />

      <div className="demo-stage-grid">
        <section className="workspace">
          <div className="section-title">
            <div>
              <h2>待处理订单</h2>
              <p>订单池覆盖正常履约、缺货调拨、高风险复核和企业导入订单等常见运营场景。</p>
            </div>
          </div>
          <div className="scenario-grid">
            {mockOrders.map((order) => (
              <OrderScenarioCard
                key={order.orderId}
                order={order}
                selected={order.orderId === selectedOrderId}
                onSelect={() => setSelectedOrderId(order.orderId)}
                onAnalyze={() => analyze(order)}
              />
            ))}
          </div>
        </section>

        <aside className="workspace demo-command demo-command--stage">
          <div className="section-title">
            <div>
              <h2>智能分析</h2>
              <p>调用 `/hybrid/run` 发起智能分析，由后端根据任务复杂度选择处理路径。</p>
            </div>
            <Sparkles size={20} />
          </div>

          <div className="selected-order-card">
            <span>当前订单</span>
            <strong>{selectedOrder.orderId}</strong>
            <small>{selectedOrder.scenario} · {selectedOrder.platform} · {selectedOrder.region}</small>
          </div>

          <label className="field-label" htmlFor="question">分析目标</label>
          <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} rows={3} />
          <div className="template-row template-row--compact">
            {analysisTemplates.map((template) => (
              <button className={question === template ? "selected" : ""} key={template} type="button" onClick={() => setQuestion(template)}>
                {template}
              </button>
            ))}
          </div>

          <label className="field-label" htmlFor="mode">路由模式</label>
          <select id="mode" value={decisionMode} onChange={(event) => setDecisionMode(event.target.value as RouteMode)}>
            {decisionModes.map((mode) => <option key={mode}>{mode}</option>)}
          </select>

          <StepList active={runMutation.isPending} />

          {runMutation.error && (
            <div className="error-box">
              {runMutation.error instanceof Error ? runMutation.error.message : "智能分析失败，请检查后端服务或模型配置。"}
            </div>
          )}

          <button className="primary-button run-button" type="button" disabled={runMutation.isPending} onClick={() => analyze()}>
            <Play size={18} />
            {runMutation.isPending ? "分析中..." : "开始分析"}
          </button>

          {selectedOrder.needsReview && (
            <button className="secondary-button full-width" type="button" onClick={() => addDemoReview(selectedOrder)}>
              <ShieldAlert size={17} />
              送入人工审查
            </button>
          )}
        </aside>
      </div>

      <ResultPanel result={lastResult} pending={runMutation.isPending} selectedOrder={selectedOrder} />

      <div className="detail-toggle-row">
        <button className="secondary-button" type="button" onClick={() => setDetailOpen((value) => !value)}>
          {detailOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          {detailOpen ? "收起订单证据" : "查看订单证据"}
        </button>
        <span><CheckCircle2 size={16} /> 证据区默认收起，便于运营人员优先查看结论和建议动作。</span>
      </div>

      {detailOpen && <OrderDetail order={selectedOrder} />}

      <section className="demo-bottom-notes">
        <article><Clock3 size={18} /><span>Workflow 处理确定性库存与 SLA 校验</span></article>
        <article><Bot size={18} /><span>Agent 负责开放式运营建议生成</span></article>
        <article><ShieldAlert size={18} /><span>高风险订单通过 HITL 暂停并恢复流程</span></article>
      </section>
    </div>
  );
}
