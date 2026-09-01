import { useQuery } from "@tanstack/react-query";
import { Activity, Brain, Database, Eye, GitBranch, Radio, Route, Server } from "lucide-react";

import { StatusPill } from "../components/StatusPill";
import { getEnterpriseStats, getHealth, getHybridStats, listBusinessTraces } from "../lib/api";

type ServiceTone = "ok" | "warn" | "danger" | "info";

function ServiceCard({
  icon: Icon,
  name,
  value,
  detail,
  tone = "ok",
}: {
  icon: typeof Server;
  name: string;
  value: string | number;
  detail: string;
  tone?: ServiceTone;
}) {
  return (
    <article className="service-simple-card service-simple-card--focused">
      <Icon size={22} />
      <div>
        <span>{name}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
      <StatusPill tone={tone}>{tone === "ok" ? "正常" : tone === "warn" ? "观察" : tone === "danger" ? "异常" : "可用"}</StatusPill>
    </article>
  );
}

export function SystemPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth });
  const enterpriseStats = useQuery({ queryKey: ["enterprise-stats"], queryFn: getEnterpriseStats });
  const hybridStats = useQuery({ queryKey: ["hybrid-stats"], queryFn: getHybridStats });
  const recentTraces = useQuery({ queryKey: ["recent-business-traces"], queryFn: () => listBusinessTraces({ limit: 5 }) });
  const apiOk = health.data?.data.status === "ok";

  return (
    <div className="page page--demo">
      <header className="page-header demo-hero demo-hero--focused">
        <div>
          <p className="eyebrow">系统运行状态</p>
          <h1>系统状态</h1>
          <p>集中展示 API、Redis、ChromaDB、MySQL、Trace Center 和 Hybrid Routing 的运行状态，帮助运营与技术人员快速判断系统是否可用。</p>
        </div>
        <StatusPill tone={apiOk ? "ok" : health.isError ? "danger" : "warn"}>
          {apiOk ? "API 正常" : health.isError ? "API 异常" : "检查中"}
        </StatusPill>
      </header>

      <section className="workspace">
        <div className="section-title">
          <div>
            <h2>核心服务</h2>
            <p>展示履约决策链路依赖的关键基础设施。</p>
          </div>
        </div>
        <div className="service-simple-grid service-simple-grid--focused">
          <ServiceCard icon={Server} name="API 服务" value={apiOk ? "在线" : "检查中"} detail={health.data?.data.version || "FastAPI"} tone={apiOk ? "ok" : "warn"} />
          <ServiceCard icon={Radio} name="Redis" value="启用" detail="短期记忆 / 工具缓存 / 速率限制" />
          <ServiceCard icon={Brain} name="ChromaDB" value="向量检索" detail="当前 RAG 与长期记忆语义索引，Milvus 暂不启用" tone="info" />
          <ServiceCard icon={Database} name="MySQL" value={enterpriseStats.data?.data.order_count ?? 0} detail="企业结构化数据、HITL、Trace 和长期记忆元数据" tone="info" />
          <ServiceCard icon={Eye} name="Trace Center" value="业务链路" detail="Hybrid / RAG / Tool / HITL 观测" />
        </div>
      </section>

      <div className="system-simple-grid system-simple-grid--focused">
        <section className="workspace">
          <div className="section-title">
            <div>
              <h2>智能路由</h2>
              <p>Hybrid 根据任务复杂度选择 Workflow / Agent / RAG / Multi-Agent。</p>
            </div>
          </div>
          <div className="route-simple-list">
            <article><GitBranch size={18} /><div><strong>Workflow</strong><span>确定性库存、SLA、风险规则校验。</span></div></article>
            <article><GitBranch size={18} /><div><strong>Agent</strong><span>开放式履约分析、工具调用和运营建议。</span></div></article>
            <article><GitBranch size={18} /><div><strong>RAG</strong><span>检索规则、SOP、替代 SKU 和业务知识。</span></div></article>
            <article><GitBranch size={18} /><div><strong>Multi-Agent</strong><span>库存、履约、风险多个专家协同判断。</span></div></article>
          </div>
          <div className="facts-list facts-list--flat">
            <div><dt>当前策略</dt><dd>Hybrid</dd></div>
            <div><dt>简单阈值</dt><dd>{hybridStats.data?.data.simple_threshold ?? "--"}</dd></div>
            <div><dt>复杂阈值</dt><dd>{hybridStats.data?.data.complex_threshold ?? "--"}</dd></div>
          </div>
        </section>

        <section className="workspace">
          <div className="section-title">
            <div>
              <h2>能力说明</h2>
              <p>说明各基础设施在智能履约链路中的职责。</p>
            </div>
          </div>
          <div className="tech-note-list">
            <article><strong>Redis</strong><span>用于短期记忆、工具缓存和接口速率限制。</span></article>
            <article><strong>ChromaDB</strong><span>当前用于 RAG 和长期记忆的向量检索；Milvus 仅作为可选扩展。</span></article>
            <article><strong>MySQL</strong><span>保存企业结构化数据与长期记忆元数据。</span></article>
            <article><strong>Trace Center</strong><span>记录业务决策步骤、RAG 证据、工具调用和审计事件。</span></article>
            <article><strong>Hybrid Routing</strong><span>按问题复杂度选择 Workflow / Agent / RAG / Multi-Agent。</span></article>
          </div>
        </section>
      </div>

      <section className="workspace">
        <div className="section-title">
          <div>
            <h2>最近运行</h2>
            <p>来自 Trace Center 的真实业务链路记录。</p>
          </div>
          <Route size={20} />
        </div>
        <div className="run-card-list">
          {recentTraces.isLoading && <div className="empty-mini">正在读取最近运行记录...</div>}
          {recentTraces.isError && <div className="error-box">最近运行加载失败，请检查 Trace Center 和 MySQL。</div>}
          {(recentTraces.data?.data.traces || []).map((run) => (
            <article key={run.trace_id}>
              <div>
                <strong>{run.order_id || run.trace_id}</strong>
                <span>{run.route || "Hybrid"} · {run.duration_ms ? `${Math.round(run.duration_ms)}ms` : "--"} · {run.status || "--"}</span>
              </div>
              <StatusPill tone={run.status === "pending_human" || run.status === "interrupted" ? "warn" : run.status === "error" ? "danger" : "ok"}>
                {run.status === "pending_human" || run.status === "interrupted" ? "HITL 是" : "HITL 否"}
              </StatusPill>
            </article>
          ))}
          {!recentTraces.isLoading && (recentTraces.data?.data.traces || []).length === 0 && (
            <div className="empty-mini">暂无真实运行记录，完成一次智能履约分析后会显示在这里。</div>
          )}
        </div>
      </section>

      <section className="demo-bottom-notes">
        <article><Activity size={18} /><span>API、数据、路由和观测形成完整工程链路。</span></article>
        <article><Brain size={18} /><span>ChromaDB + MySQL 表达长期记忆，避免本地开发启用高内存 Milvus。</span></article>
        <article><Eye size={18} /><span>Trace Center 用于查看每次履约决策链路。</span></article>
      </section>
    </div>
  );
}
