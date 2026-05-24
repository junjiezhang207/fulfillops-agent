import { useQuery } from "@tanstack/react-query";
import { Activity, Brain, Database, Eye, GitBranch, Radio, Route, Server } from "lucide-react";

import { StatusPill } from "../components/StatusPill";
import { getEnterpriseStats, getHealth, getHybridStats } from "../lib/api";
import { recentRuns } from "../lib/mockData";

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
  const apiOk = health.data?.data.status === "ok";

  return (
    <div className="page page--demo">
      <header className="page-header demo-hero demo-hero--focused">
        <div>
          <p className="eyebrow">系统运行状态</p>
          <h1>系统状态</h1>
          <p>集中展示 API、Redis、Milvus、MySQL、Langfuse 和 Hybrid Routing 的运行状态，帮助运营与技术人员快速判断系统是否可用。</p>
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
          <ServiceCard icon={Brain} name="Milvus" value="向量检索" detail="RAG 与长期记忆语义搜索" tone="info" />
          <ServiceCard icon={Database} name="MySQL / 本地数据" value={enterpriseStats.data?.data.order_count ?? 0} detail="企业结构化数据和长期记忆元数据" tone="info" />
          <ServiceCard icon={Eye} name="Langfuse" value="Trace" detail="Agent 调用链路和质量观测" />
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
            <article><strong>Milvus</strong><span>用于 RAG 和长期记忆的向量检索。</span></article>
            <article><strong>MySQL</strong><span>保存企业结构化数据与长期记忆元数据。</span></article>
            <article><strong>Langfuse</strong><span>记录 Agent Trace、模型调用和执行质量。</span></article>
            <article><strong>Hybrid Routing</strong><span>按问题复杂度选择 Workflow / Agent / RAG / Multi-Agent。</span></article>
          </div>
        </section>
      </div>

      <section className="workspace">
        <div className="section-title">
          <div>
            <h2>最近运行</h2>
            <p>用几条订单分析记录说明路由路径、耗时和是否进入人工审查。</p>
          </div>
          <Route size={20} />
        </div>
        <div className="run-card-list">
          {recentRuns.map((run) => (
            <article key={run.orderId}>
              <div>
                <strong>{run.orderId}</strong>
                <span>{run.route} · {run.latency} · {run.result}</span>
              </div>
              <StatusPill tone={run.review === "是" ? "warn" : "ok"}>HITL {run.review}</StatusPill>
            </article>
          ))}
        </div>
      </section>

      <section className="demo-bottom-notes">
        <article><Activity size={18} /><span>API、数据、路由和观测形成完整工程链路。</span></article>
        <article><Brain size={18} /><span>Milvus + MySQL 表达长期记忆，不使用 PostgreSQL。</span></article>
        <article><Eye size={18} /><span>Langfuse 用于查看每次 Agent 调用和 Trace。</span></article>
      </section>
    </div>
  );
}
