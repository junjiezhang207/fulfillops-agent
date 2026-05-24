import { useQuery } from "@tanstack/react-query";
import { ClipboardCheck, Database, Gauge, LayoutDashboard } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

import { getEnterpriseStats, getHealth } from "../lib/api";
import { useReviewStore } from "../store/reviewStore";
import { StatusPill } from "./StatusPill";

export function Shell() {
  const pendingCount = useReviewStore((state) => state.pendingReviews.length);
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: 1 });
  const enterpriseStats = useQuery({ queryKey: ["enterprise-stats"], queryFn: getEnterpriseStats, retry: 1 });
  const apiOk = health.data?.data.status === "ok";
  const dataReady = Boolean((enterpriseStats.data?.data.order_count || 0) > 0);

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="主导航">
        <div className="brand">
          <div className="brand-mark">MS</div>
          <div>
            <strong>Multiship</strong>
            <span>智能履约中台</span>
          </div>
        </div>

        <nav className="nav-list">
          <NavLink to="/" end className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
            <LayoutDashboard size={18} />
            智能履约
          </NavLink>
          <NavLink to="/reviews" className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
            <ClipboardCheck size={18} />
            人工审查
            {pendingCount > 0 && <span className="nav-count">{pendingCount}</span>}
          </NavLink>
          <NavLink to="/data" className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
            <Database size={18} />
            数据接入
          </NavLink>
          <NavLink to="/system" className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
            <Gauge size={18} />
            系统状态
          </NavLink>
        </nav>
      </aside>

      <div className="app-frame">
        <header className="top-header top-header--demo">
          <div>
            <strong>Multiship 智能履约中台</strong>
            <span>上传数据 → Hybrid Routing → HITL 审查 → Langfuse 观测</span>
          </div>
          <div className="top-header__meta">
            <span>Retail Operations · Production</span>
            <StatusPill tone={dataReady ? "ok" : "warn"}>{dataReady ? "数据已接入" : "等待数据"}</StatusPill>
            <StatusPill tone={apiOk ? "ok" : health.isError ? "danger" : "warn"}>
              {apiOk ? "API 正常" : health.isError ? "API 异常" : "API 检查中"}
            </StatusPill>
          </div>
        </header>
        <main className="main-surface">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
