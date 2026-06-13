import { useQuery } from "@tanstack/react-query";
import { Activity, BotMessageSquare, Database } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

import { getEnterpriseStats, getHealth } from "../lib/api";
import { StatusPill } from "./StatusPill";

export function Shell() {
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: 1 });
  const enterpriseStats = useQuery({ queryKey: ["enterprise-stats"], queryFn: getEnterpriseStats, retry: 1 });
  const apiOk = health.data?.data.status === "ok";
  const orderCount = enterpriseStats.data?.data.order_count ?? 0;
  const inventoryCount = enterpriseStats.data?.data.inventory_record_count ?? 0;
  const dataReady = orderCount > 0 || inventoryCount > 0;

  return (
    <div className="app-shell app-shell--focused">
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
            <BotMessageSquare size={18} />
            智能履约
          </NavLink>
          <NavLink to="/data" className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
            <Database size={18} />
            数据接入
          </NavLink>
          <NavLink to="/traces" className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
            <Activity size={18} />
            Trace Center
          </NavLink>
        </nav>
      </aside>

      <div className="app-frame">
        <header className="top-header top-header--demo">
          <div>
            <strong>Multiship 智能履约中台</strong>
            <span>Demo Retail Group · Sandbox · Hybrid Routing</span>
          </div>
          <div className="top-header__meta">
            <StatusPill tone={dataReady ? "ok" : "warn"}>
              {dataReady ? "数据已接入" : "等待数据"}
            </StatusPill>
            <StatusPill tone={apiOk ? "ok" : health.isError ? "danger" : "warn"}>
              {apiOk ? "API 正常" : health.isError ? "API 异常" : "API 检查中"}
            </StatusPill>
          </div>
        </header>
        <main className="main-surface main-surface--focused">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
