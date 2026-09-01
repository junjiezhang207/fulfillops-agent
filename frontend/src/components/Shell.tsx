import { useQuery } from "@tanstack/react-query";
import { ClipboardList, Database } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

import { getEnterpriseStats, getHealth } from "../lib/api";

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
          <div className="brand-mark">FA</div>
          <div>
            <strong>fulfillops-agent</strong>
            <span>电商履约运营智能协同 Agent</span>
          </div>
        </div>

        <nav className="nav-list">
          <NavLink to="/" end className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
            <ClipboardList size={18} />
            异常处置
          </NavLink>
          <NavLink to="/data" className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
            <Database size={18} />
            数据接入
          </NavLink>
        </nav>

        <div className="sidebar-status">
          <span>Environment</span>
          <strong>Sandbox</strong>
          <div>
            <i className={apiOk ? "ok" : health.isError ? "danger" : "warn"} />
            {apiOk ? "API 正常" : health.isError ? "API 异常" : "API 检查中"}
          </div>
        </div>
      </aside>

      <div className="app-frame">
        <header className="top-header top-header--demo">
          <div>
            <strong>异常订单处置</strong>
            <span>AI-assisted Fulfillment Operations</span>
          </div>
          <div className="top-header__meta">
            <span className={`header-dot-status header-dot-status--${dataReady ? "ok" : "warn"}`}>
              <i />
              {dataReady ? "数据已接入" : "等待数据"}
            </span>
            <span className={`header-dot-status header-dot-status--${apiOk ? "ok" : health.isError ? "danger" : "warn"}`}>
              <i />
              {apiOk ? "API 正常" : health.isError ? "API 异常" : "API 检查中"}
            </span>
          </div>
        </header>
        <main className="main-surface main-surface--focused">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
