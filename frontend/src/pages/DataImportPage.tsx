import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Database, FileCheck2, UploadCloud } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import { StatusPill } from "../components/StatusPill";
import {
  getEnterpriseStats,
  importEnterpriseInventory,
  importEnterpriseOrders,
  type EnterpriseImportResult,
  type InventoryRecord,
  type OrderRecord,
  upsertEnterpriseSource,
} from "../lib/api";
import { importHistory } from "../lib/mockData";

type ImportMode = "orders" | "inventory";
type ParsedPayload =
  | { mode: "orders"; records: OrderRecord[] }
  | { mode: "inventory"; records: InventoryRecord[] };

type ParseState = {
  fileName: string;
  fileSize: number;
  headers: string[];
  payload: ParsedPayload;
  errors: string[];
  warnings: string[];
};

const orderColumns = ["order_id", "platform", "order_time", "order_status", "region", "priority", "sku_id", "product_name", "quantity", "unit_price"];
const inventoryColumns = ["warehouse_id", "warehouse_name", "region", "sku_id", "available_stock", "locked_stock", "updated_at"];

function normalizeSourceId(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, "-") || "manual";
}

function parseCsv(text: string) {
  const rows = text.trim().split(/\r?\n/).filter(Boolean);
  if (rows.length === 0) return [];
  const headers = rows[0].split(",").map((item) => item.replace(/^\uFEFF/, "").trim());
  return rows.slice(1).map((row) => {
    const cells = row.split(",").map((item) => item.trim());
    return Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""]));
  });
}

function numberValue(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function readRows(text: string, fileName: string, mode: ImportMode) {
  if (fileName.toLowerCase().endsWith(".json")) {
    const raw = JSON.parse(text) as unknown;
    if (Array.isArray(raw)) return raw as Record<string, unknown>[];
    if (mode === "orders") return ((raw as { orders?: unknown[] }).orders || []) as Record<string, unknown>[];
    return ((raw as { records?: unknown[] }).records || []) as Record<string, unknown>[];
  }
  return parseCsv(text);
}

function parseOrders(rows: Record<string, unknown>[]) {
  const orderMap = new Map<string, OrderRecord>();
  rows.forEach((row) => {
    const orderId = String(row.order_id || "").trim();
    if (!orderId) return;
    const item = {
      sku_id: String(row.sku_id || ""),
      product_name: String(row.product_name || ""),
      quantity: numberValue(row.quantity),
      unit_price: numberValue(row.unit_price),
    };
    const existing = orderMap.get(orderId);
    if (existing) existing.items.push(item);
    else {
      orderMap.set(orderId, {
        order_id: orderId,
        platform: String(row.platform || ""),
        order_time: String(row.order_time || ""),
        order_status: String(row.order_status || ""),
        region: String(row.region || ""),
        priority: String(row.priority || ""),
        items: [item],
      });
    }
  });
  return [...orderMap.values()];
}

function parseInventory(rows: Record<string, unknown>[]): InventoryRecord[] {
  return rows.map((row) => ({
    warehouse_id: String(row.warehouse_id || ""),
    warehouse_name: String(row.warehouse_name || ""),
    region: String(row.region || ""),
    sku_id: String(row.sku_id || ""),
    available_stock: numberValue(row.available_stock),
    locked_stock: numberValue(row.locked_stock),
    updated_at: String(row.updated_at || ""),
  }));
}

function parseFile(fileName: string, fileSize: number, text: string, mode: ImportMode): ParseState {
  const required = mode === "orders" ? orderColumns : inventoryColumns;
  try {
    const rows = readRows(text, fileName, mode);
    const headers = Object.keys(rows[0] || {});
    const missing = required.filter((field) => !headers.includes(field));
    const records = mode === "orders" ? parseOrders(rows) : parseInventory(rows);
    return {
      fileName,
      fileSize,
      headers,
      payload: mode === "orders"
        ? { mode, records: records as OrderRecord[] }
        : { mode, records: records as InventoryRecord[] },
      errors: records.length === 0 ? ["文件中没有可导入数据。"] : [],
      warnings: missing.map((field) => `缺少字段：${field}`),
    };
  } catch (error) {
    return {
      fileName,
      fileSize,
      headers: [],
      payload: mode === "orders" ? { mode, records: [] } : { mode, records: [] },
      errors: [error instanceof Error ? error.message : "文件解析失败。"],
      warnings: [],
    };
  }
}

function formatFileSize(bytes: number) {
  if (bytes > 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function previewRows(payload?: ParsedPayload) {
  if (!payload) return [];
  if (payload.mode === "orders") {
    return payload.records.slice(0, 5).map((order) => ({
      id: order.order_id,
      main: order.platform,
      sub: `${order.region} · ${order.items.length} 个 SKU · ${order.priority}`,
    }));
  }
  return payload.records.slice(0, 5).map((record) => ({
    id: record.sku_id,
    main: record.warehouse_name,
    sub: `${record.region} · 可用 ${record.available_stock} · 锁定 ${record.locked_stock}`,
  }));
}

function FieldCheck({ parseState, requiredFields }: { parseState: ParseState | null; requiredFields: string[] }) {
  const matched = requiredFields.filter((field) => parseState?.headers.includes(field));
  const missing = requiredFields.filter((field) => !parseState?.headers.includes(field));

  return (
    <section className="workspace field-check-card">
      <div className="section-title">
        <div>
          <h2>字段校验</h2>
          <p>导入前先说明数据是否能进入后续智能履约链路。</p>
        </div>
        <FileCheck2 size={20} />
      </div>
      <div className="field-check-grid">
        <article><span>已匹配</span><strong>{matched.length}</strong></article>
        <article><span>缺失</span><strong>{parseState ? missing.length : requiredFields.length}</strong></article>
        <article><span>可导入</span><strong>{parseState?.payload.records.length ?? 0}</strong></article>
        <article><span>错误</span><strong>{parseState?.errors.length ?? 0}</strong></article>
      </div>
      <div className="tag-row">
        {requiredFields.map((field) => (
          <span className={parseState?.headers.includes(field) ? "tag tag--ok" : "tag tag--danger"} key={field}>
            {field}
          </span>
        ))}
      </div>
    </section>
  );
}

export function DataImportPage() {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const queryClient = useQueryClient();
  const [mode, setMode] = useState<ImportMode>("orders");
  const [sourceName, setSourceName] = useState("OMS 每日订单快照");
  const [sourceId, setSourceId] = useState("oms-daily");
  const [replaceSource, setReplaceSource] = useState(false);
  const [parseState, setParseState] = useState<ParseState | null>(null);
  const [result, setResult] = useState<EnterpriseImportResult | null>(null);
  const stats = useQuery({ queryKey: ["enterprise-stats"], queryFn: getEnterpriseStats });
  const requiredFields = mode === "orders" ? orderColumns : inventoryColumns;
  const preview = useMemo(() => previewRows(parseState?.payload), [parseState]);

  const importMutation = useMutation({
    mutationFn: async () => {
      if (!parseState || parseState.errors.length > 0) throw new Error("请先上传并通过基础校验。");
      const normalizedSourceId = normalizeSourceId(sourceId || sourceName);
      await upsertEnterpriseSource({
        sourceId: normalizedSourceId,
        name: sourceName,
        sourceType: "json",
        description: mode === "orders" ? "OMS / ERP 订单导入" : "WMS 多仓库存导入",
      });
      if (parseState.payload.mode === "orders") {
        return importEnterpriseOrders({ sourceId: normalizedSourceId, orders: parseState.payload.records, replaceSource });
      }
      return importEnterpriseInventory({ sourceId: normalizedSourceId, records: parseState.payload.records, replaceSource });
    },
    onSuccess: (response) => {
      setResult(response.data);
      queryClient.invalidateQueries({ queryKey: ["enterprise-stats"] });
    },
  });

  async function handleFile(file?: File) {
    if (!file) return;
    const text = await file.text();
    setParseState(parseFile(file.name, file.size, text, mode));
    setResult(null);
  }

  return (
    <div className="page page--demo">
      <header className="page-header demo-hero demo-hero--focused">
        <div>
          <p className="eyebrow">企业数据接入</p>
          <h1>企业数据接入</h1>
          <p>上传 OMS / ERP 订单和 WMS 库存文件，完成字段校验与预览后导入系统，供后续履约分析使用。</p>
        </div>
        <div className="demo-metric-row">
          <article className="demo-metric"><span>数据源</span><strong>{stats.data?.data.source_count ?? 0}</strong><small>已接入</small></article>
          <article className="demo-metric"><span>订单</span><strong>{stats.data?.data.order_count ?? 0}</strong><small>企业订单</small></article>
          <article className="demo-metric"><span>库存</span><strong>{stats.data?.data.inventory_record_count ?? 0}</strong><small>WMS 记录</small></article>
        </div>
      </header>

      <div className="data-simple-grid data-simple-grid--focused">
        <section className="workspace upload-workspace">
          <div className="section-title">
            <div>
              <h2>上传配置</h2>
              <p>选择数据类型、绑定来源，再上传 CSV / JSON 文件。</p>
            </div>
            <Database size={20} />
          </div>

          <div className="segmented-control">
            <button className={mode === "orders" ? "selected" : ""} type="button" onClick={() => setMode("orders")}>订单数据</button>
            <button className={mode === "inventory" ? "selected" : ""} type="button" onClick={() => setMode("inventory")}>库存数据</button>
          </div>

          <div className="form-grid form-grid--two">
            <label>
              <span>数据源名称</span>
              <input value={sourceName} onChange={(event) => setSourceName(event.target.value)} />
            </label>
            <label>
              <span>数据源 ID</span>
              <input value={sourceId} onChange={(event) => setSourceId(event.target.value)} />
            </label>
          </div>

          <label className="checkbox-row">
            <input type="checkbox" checked={replaceSource} onChange={(event) => setReplaceSource(event.target.checked)} />
            <span>覆盖同来源旧数据</span>
          </label>

          <div className="upload-zone upload-zone--compact" onClick={() => fileRef.current?.click()}>
            <UploadCloud size={30} />
            <strong>{parseState ? parseState.fileName : "选择 JSON 或 CSV 文件"}</strong>
            <span>{parseState ? formatFileSize(parseState.fileSize) : "建议单文件不超过 10MB"}</span>
            <button className="secondary-button" type="button">选择文件</button>
            <input ref={fileRef} hidden type="file" accept=".csv,.json" onChange={(event) => handleFile(event.target.files?.[0])} />
          </div>

          {parseState?.errors.map((error) => <div className="error-box" key={error}>{error}</div>)}
          {parseState?.warnings.slice(0, 3).map((warning) => <div className="info-box" key={warning}>{warning}</div>)}
          {result && <div className="success-box">导入完成：成功 {result.imported_count} 条。</div>}
          {importMutation.error && <div className="error-box">{importMutation.error instanceof Error ? importMutation.error.message : "导入失败"}</div>}

          <button className="primary-button full-width" type="button" disabled={!parseState || importMutation.isPending} onClick={() => importMutation.mutate()}>
            {importMutation.isPending ? "导入中..." : "导入企业数据"}
          </button>
        </section>

        <div className="side-stack">
          <FieldCheck parseState={parseState} requiredFields={requiredFields} />
          <section className="workspace">
            <div className="section-title">
              <div>
                <h2>前 5 行预览</h2>
                <p>上传后先校验并预览，再导入。</p>
              </div>
            </div>
            {preview.length ? (
              <div className="preview-list">
                {preview.map((row) => (
                  <article key={`${row.id}-${row.sub}`}>
                    <strong>{row.id}</strong>
                    <span>{row.main} · {row.sub}</span>
                  </article>
                ))}
              </div>
            ) : (
              <div className="empty-mini">还没有选择文件。</div>
            )}
          </section>
        </div>
      </div>

      <section className="workspace">
        <div className="section-title">
          <div>
            <h2>最近导入</h2>
            <p>展示最近批次的导入结果，便于运营人员追踪数据来源和失败数量。</p>
          </div>
        </div>
        <div className="history-compact">
          {importHistory.slice(0, 3).map((item) => (
            <article key={item.batch}>
              <CheckCircle2 size={18} />
              <div>
                <strong>{item.file}</strong>
                <span>{item.type} · 成功 {item.success} / 失败 {item.failed} · {item.time}</span>
              </div>
              <StatusPill tone={item.status === "完成" ? "ok" : "warn"}>{item.status}</StatusPill>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
