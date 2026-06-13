import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  ClipboardCheck,
  Database,
  FileCheck2,
  FileJson2,
  Network,
  ServerCog,
  ShieldCheck,
  TableProperties,
  UploadCloud,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";
import * as XLSX from "xlsx";

import { StatusPill } from "../components/StatusPill";
import {
  getEnterpriseStats,
  getKnowledgeIndexStatus,
  importEnterpriseInventory,
  importEnterpriseOrders,
  listEnterpriseSources,
  type EnterpriseImportResult,
  type InventoryRecord,
  type KnowledgeUploadResult,
  type OrderRecord,
  upsertEnterpriseSource,
  uploadKnowledgeDocument,
} from "../lib/api";

type ImportMode = "orders" | "inventory" | "rules";
type IngestionChannelKey = "file" | "api" | "sftp" | "database";
type ParsedPayload =
  | { mode: "orders"; records: OrderRecord[] }
  | { mode: "inventory"; records: InventoryRecord[] }
  | { mode: "rules"; records: RuleDocument[] };

type RuleDocument = {
  documentId: string;
  fileName: string;
  format: string;
  title: string;
  category: string;
  content: string;
  sourceFile?: File;
  size: number;
};

type ParseState = {
  fileName: string;
  fileSize: number;
  headers: string[];
  payload: ParsedPayload;
  errors: string[];
  warnings: string[];
};

type ImportFileState = {
  id: string;
  fileName: string;
  fileSize: number;
  format: string;
  recordCount: number;
  status: "ready" | "error";
  errors: string[];
  warnings: string[];
};

const IMPORT_CHUNK_SIZE = 1000;

const orderColumns = [
  "order_no",
  "platform_order_no",
  "platform",
  "shop_name",
  "paid_at",
  "order_status",
  "buyer_level",
  "province",
  "city",
  "sku_code",
  "sku_name",
  "sku_qty",
  "sale_price",
  "warehouse_code",
  "fulfillment_priority",
  "promised_ship_at",
];

const inventoryColumns = [
  "warehouse_code",
  "warehouse_name",
  "warehouse_region",
  "sku_code",
  "sku_name",
  "sellable_qty",
  "locked_qty",
  "in_transit_qty",
  "safety_stock",
  "batch_no",
  "updated_at",
];

const ruleColumns = [
  "rule_id",
  "rule_name",
  "rule_category",
  "trigger_condition",
  "action",
  "priority",
  "effective_from",
  "owner",
];

const enterpriseDataTypes = [
  { name: "订单", owner: "OMS / ERP / 平台", status: "实时接入", detail: "通过 API / Webhook / 数据库同步获取订单最新状态" },
  { name: "库存", owner: "WMS", status: "实时接入", detail: "通过 WMS API / 库存事件 / CDC 同步可售与锁定库存" },
  { name: "商品主数据", owner: "PIM / ERP", status: "规划中", detail: "SKU 编码、品名、类目、重量体积、替代关系" },
  { name: "仓配时效", owner: "TMS / WMS", status: "规划中", detail: "仓库覆盖区域、截单时间、承运商 SLA" },
  { name: "业务规则", owner: "运营 / 售后", status: "文档导入", detail: "PDF / Word / Markdown 进入 RAG 知识库" },
  { name: "客户与会员", owner: "CRM", status: "规划中", detail: "客户等级、黑白名单、售后偏好、履约限制" },
];

const ingestionChannels: Array<{
  key: IngestionChannelKey;
  title: string;
  icon: typeof FileJson2;
  status: string;
  detail: string;
}> = [
  { key: "api", title: "订单/库存实时 API", icon: Network, status: "推荐", detail: "OMS/WMS 通过 API 或 Webhook 推送订单和库存变更。" },
  { key: "database", title: "数据库 CDC 同步", icon: Database, status: "企业常用", detail: "连接业务库、数仓或变更流，获取准实时履约状态。" },
  { key: "sftp", title: "SFTP 批次补偿", icon: ServerCog, status: "补偿通道", detail: "用于主数据、历史补数、对账文件，不作为订单库存主链路。" },
  { key: "file", title: "RAG 规则文档导入", icon: FileJson2, status: "当前可用", detail: "上传 PDF / Word / Markdown / TXT，进入规则知识库。" },
];

const qualityRules = [
  { label: "业务主键完整", detail: "订单号、平台单号、SKU、仓库编码不能为空", level: "阻断" },
  { label: "数量口径合法", detail: "购买数量、可售库存、锁定库存必须是非异常数值", level: "阻断" },
  { label: "来源可追溯", detail: "每条记录绑定 source_id 和导入批次", level: "阻断" },
  { label: "时效字段完整", detail: "付款时间、承诺发货时间、库存更新时间用于 SLA 判断", level: "提醒" },
];

const orderFieldDescriptions: Record<string, string> = {
  order_no: "OMS 内部订单号",
  platform_order_no: "平台订单号",
  platform: "销售平台",
  shop_name: "店铺名称",
  paid_at: "付款时间",
  order_status: "订单状态",
  buyer_level: "会员/客户等级",
  province: "收货省份",
  city: "收货城市",
  sku_code: "商家 SKU 编码",
  sku_name: "商品名称",
  sku_qty: "购买数量",
  sale_price: "成交单价",
  warehouse_code: "履约仓编码",
  fulfillment_priority: "履约优先级",
  promised_ship_at: "承诺发货时间",
};

const inventoryFieldDescriptions: Record<string, string> = {
  warehouse_code: "仓库编码",
  warehouse_name: "仓库名称",
  warehouse_region: "仓库区域",
  sku_code: "商家 SKU 编码",
  sku_name: "商品名称",
  sellable_qty: "可售库存",
  locked_qty: "已锁库存",
  in_transit_qty: "在途库存",
  safety_stock: "安全库存",
  batch_no: "库存批次号",
  updated_at: "库存更新时间",
};

const ruleFieldDescriptions: Record<string, string> = {
  rule_id: "规则编号",
  rule_name: "规则名称",
  rule_category: "规则分类",
  trigger_condition: "触发条件",
  action: "处理动作",
  priority: "优先级",
  effective_from: "生效时间",
  owner: "规则负责人",
};

const fieldAliases: Record<ImportMode, Record<string, string[]>> = {
  orders: {
    order_no: ["order_no", "order_id", "orderId", "oms_order_no"],
    platform_order_no: ["platform_order_no", "platformOrderNo", "external_order_no"],
    platform: ["platform", "channel"],
    shop_name: ["shop_name", "shop", "store_name"],
    paid_at: ["paid_at", "order_time", "created_at"],
    order_status: ["order_status", "status"],
    buyer_level: ["buyer_level", "customer_level", "member_level"],
    province: ["province", "receiver_province"],
    city: ["city", "receiver_city"],
    sku_code: ["sku_code", "sku_id", "sku", "seller_sku"],
    sku_name: ["sku_name", "product_name", "product_title"],
    sku_qty: ["sku_qty", "quantity", "qty"],
    sale_price: ["sale_price", "unit_price", "price"],
    warehouse_code: ["warehouse_code", "warehouse_id", "warehouse"],
    fulfillment_priority: ["fulfillment_priority", "priority"],
    promised_ship_at: ["promised_ship_at", "sla_deadline", "ship_before"],
  },
  inventory: {
    warehouse_code: ["warehouse_code", "warehouse_id", "warehouse"],
    warehouse_name: ["warehouse_name"],
    warehouse_region: ["warehouse_region", "region"],
    sku_code: ["sku_code", "sku_id", "sku", "seller_sku"],
    sku_name: ["sku_name", "product_name", "product_title"],
    sellable_qty: ["sellable_qty", "available_qty", "available_stock"],
    locked_qty: ["locked_qty", "reserved_qty", "locked_stock"],
    in_transit_qty: ["in_transit_qty", "inbound_qty"],
    safety_stock: ["safety_stock"],
    batch_no: ["batch_no", "stock_batch_no"],
    updated_at: ["updated_at", "stock_updated_at"],
  },
  rules: {
    rule_id: ["rule_id", "rule_code", "id"],
    rule_name: ["rule_name", "title", "name"],
    rule_category: ["rule_category", "category", "tag"],
    trigger_condition: ["trigger_condition", "condition", "trigger"],
    action: ["action", "suggested_action", "handling_action"],
    priority: ["priority", "rule_priority"],
    effective_from: ["effective_from", "effective_at", "start_date"],
    owner: ["owner", "department", "maintainer"],
  },
};

function normalizeSourceId(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, "-") || "manual";
}

function parseDelimited(text: string, delimiter = ",") {
  const rows = text.trim().split(/\r?\n/).filter(Boolean);
  if (rows.length === 0) return [];
  const headers = rows[0].split(delimiter).map((item) => item.replace(/^\uFEFF/, "").trim());
  return rows.slice(1).map((row) => {
    const cells = row.split(delimiter).map((item) => item.trim());
    return Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""]));
  });
}

function fileFormat(fileName: string) {
  return fileName.split(".").pop()?.toLowerCase() || "unknown";
}

function readJsonRows(text: string, mode: ImportMode) {
  const raw = JSON.parse(text) as unknown;
  if (Array.isArray(raw)) return raw as Record<string, unknown>[];
  const object = raw as { orders?: unknown[]; records?: unknown[]; inventory?: unknown[]; rules?: unknown[]; documents?: unknown[]; data?: unknown[] };
  if (mode === "orders") return (object.orders || object.records || object.data || []) as Record<string, unknown>[];
  if (mode === "rules") return (object.rules || object.documents || object.records || object.data || []) as Record<string, unknown>[];
  return (object.inventory || object.records || object.data || []) as Record<string, unknown>[];
}

function readJsonLines(text: string) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

function readWorkbookRows(buffer: ArrayBuffer) {
  const workbook = XLSX.read(buffer, { type: "array" });
  const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
  if (!firstSheet) return [];
  return XLSX.utils.sheet_to_json<Record<string, unknown>>(firstSheet, { defval: "", raw: false });
}

function numberValue(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function firstValue(row: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    if (row[key] !== undefined && row[key] !== "") return row[key];
  }
  return "";
}

function valueFor(row: Record<string, unknown>, mode: ImportMode, field: string) {
  return firstValue(row, fieldAliases[mode][field] || [field]);
}

function hasRequiredField(headers: string[], mode: ImportMode, field: string) {
  return (fieldAliases[mode][field] || [field]).some((alias) => headers.includes(alias));
}

function regionFromOrder(row: Record<string, unknown>) {
  const explicitRegion = String(firstValue(row, ["fulfillment_region", "region"])).trim();
  if (explicitRegion) return explicitRegion;
  return [valueFor(row, "orders", "province"), valueFor(row, "orders", "city")]
    .map((value) => String(value).trim())
    .filter(Boolean)
    .join("/");
}

function documentIdFromFile(fileName: string) {
  return fileName
    .replace(/\.[^.]+$/, "")
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}._-]+/gu, "-")
    .replace(/^-+|-+$/g, "") || "business-rule";
}

function categoryFromRuleFile(fileName: string) {
  const lower = fileName.toLowerCase();
  if (lower.includes("stockout") || lower.includes("缺货")) return "stockout_rule";
  if (lower.includes("priority") || lower.includes("优先级") || lower.includes("高优")) return "priority_rule";
  if (lower.includes("regional") || lower.includes("区域") || lower.includes("跨仓") || lower.includes("调拨")) return "regional_strategy";
  if (lower.includes("split") || lower.includes("merge") || lower.includes("拆单") || lower.includes("合单")) return "split_merge_rule";
  if (lower.includes("after") || lower.includes("售后") || lower.includes("补发") || lower.includes("退换货")) return "after_sales_rule";
  return "general";
}

function ruleRowsToMarkdown(fileName: string, rows: Record<string, unknown>[], mode: ImportMode = "rules") {
  const title = fileName.replace(/\.[^.]+$/, "");
  const category = categoryFromRuleFile(fileName);
  const lines = [`# ${title}`, "", `规则分类：${category}`, "", "## 规则清单"];
  rows.forEach((row, index) => {
    const ruleName = String(valueFor(row, mode, "rule_name") || `规则 ${index + 1}`);
    lines.push("", `### ${ruleName}`);
    const fields = mode === "rules" ? ruleColumns : Object.keys(row);
    fields.forEach((field) => {
      const value = mode === "rules" ? valueFor(row, mode, field) : row[field];
      if (value !== undefined && value !== "") {
        lines.push(`- ${ruleFieldDescriptions[field] || field}：${String(value)}`);
      }
    });
  });
  return lines.join("\n");
}

async function readRows(file: File, mode: ImportMode) {
  const format = fileFormat(file.name);
  if (format === "xlsx" || format === "xls") {
    return readWorkbookRows(await file.arrayBuffer());
  }
  const text = await file.text();
  if (format === "json") return readJsonRows(text, mode);
  if (format === "jsonl" || format === "ndjson") return readJsonLines(text);
  if (format === "tsv") return parseDelimited(text, "\t");
  return parseDelimited(text);
}

function parseOrders(rows: Record<string, unknown>[]) {
  const orderMap = new Map<string, OrderRecord>();
  rows.forEach((row) => {
    const orderId = String(valueFor(row, "orders", "order_no")).trim();
    if (!orderId) return;
    const item = {
      sku_id: String(valueFor(row, "orders", "sku_code") || ""),
      product_name: String(valueFor(row, "orders", "sku_name") || ""),
      quantity: numberValue(valueFor(row, "orders", "sku_qty")),
      unit_price: numberValue(valueFor(row, "orders", "sale_price")),
    };
    const existing = orderMap.get(orderId);
    if (existing) existing.items.push(item);
    else {
      orderMap.set(orderId, {
        order_id: orderId,
        platform: String(valueFor(row, "orders", "platform") || ""),
        order_time: String(valueFor(row, "orders", "paid_at") || ""),
        order_status: String(valueFor(row, "orders", "order_status") || ""),
        region: regionFromOrder(row),
        priority: String(valueFor(row, "orders", "fulfillment_priority") || ""),
        items: [item],
      });
    }
  });
  return [...orderMap.values()];
}

function parseInventory(rows: Record<string, unknown>[]): InventoryRecord[] {
  return rows.map((row) => ({
    warehouse_id: String(valueFor(row, "inventory", "warehouse_code") || ""),
    warehouse_name: String(valueFor(row, "inventory", "warehouse_name") || ""),
    region: String(valueFor(row, "inventory", "warehouse_region") || ""),
    sku_id: String(valueFor(row, "inventory", "sku_code") || ""),
    available_stock: numberValue(valueFor(row, "inventory", "sellable_qty")),
    locked_stock: numberValue(valueFor(row, "inventory", "locked_qty")),
    updated_at: String(valueFor(row, "inventory", "updated_at") || ""),
  }));
}

function parseRuleRows(fileName: string, fileSize: number, rows: Record<string, unknown>[]): RuleDocument {
  return {
    documentId: documentIdFromFile(fileName),
    fileName,
    format: fileFormat(fileName),
    title: fileName.replace(/\.[^.]+$/, ""),
    category: categoryFromRuleFile(fileName),
    content: ruleRowsToMarkdown(fileName, rows),
    size: fileSize,
  };
}

async function parseRuleDocument(file: File): Promise<RuleDocument> {
  const format = fileFormat(file.name);
  if (format === "md" || format === "txt") {
    const content = await file.text();
    return {
      documentId: documentIdFromFile(file.name),
      fileName: file.name,
      format,
      title: file.name.replace(/\.[^.]+$/, ""),
      category: categoryFromRuleFile(file.name),
      content,
      sourceFile: file,
      size: file.size,
    };
  }
  if (format === "pdf" || format === "docx") {
    return {
      documentId: documentIdFromFile(file.name),
      fileName: file.name,
      format,
      title: file.name.replace(/\.[^.]+$/, ""),
      category: categoryFromRuleFile(file.name),
      content: `${file.name} will be parsed by the backend ingestion pipeline.`,
      sourceFile: file,
      size: file.size,
    };
  }
  if (format === "doc") {
    throw new Error("暂不支持旧版 .doc，请转换为 .docx 或 PDF 后上传。");
  }
  const rows = await readRows(file, "rules");
  return parseRuleRows(file.name, file.size, rows);
}

function parseRows(fileName: string, fileSize: number, rows: Record<string, unknown>[], mode: Exclude<ImportMode, "rules">): ParseState {
  const required = mode === "orders" ? orderColumns : inventoryColumns;
  try {
    const headers = Object.keys(rows[0] || {});
    const missing = required.filter((field) => !hasRequiredField(headers, mode, field));
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

async function parseDataFile(file: File, mode: ImportMode): Promise<ParseState> {
  try {
    if (mode === "rules") {
      const document = await parseRuleDocument(file);
      const format = fileFormat(file.name);
      const headers = ["md", "txt", "pdf", "docx", "doc"].includes(format)
        ? []
        : Object.keys((await readRows(file, mode))[0] || {});
      return {
        fileName: file.name,
        fileSize: file.size,
        headers,
        payload: { mode, records: [document] },
        errors: document.content.trim() ? [] : ["规则文件没有可导入内容。"],
        warnings: [],
      };
    }
    const rows = await readRows(file, mode);
    return parseRows(file.name, file.size, rows, mode);
  } catch (error) {
    return {
      fileName: file.name,
      fileSize: file.size,
      headers: [],
      payload: mode === "orders" ? { mode, records: [] } : mode === "inventory" ? { mode, records: [] } : { mode, records: [] },
      errors: [error instanceof Error ? error.message : "文件解析失败。"],
      warnings: [],
    };
  }
}

function mergeParseStates(states: ParseState[], mode: ImportMode): ParseState | null {
  if (states.length === 0) return null;
  const headers = [...new Set(states.flatMap((state) => state.headers))];
  const fileSize = states.reduce((total, state) => total + state.fileSize, 0);
  const errors = states.flatMap((state) => state.errors.map((error) => `${state.fileName}: ${error}`));
  const warnings = states.flatMap((state) => state.warnings.map((warning) => `${state.fileName}: ${warning}`));
  if (mode === "orders") {
    const records = states.flatMap((state) => state.payload.mode === "orders" ? state.payload.records : []);
    return {
      fileName: states.length === 1 ? states[0].fileName : `${states.length} 个订单文件`,
      fileSize,
      headers,
      payload: { mode, records },
      errors,
      warnings,
    };
  }
  if (mode === "rules") {
    const records = states.flatMap((state) => state.payload.mode === "rules" ? state.payload.records : []);
    return {
      fileName: states.length === 1 ? states[0].fileName : `${states.length} 个规则文件`,
      fileSize,
      headers,
      payload: { mode, records },
      errors,
      warnings,
    };
  }
  const records = states.flatMap((state) => state.payload.mode === "inventory" ? state.payload.records : []);
  return {
    fileName: states.length === 1 ? states[0].fileName : `${states.length} 个库存文件`,
    fileSize,
    headers,
    payload: { mode, records },
    errors,
    warnings,
  };
}

function formatFileSize(bytes: number) {
  if (bytes > 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function previewRows(payload?: ParsedPayload) {
  if (!payload) return [];
  if (payload.mode === "rules") {
    return payload.records.slice(0, 5).map((document) => ({
      id: document.documentId,
      main: document.category,
      sub: `${document.format.toUpperCase()} · ${formatFileSize(document.size)}`,
    }));
  }
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

function FieldCheck({ mode, parseState, requiredFields }: { mode: ImportMode; parseState: ParseState | null; requiredFields: string[] }) {
  if (mode === "rules") {
    const documents = parseState?.payload.mode === "rules" ? parseState.payload.records : [];
    return (
      <section className="workspace field-check-card">
        <div className="section-title">
          <div>
            <h2>规则文档校验</h2>
            <p>RAG 规则以文档内容为主，上传后统一转为 Markdown 进入知识库索引。</p>
          </div>
          <FileCheck2 size={20} />
        </div>
        <div className="field-check-grid">
          <article><span>文档数</span><strong>{documents.length}</strong></article>
          <article><span>总大小</span><strong>{parseState ? formatFileSize(parseState.fileSize) : "--"}</strong></article>
          <article><span>错误</span><strong>{parseState?.errors.length ?? 0}</strong></article>
          <article><span>待索引</span><strong>{documents.length}</strong></article>
        </div>
        <div className="tag-row">
          {["PDF", "Word .docx", "Markdown", "TXT", "规则表"].map((format) => (
            <span className="tag tag--ok" key={format}>{format}</span>
          ))}
        </div>
      </section>
    );
  }

  function hasField(field: string) {
    if (!parseState) return false;
    return hasRequiredField(parseState.headers, mode, field);
  }

  const matched = requiredFields.filter((field) => hasField(field));
  const missing = requiredFields.filter((field) => !hasField(field));

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
          <span className={hasField(field) ? "tag tag--ok" : "tag tag--danger"} key={field}>
            {field}
          </span>
        ))}
      </div>
    </section>
  );
}

function KnowledgeIndexStatusPanel({
  status,
}: {
  status?: {
    document_count: number;
    indexed_document_count: number;
    total_chunk_count: number;
    embedding_ready: boolean;
    vector_store_type: string;
    rebuild_running: boolean;
    last_rebuild_error?: string | null;
  };
}) {
  const ready = Boolean(status?.embedding_ready);
  const running = Boolean(status?.rebuild_running);
  const tone = ready ? "ok" : running ? "info" : "warn";
  const label = ready ? "Embedding 已完成" : running ? "索引重建中" : "Embedding 未完成";

  return (
    <section className="workspace">
      <div className="section-title">
        <div>
          <h2>知识库索引状态</h2>
          <p>文档入库不等于已经可向量检索，这里显示最新索引结果。</p>
        </div>
        <StatusPill tone={tone}>{label}</StatusPill>
      </div>
      <div className="field-check-grid">
        <article><span>正式文档</span><strong>{status?.document_count ?? 0}</strong></article>
        <article><span>已索引文档</span><strong>{status?.indexed_document_count ?? 0}</strong></article>
        <article><span>向量片段</span><strong>{status?.total_chunk_count ?? 0}</strong></article>
        <article><span>向量库</span><strong>{status?.vector_store_type || "--"}</strong></article>
      </div>
      {status?.last_rebuild_error && (
        <div className="error-box">{status.last_rebuild_error}</div>
      )}
    </section>
  );
}

function EnterpriseDataTypes() {
  function tone(statusText: string) {
    return statusText === "实时接入" || statusText === "文档导入" || statusText === "已支持" ? "ok" : "neutral";
  }

  return (
    <section className="workspace">
      <div className="section-title">
        <div>
          <h2>数据域覆盖</h2>
          <p>按企业系统边界拆分，避免把所有文件混成一个上传入口。</p>
        </div>
      </div>
      <div className="data-type-list">
        {enterpriseDataTypes.map((item) => (
          <article key={item.name}>
            <div>
              <strong>{item.name}</strong>
              <span>{item.owner} · {item.detail}</span>
            </div>
            <StatusPill tone={tone(item.status)}>{item.status}</StatusPill>
          </article>
        ))}
      </div>
    </section>
  );
}

function RealtimeConnectionGuide({ channel }: { channel: IngestionChannelKey }) {
  const items = channel === "api"
    ? [
      ["订单事件", "订单创建、付款、取消、发货、售后状态通过 Webhook/API 增量同步。"],
      ["库存事件", "WMS 推送可售库存、锁定库存、释放库存、缺货变更。"],
      ["查询兜底", "AI 分析时可实时查询 OMS/WMS，避免基于过期快照决策。"],
    ]
    : channel === "database"
      ? [
        ["只读视图", "对接订单、库存、仓库、SKU 的只读视图或数仓表。"],
        ["CDC 增量", "按 updated_at、binlog 或消息流同步准实时变更。"],
        ["缓存策略", "保留短 TTL 本地缓存，失败时降级提示数据可能过期。"],
      ]
      : [
        ["补偿文件", "用于商品主数据、SLA 参数、历史补数和对账，不作为实时库存主链路。"],
        ["批次留痕", "每个文件记录批次号、来源系统、校验结果和失败明细。"],
        ["人工复核", "异常批次进入复核，不直接覆盖实时订单/库存状态。"],
      ];

  return (
    <section className="workspace">
      <div className="section-title">
        <div>
          <h2>实时数据边界</h2>
          <p>订单和库存是动态状态，应通过接口、事件或 CDC 接入，而不是运营手工上传。</p>
        </div>
        <ShieldCheck size={20} />
      </div>
      <div className="quality-gate-list">
        {items.map(([title, detail]) => (
          <article key={title}>
            <ClipboardCheck size={17} />
            <div>
              <strong>{title}</strong>
              <span>{detail}</span>
            </div>
            <StatusPill tone="ok">建议</StatusPill>
          </article>
        ))}
      </div>
    </section>
  );
}

function ConnectorCatalog({
  selected,
  onSelect,
}: {
  selected: IngestionChannelKey;
  onSelect: (channel: IngestionChannelKey) => void;
}) {
  return (
    <section className="workspace">
      <div className="section-title">
        <div>
          <h2>接入方式</h2>
          <p>订单和库存走实时接入；RAG 规则走文档导入；批次文件只做补偿和治理。</p>
        </div>
        <ServerCog size={20} />
      </div>
      <div className="connector-grid">
        {ingestionChannels.map((channel) => {
          const Icon = channel.icon;
          return (
            <button
              className={selected === channel.key ? "selected" : ""}
              key={channel.key}
              type="button"
              onClick={() => onSelect(channel.key)}
            >
              <Icon size={18} />
              <div>
                <strong>{channel.title}</strong>
                <span>{channel.detail}</span>
              </div>
              <StatusPill tone={selected === channel.key ? "ok" : "info"}>{channel.status}</StatusPill>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function QualityGate({ parseState }: { parseState: ParseState | null }) {
  const hasBlockingIssue = Boolean(parseState?.errors.length || parseState?.warnings.length);
  return (
    <section className="workspace">
      <div className="section-title">
        <div>
          <h2>质量门禁</h2>
          <p>导入前先判断是否会影响履约分析可信度。</p>
        </div>
        <ShieldCheck size={20} />
      </div>
      <div className="quality-gate-list">
        {qualityRules.map((rule) => (
          <article key={rule.label}>
            <ClipboardCheck size={17} />
            <div>
              <strong>{rule.label}</strong>
              <span>{rule.detail}</span>
            </div>
            <StatusPill tone={rule.level === "阻断" ? "warn" : "info"}>{rule.level}</StatusPill>
          </article>
        ))}
      </div>
      <div className={hasBlockingIssue ? "data-readiness data-readiness--warn" : "data-readiness"}>
        <strong>{parseState ? (hasBlockingIssue ? "需要处理字段问题" : "基础校验通过") : "等待文件校验"}</strong>
        <span>{parseState ? `发现 ${parseState.errors.length} 个错误、${parseState.warnings.length} 个提醒。` : "上传文件后会生成校验结果。"}</span>
      </div>
    </section>
  );
}

function FieldMappingTable({ mode, parseState }: { mode: ImportMode; parseState: ParseState | null }) {
  if (mode === "rules") {
    return (
      <section className="workspace">
        <div className="section-title">
          <div>
            <h2>规则文件格式</h2>
            <p>RAG 优先接收运营规则、SOP、政策说明和异常处理文档。</p>
          </div>
          <TableProperties size={20} />
        </div>
        <div className="field-map-table">
          {[
            ["PDF", "适合正式制度、SOP、仓配政策、售后政策"],
            ["Word .docx", "适合运营团队维护的规则说明和流程文档"],
            ["Markdown / TXT", "适合可审查、可 diff 的知识库文档"],
            ["CSV / JSON / Excel", "仅适合规则台账、参数表，会自动转成 Markdown"],
          ].map(([format, description]) => (
            <article key={format}>
              <span>{format}</span>
              <strong>{description}</strong>
              <StatusPill tone="ok">可导入</StatusPill>
            </article>
          ))}
        </div>
      </section>
    );
  }

  const requiredFields = mode === "orders" ? orderColumns : inventoryColumns;
  const descriptions = mode === "orders" ? orderFieldDescriptions : inventoryFieldDescriptions;

  function hasField(field: string) {
    if (!parseState) return false;
    return hasRequiredField(parseState.headers, mode, field);
  }

  return (
    <section className="workspace">
      <div className="section-title">
        <div>
          <h2>字段映射</h2>
          <p>企业导入最容易出问题的是字段口径，不只是文件能解析。</p>
        </div>
        <TableProperties size={20} />
      </div>
      <div className="field-map-table">
        {requiredFields.map((field) => (
          <article key={field}>
            <span>{field}</span>
            <strong>{descriptions[field]}</strong>
            <StatusPill tone={hasField(field) ? "ok" : "neutral"}>{hasField(field) ? "已匹配" : "待匹配"}</StatusPill>
          </article>
        ))}
      </div>
    </section>
  );
}

function ConnectorSetupPanel({
  channel,
  sourceName,
  sourceId,
  sourceSystem,
  saving,
  savedMessage,
  onSourceNameChange,
  onSourceIdChange,
  onSourceSystemChange,
  onSave,
}: {
  channel: Exclude<IngestionChannelKey, "file">;
  sourceName: string;
  sourceId: string;
  sourceSystem: string;
  saving: boolean;
  savedMessage: string | null;
  onSourceNameChange: (value: string) => void;
  onSourceIdChange: (value: string) => void;
  onSourceSystemChange: (value: string) => void;
  onSave: () => void;
}) {
  const title = channel === "api" ? "API 接入配置" : channel === "sftp" ? "SFTP 同步配置" : "数据库同步配置";
  const hint = channel === "api"
    ? "用于 OMS/WMS/平台系统实时推送订单状态、库存变更和发货事件。"
    : channel === "sftp"
      ? "用于商品主数据、SLA 参数、历史补数和对账文件，不作为订单库存实时主链路。"
      : "用于连接企业业务库、数仓视图或 CDC 变更流，获取准实时履约状态。";

  return (
    <section className="workspace connector-setup-panel">
      <div className="section-title">
        <div>
          <h2>{title}</h2>
          <p>{hint}</p>
        </div>
        <StatusPill tone="info">配置入口</StatusPill>
      </div>

      <div className="form-grid form-grid--two">
        <label>
          <span>数据源名称</span>
          <input value={sourceName} onChange={(event) => onSourceNameChange(event.target.value)} />
        </label>
        <label>
          <span>数据源 ID</span>
          <input value={sourceId} onChange={(event) => onSourceIdChange(event.target.value)} />
        </label>
        <label>
          <span>来源系统</span>
          <select value={sourceSystem} onChange={(event) => onSourceSystemChange(event.target.value)}>
            <option>OMS</option>
            <option>ERP</option>
            <option>WMS</option>
            <option>TMS</option>
            <option>Shopify</option>
            <option>Amazon</option>
            <option>TikTok Shop</option>
            <option>Data Warehouse</option>
          </select>
        </label>
      </div>

      {channel === "api" && (
        <div className="connector-config-grid">
          <label><span>订单事件接口</span><input placeholder="https://oms.example.com/webhooks/orders" /></label>
          <label><span>库存事件接口</span><input placeholder="https://wms.example.com/webhooks/inventory" /></label>
          <label><span>鉴权方式</span><select><option>Bearer Token</option><option>API Key</option><option>OAuth 2.0</option></select></label>
          <label><span>同步模式</span><select><option>Webhook 实时推送</option><option>增量拉取</option><option>消息队列订阅</option></select></label>
        </div>
      )}

      {channel === "sftp" && (
        <div className="connector-config-grid">
          <label><span>Host</span><input placeholder="sftp.example.com" /></label>
          <label><span>目录</span><input placeholder="/exports/master-data/" /></label>
          <label><span>文件模式</span><input placeholder="sku_*.csv / sla_*.xlsx / reconcile_*.json" /></label>
          <label><span>同步频率</span><select><option>每日</option><option>每小时</option><option>按需触发</option></select></label>
        </div>
      )}

      {channel === "database" && (
        <div className="connector-config-grid">
          <label><span>数据库类型</span><select><option>PostgreSQL</option><option>MySQL</option><option>SQL Server</option><option>Snowflake</option></select></label>
          <label><span>订单视图 / Topic</span><input placeholder="dw.orders_realtime_v / oms.order.events" /></label>
          <label><span>库存视图 / Topic</span><input placeholder="dw.inventory_realtime_v / wms.stock.events" /></label>
          <label><span>增量字段</span><input placeholder="updated_at / event_time" /></label>
        </div>
      )}

      <div className="connector-action-row">
        <button className="secondary-button" type="button">测试连接</button>
        <button className="primary-button" type="button" disabled={saving} onClick={onSave}>
          {saving ? "保存中..." : "保存接入配置"}
        </button>
      </div>
      {savedMessage && <div className="success-box">{savedMessage}</div>}
    </section>
  );
}

export function DataImportPage() {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const queryClient = useQueryClient();
  const [mode, setMode] = useState<ImportMode>("rules");
  const [selectedChannel, setSelectedChannel] = useState<IngestionChannelKey>("api");
  const [sourceName, setSourceName] = useState("OMS/WMS 实时接入");
  const [sourceId, setSourceId] = useState("oms-wms-realtime");
  const [sourceSystem, setSourceSystem] = useState("OMS");
  const [replaceSource, setReplaceSource] = useState(false);
  const [parseState, setParseState] = useState<ParseState | null>(null);
  const [fileQueue, setFileQueue] = useState<ImportFileState[]>([]);
  const [result, setResult] = useState<EnterpriseImportResult | null>(null);
  const [knowledgeResult, setKnowledgeResult] = useState<KnowledgeUploadResult[] | null>(null);
  const [connectorSaved, setConnectorSaved] = useState<string | null>(null);
  const stats = useQuery({ queryKey: ["enterprise-stats"], queryFn: getEnterpriseStats });
  const sources = useQuery({ queryKey: ["enterprise-sources"], queryFn: listEnterpriseSources, retry: 1 });
  const knowledgeStatus = useQuery({ queryKey: ["knowledge-index-status"], queryFn: getKnowledgeIndexStatus, retry: 1 });
  const requiredFields = mode === "orders" ? orderColumns : mode === "inventory" ? inventoryColumns : ruleColumns;
  const preview = useMemo(() => previewRows(parseState?.payload), [parseState]);
  const sourceRows = sources.data?.data.sources ?? [];

  const importMutation = useMutation({
    mutationFn: async () => {
      if (!parseState || parseState.errors.length > 0) throw new Error("请先上传并通过基础校验。");
      if (parseState.payload.mode === "rules") {
        if (!sourceSystem.trim()) throw new Error("请填写规则来源。");
        const uploaded: KnowledgeUploadResult[] = [];
        for (const document of parseState.payload.records) {
          const file = document.sourceFile ?? new Blob([document.content], { type: "text/markdown;charset=utf-8" });
          const fileName = document.sourceFile ? document.fileName : `${document.documentId}.md`;
          const response = await uploadKnowledgeDocument({
            documentId: document.documentId,
            file,
            fileName,
            replaceExisting: replaceSource,
            rebuild: true,
            confirmBeforeIndex: false,
            category: document.category,
            title: document.title,
          });
          uploaded.push(response.data);
        }
        return {
          success: true,
          message: "规则知识导入完成。",
          data: {
            source_id: "rag-knowledge",
            imported_count: uploaded.length,
            total_orders: 0,
            total_inventory_records: 0,
            uploaded,
          },
        };
      }
      const normalizedSourceId = normalizeSourceId(sourceId || sourceName);
      await upsertEnterpriseSource({
        sourceId: normalizedSourceId,
        name: sourceName,
        sourceType: "json",
        description: `${sourceSystem} ${mode === "orders" ? "订单导入" : "多仓库存导入"}`,
      });
      if (parseState.payload.mode === "orders") {
        let aggregate: EnterpriseImportResult | null = null;
        let importedCount = 0;
        for (let index = 0; index < parseState.payload.records.length; index += IMPORT_CHUNK_SIZE) {
          const response = await importEnterpriseOrders({
            sourceId: normalizedSourceId,
            orders: parseState.payload.records.slice(index, index + IMPORT_CHUNK_SIZE),
            replaceSource: replaceSource && index === 0,
          });
          importedCount += response.data.imported_count;
          aggregate = {
            ...response.data,
            imported_count: importedCount,
          };
        }
        if (!aggregate) throw new Error("没有可导入订单。");
        return { success: true, message: "批量导入完成。", data: aggregate };
      }
      let aggregate: EnterpriseImportResult | null = null;
      let importedCount = 0;
      for (let index = 0; index < parseState.payload.records.length; index += IMPORT_CHUNK_SIZE) {
        const response = await importEnterpriseInventory({
          sourceId: normalizedSourceId,
          records: parseState.payload.records.slice(index, index + IMPORT_CHUNK_SIZE),
          replaceSource: replaceSource && index === 0,
        });
        importedCount += response.data.imported_count;
        aggregate = {
          ...response.data,
          imported_count: importedCount,
        };
      }
      if (!aggregate) throw new Error("没有可导入库存。");
      return { success: true, message: "批量导入完成。", data: aggregate };
    },
    onSuccess: (response) => {
      if ("uploaded" in response.data) {
        setKnowledgeResult(response.data.uploaded);
        setResult(null);
      } else {
        setResult(response.data);
        setKnowledgeResult(null);
      }
      queryClient.invalidateQueries({ queryKey: ["enterprise-stats"] });
      queryClient.invalidateQueries({ queryKey: ["knowledge-index-status"] });
    },
  });

  const connectorMutation = useMutation({
    mutationFn: async () => {
      if (selectedChannel === "file") return null;
      const normalizedSourceId = normalizeSourceId(sourceId || sourceName);
      return upsertEnterpriseSource({
        sourceId: normalizedSourceId,
        name: sourceName,
        sourceType: selectedChannel,
        description: `${sourceSystem} ${selectedChannel.toUpperCase()} 接入配置`,
      });
    },
    onSuccess: () => {
      setConnectorSaved("接入配置已保存，后续可由后端调度器执行同步任务。");
      queryClient.invalidateQueries({ queryKey: ["enterprise-sources"] });
      queryClient.invalidateQueries({ queryKey: ["enterprise-stats"] });
    },
  });

  async function handleFiles(files?: FileList | File[]) {
    const selectedFiles = Array.from(files || []);
    if (selectedFiles.length === 0) return;
    const states = await Promise.all(selectedFiles.map((file) => parseDataFile(file, mode)));
    setFileQueue(states.map((state) => ({
      id: `${state.fileName}-${state.fileSize}`,
      fileName: state.fileName,
      fileSize: state.fileSize,
      format: fileFormat(state.fileName).toUpperCase(),
      recordCount: state.payload.records.length,
      status: state.errors.length > 0 ? "error" : "ready",
      errors: state.errors,
      warnings: state.warnings,
    })));
    setParseState(mergeParseStates(states, mode));
    setResult(null);
    setKnowledgeResult(null);
  }

  function selectChannel(channel: IngestionChannelKey) {
    setSelectedChannel(channel);
    setConnectorSaved(null);
    if (channel === "api") {
      setSourceName("OMS/WMS 实时接入");
      setSourceId("oms-wms-realtime");
      setSourceSystem("OMS");
    } else if (channel === "sftp") {
      setSourceName("主数据 SLA 批次补偿");
      setSourceId("masterdata-sftp");
      setSourceSystem("ERP");
    } else if (channel === "database") {
      setSourceName("订单库存 CDC 同步");
      setSourceId("fulfillment-cdc");
      setSourceSystem("Data Warehouse");
    } else {
      setMode("rules");
      setSourceName("RAG 规则知识库");
      setSourceId("rag-rules");
      setSourceSystem("");
    }
  }

  return (
    <div className="page page--demo">
      <header className="page-header demo-hero demo-hero--focused">
        <div>
          <p className="eyebrow">Enterprise Data Ingestion</p>
          <h1>企业数据接入工作台</h1>
          <p>订单和库存通过实时接口、事件或 CDC 接入；规则、SOP 和政策文档进入 RAG 知识库，供 AI 履约分析检索。</p>
        </div>
        <div className="demo-metric-row">
          <article className="demo-metric"><span>数据源</span><strong>{stats.data?.data.source_count ?? 0}</strong><small>已接入</small></article>
          <article className="demo-metric"><span>订单</span><strong>{stats.data?.data.order_count ?? 0}</strong><small>企业订单</small></article>
          <article className="demo-metric"><span>库存</span><strong>{stats.data?.data.inventory_record_count ?? 0}</strong><small>WMS 记录</small></article>
          <article className="demo-metric"><span>更新时间</span><strong>{stats.data?.data.last_updated_at ? "已同步" : "--"}</strong><small>{stats.data?.data.last_updated_at || "等待导入"}</small></article>
        </div>
      </header>

      <ConnectorCatalog selected={selectedChannel} onSelect={selectChannel} />

      <div className="data-simple-grid data-simple-grid--focused">
        <div className="data-main-stack">
          {selectedChannel === "file" ? (
            <section className="workspace upload-workspace">
              <div className="section-title">
                <div>
                  <h2>RAG 规则文档导入</h2>
                  <p>上传履约规则、缺货 SOP、售后政策、仓配策略等文档，系统会转入知识库并重建索引。</p>
                </div>
                <Database size={20} />
              </div>

              <div className="form-grid form-grid--two">
                <label>
                  <span>知识库名称</span>
                  <input value={sourceName} onChange={(event) => setSourceName(event.target.value)} />
                </label>
                <label>
                  <span>文档批次 ID</span>
                  <input value={sourceId} onChange={(event) => setSourceId(event.target.value)} />
                </label>
                <label>
                  <span>规则来源</span>
                  <input
                    value={sourceSystem}
                    onChange={(event) => setSourceSystem(event.target.value)}
                    placeholder="填写规则来源，例如：运营规则库、售后知识库、仓配 SOP"
                    required
                  />
                </label>
              </div>

              <label className="checkbox-row">
                <input type="checkbox" checked={replaceSource} onChange={(event) => setReplaceSource(event.target.checked)} />
                <span>覆盖同名规则文档</span>
              </label>

              <div
                className="upload-zone upload-zone--compact"
                onClick={() => fileRef.current?.click()}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  void handleFiles(event.dataTransfer.files);
                }}
              >
                <UploadCloud size={30} />
                <strong>{fileQueue.length ? `已选择 ${fileQueue.length} 个文件` : "选择或拖拽规则知识文件"}</strong>
                <span>
                  {parseState
                    ? `${formatFileSize(parseState.fileSize)} · ${parseState.payload.records.length} 个文档`
                    : "支持 PDF / Word(.docx) / MD / TXT；规则表 CSV、JSON、Excel 也可转为 Markdown"}
                </span>
                <button className="secondary-button" type="button">批量选择规则文档</button>
                <input
                  ref={fileRef}
                  hidden
                  multiple
                  type="file"
                  accept=".pdf,.docx,.md,.txt,.csv,.tsv,.json,.jsonl,.ndjson,.xlsx,.xls"
                  onChange={(event) => void handleFiles(event.target.files || undefined)}
                />
              </div>

              {fileQueue.length > 0 && (
                <div className="file-queue-list">
                  {fileQueue.map((file) => (
                    <article key={file.id}>
                      <div>
                        <strong>{file.fileName}</strong>
                        <span>{file.format} · {formatFileSize(file.fileSize)} · {file.recordCount} 条记录</span>
                      </div>
                      <StatusPill tone={file.status === "ready" ? "ok" : "danger"}>{file.status === "ready" ? "待导入" : "解析失败"}</StatusPill>
                    </article>
                  ))}
                </div>
              )}

              {parseState?.errors.map((error) => <div className="error-box" key={error}>{error}</div>)}
              {parseState?.warnings.slice(0, 5).map((warning) => <div className="info-box" key={warning}>{warning}</div>)}
              {knowledgeResult && (
                <div className="success-box">
                  规则知识入库完成：{knowledgeResult.length} 个文档已入库。
                </div>
              )}
              {importMutation.error && <div className="error-box">{importMutation.error instanceof Error ? importMutation.error.message : "导入失败"}</div>}

              <button className="primary-button full-width" type="button" disabled={!parseState || importMutation.isPending} onClick={() => importMutation.mutate()}>
                {importMutation.isPending ? "导入并重建索引中..." : "批量导入规则知识"}
              </button>
            </section>
          ) : (
            <ConnectorSetupPanel
              channel={selectedChannel}
              sourceName={sourceName}
              sourceId={sourceId}
              sourceSystem={sourceSystem}
              saving={connectorMutation.isPending}
              savedMessage={connectorSaved}
              onSourceNameChange={setSourceName}
              onSourceIdChange={setSourceId}
              onSourceSystemChange={setSourceSystem}
              onSave={() => connectorMutation.mutate()}
            />
          )}

          {selectedChannel === "file" ? (
            <FieldMappingTable mode="rules" parseState={parseState} />
          ) : (
            <RealtimeConnectionGuide channel={selectedChannel} />
          )}
        </div>

        <div className="side-stack">
          <KnowledgeIndexStatusPanel status={knowledgeStatus.data?.data} />
          <EnterpriseDataTypes />
          {selectedChannel === "file" && (
            <>
              <FieldCheck mode="rules" parseState={parseState} requiredFields={requiredFields} />
              <QualityGate parseState={parseState} />
              <section className="workspace">
                <div className="section-title">
                  <div>
                    <h2>规则文档预览</h2>
                    <p>上传后先校验文件，再导入 RAG 知识库。</p>
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
                  <div className="empty-mini">还没有选择规则文档。</div>
                )}
              </section>
            </>
          )}
          {selectedChannel !== "file" && (
            <section className="workspace">
              <div className="section-title">
                <div>
                  <h2>订单/库存接入原则</h2>
                  <p>实时数据不走人工文件导入，避免履约判断使用过期快照。</p>
                </div>
              </div>
              <div className="data-readiness">
                <strong>建议使用 {selectedChannel === "api" ? "API / Webhook" : selectedChannel === "database" ? "数据库视图 / CDC" : "SFTP 补偿通道"}</strong>
                <span>文件入口仅保留给 RAG 规则文档、SOP、政策和少量规则台账。</span>
              </div>
            </section>
          )}
        </div>
      </div>

      {result && <div className="success-box">导入完成：成功 {result.imported_count} 条。</div>}

      <section className="workspace">
        <div className="section-title">
          <div>
            <h2>已接入数据源</h2>
            <p>轻量展示真实来源，避免复杂维护操作干扰主流程。</p>
          </div>
        </div>
        <div className="source-compact-list">
          {sourceRows.slice(0, 5).map((source) => (
            <article key={source.source_id}>
              <strong>{source.name}</strong>
              <span>{source.source_type} · {source.source_id}</span>
              <small>订单 {source.order_count ?? 0} · 库存 {source.inventory_record_count ?? 0}</small>
              <StatusPill tone={source.enabled ? "ok" : "warn"}>{source.enabled ? "正常" : "停用"}</StatusPill>
            </article>
          ))}
          {sourceRows.length === 0 && (
            <div className="empty-mini">还没有真实数据源，上传订单或库存后会显示在这里。</div>
          )}
        </div>
      </section>

      <section className="workspace">
        <div className="section-title">
          <div>
            <h2>最近接入</h2>
            <p>展示 MySQL 中真实数据源的更新时间，便于运营人员追踪数据来源。</p>
          </div>
        </div>
        <div className="history-compact">
          {sourceRows.slice(0, 3).map((item) => (
            <article key={item.source_id}>
              <CheckCircle2 size={18} />
              <div>
                <strong>{item.name}</strong>
                <span>{item.source_type} · 订单 {item.order_count ?? 0} / 库存 {item.inventory_record_count ?? 0} · {item.updated_at || "未同步"}</span>
              </div>
              <StatusPill tone={item.enabled ? "ok" : "warn"}>{item.enabled ? "启用" : "停用"}</StatusPill>
            </article>
          ))}
          {sourceRows.length === 0 && (
            <div className="empty-mini">暂无真实接入记录，保存数据源或导入文件后会显示在这里。</div>
          )}
        </div>
      </section>
    </div>
  );
}
