export type ApiResponse<T> = {
  success: boolean;
  message: string;
  data: T;
};

export type HealthPayload = {
  status: string;
  app_name: string;
  version: string;
};

export type InterruptEvent = {
  type: string;
  node: string;
  prompt: string;
  context: Record<string, unknown>;
  options: string[];
  thread_id: string;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | string;
  risk_signals: string[];
  timeout_seconds: number;
};

export type BusinessTraceStep = {
  id: string;
  parent_id?: string | null;
  type: string;
  name: string;
  status: "success" | "error" | "skipped" | "interrupted" | "pending_human" | string;
  started_at?: string;
  ended_at?: string | null;
  duration_ms?: number | null;
  summary?: string;
  error_code?: string | null;
  error_message?: string | null;
  input_summary?: string | null;
  output_summary?: string | null;
  metadata?: Record<string, unknown>;
  evidence?: Array<Record<string, unknown>>;
};

export type AuditEvent = {
  id: string;
  trace_id?: string | null;
  order_id?: string | null;
  event_type: string;
  action?: string | null;
  actor_type?: string | null;
  status?: string | null;
  created_at?: string | null;
  summary?: string | null;
  metadata?: Record<string, unknown>;
};

export type BusinessTrace = {
  trace_id: string;
  request_id?: string;
  session_id?: string | null;
  tenant_id?: string | null;
  user_id?: string | null;
  order_id?: string | null;
  route?: string | null;
  status?: string;
  started_at?: string;
  ended_at?: string | null;
  duration_ms?: number | null;
  error_code?: string | null;
  error_message?: string | null;
  metadata?: Record<string, unknown>;
  steps?: BusinessTraceStep[];
  audit_events?: AuditEvent[];
  step_count?: number | null;
};

export type HybridRunResult = {
  order_id: string;
  question?: string | null;
  final_answer?: string | null;
  path_used?: string;
  intent_level?: string;
  confidence?: number;
  tools_called?: string[];
  execution_time_ms?: number;
  thread_id?: string;
  conversation_turns?: number;
  status: "completed" | "interrupted" | "error" | string;
  interrupt?: InterruptEvent | null;
  from_cache?: boolean;
  business_trace?: BusinessTrace | null;
};

export type HybridStats = {
  simple_threshold?: number;
  complex_threshold?: number;
  available_paths?: string[];
  routing_counts?: Record<string, number>;
};

export type EnterpriseStats = {
  source_count?: number;
  order_count?: number;
  inventory_record_count?: number;
  inventory_sku_count?: number;
  last_updated_at?: string | null;
};

export type OrderItem = {
  sku_id: string;
  product_name: string;
  quantity: number;
  unit_price: number;
};

export type OrderRecord = {
  order_id: string;
  platform: string;
  order_time: string;
  order_status: string;
  region: string;
  priority: string;
  items: OrderItem[];
};

export type ApprovalAuditEntry = {
  thread_id: string;
  order_id: string;
  interrupt_type: string;
  risk_level: string;
  risk_signals: string[];
  decision: string;
  reason: string;
  approver_id: string;
  requested_at: string;
  decided_at: string;
};

export type HitlStats = {
  pending_count?: number;
  approved_count?: number;
  rejected_count?: number;
  escalated_count?: number;
  total_count?: number;
};

export type InventoryRecord = {
  warehouse_id: string;
  warehouse_name: string;
  region: string;
  sku_id: string;
  available_stock: number;
  locked_stock: number;
  updated_at: string;
};

export type EnterpriseDataSource = {
  source_id: string;
  name: string;
  source_type: "manual" | "json" | "api" | "database" | "sftp" | string;
  description: string;
  config: Record<string, unknown>;
  enabled: boolean;
  created_at?: string;
  updated_at?: string;
  order_count?: number;
  inventory_record_count?: number;
};

export type EnterpriseImportResult = {
  source_id: string;
  imported_count: number;
  total_orders: number;
  total_inventory_records: number;
};

export type KnowledgeUploadResult = {
  document_id: string;
  filename: string;
  replaced?: boolean;
  ingestion_id?: string;
  status?: string;
  version?: string;
  quality_score?: number;
  quality_passed?: boolean;
  expired?: boolean;
  warnings?: string[];
  errors?: string[];
  rebuild_scheduled: boolean;
};

export type KnowledgeIndexStatus = {
  document_count: number;
  indexed_document_count: number;
  unindexed_document_count: number;
  total_chunk_count: number;
  embedding_ready: boolean;
  vector_store_type: string;
  index_built: boolean;
  rebuild_running: boolean;
  last_rebuild_result?: string | null;
  last_rebuild_error?: string | null;
  pending_ingestion_count: number;
  knowledge_dir: string;
  knowledge_extra_dirs?: string;
  index_cache_dir: string;
  document_registry?: Record<string, unknown>;
};

const API_BASE = import.meta.env.VITE_API_BASE || "/api/v1";
const REQUEST_TIMEOUT_MS = 90_000;

function makeTraceId() {
  return globalThis.crypto?.randomUUID?.() || `trace-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<ApiResponse<T>> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const traceId = makeTraceId();

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: {
        "Content-Type": "application/json",
        "X-Trace-Id": traceId,
        "X-Request-Started-At": new Date().toISOString(),
        ...(init?.headers || {}),
      },
      ...init,
      signal: init?.signal || controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("请求超时：后端智能分析仍未返回，请检查模型服务、网络或稍后重试。");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail || body?.message || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body as ApiResponse<T>;
}

export function getHealth() {
  return request<HealthPayload>("/health");
}

export function getHybridStats() {
  return request<HybridStats>("/hybrid/stats");
}

export function getEnterpriseStats() {
  return request<EnterpriseStats>("/enterprise-data/stats");
}

export function listEnterpriseSources() {
  return request<{ sources: EnterpriseDataSource[] }>("/enterprise-data/sources");
}

export function listEnterpriseOrders(limit = 50) {
  const params = new URLSearchParams({ limit: String(limit) });
  return request<{ orders: OrderRecord[] }>(`/enterprise-data/orders?${params.toString()}`);
}

export function listPendingApprovals(riskLevel?: string) {
  const params = new URLSearchParams();
  if (riskLevel) params.set("risk_level", riskLevel);
  const query = params.toString();
  return request<{ items: ApprovalAuditEntry[] }>(`/workflow/approvals/pending${query ? `?${query}` : ""}`);
}

export function getHitlStats() {
  return request<HitlStats>("/workflow/approvals/stats");
}

export function upsertEnterpriseSource(input: {
  sourceId?: string;
  name: string;
  sourceType: EnterpriseDataSource["source_type"];
  description: string;
  enabled?: boolean;
}) {
  return request<{ source: EnterpriseDataSource }>("/enterprise-data/sources", {
    method: "POST",
    body: JSON.stringify({
      source_id: input.sourceId || undefined,
      name: input.name,
      source_type: input.sourceType,
      description: input.description,
      config: { ingestion_channel: "frontend-upload" },
      enabled: input.enabled ?? true,
    }),
  });
}

export function importEnterpriseOrders(input: {
  sourceId: string;
  orders: OrderRecord[];
  replaceSource: boolean;
}) {
  return request<EnterpriseImportResult>("/enterprise-data/orders/import", {
    method: "POST",
    body: JSON.stringify({
      source_id: input.sourceId,
      orders: input.orders,
      replace_source: input.replaceSource,
    }),
  });
}

export function importEnterpriseInventory(input: {
  sourceId: string;
  records: InventoryRecord[];
  replaceSource: boolean;
}) {
  return request<EnterpriseImportResult>("/enterprise-data/inventory/import", {
    method: "POST",
    body: JSON.stringify({
      source_id: input.sourceId,
      records: input.records,
      replace_source: input.replaceSource,
    }),
  });
}

export async function uploadKnowledgeDocument(input: {
  documentId: string;
  file: File | Blob;
  fileName: string;
  replaceExisting: boolean;
  rebuild: boolean;
  confirmBeforeIndex?: boolean;
  category?: string;
  title?: string;
}) {
  const formData = new FormData();
  formData.append("file", input.file, input.fileName);
  formData.append("document_id", input.documentId);
  formData.append("replace_existing", String(input.replaceExisting));
  formData.append("rebuild", String(input.rebuild));
  if (input.confirmBeforeIndex !== undefined) {
    formData.append("confirm_before_index", String(input.confirmBeforeIndex));
  }
  if (input.category) formData.append("category", input.category);
  if (input.title) formData.append("title", input.title);

  const response = await fetch(`${API_BASE}/knowledge-mgmt/documents`, {
    method: "POST",
    body: formData,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail || body?.message || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body as ApiResponse<KnowledgeUploadResult>;
}

export function getKnowledgeIndexStatus() {
  return request<KnowledgeIndexStatus>("/knowledge-mgmt/status");
}

export async function runHybrid(input: {
  orderId?: string;
  question: string;
  threadId?: string;
  clientInputAt?: string;
}) {
  const params = new URLSearchParams({ question: input.question });
  if (input.orderId) {
    params.set("order_id", input.orderId);
  }
  if (input.threadId) {
    params.set("thread_id", input.threadId);
  }
  return request<HybridRunResult>(`/hybrid/run?${params.toString()}`, {
    method: "POST",
    headers: input.clientInputAt ? { "X-Client-Input-At": input.clientInputAt } : undefined,
  });
}

export async function resumeHybrid(input: {
  threadId: string;
  decision: "approved" | "rejected";
  notes: string;
}) {
  const params = new URLSearchParams({
    thread_id: input.threadId,
    decision: input.decision,
    notes: input.notes,
  });
  return request<HybridRunResult>(`/hybrid/resume?${params.toString()}`, { method: "POST" });
}

export function listBusinessTraces(input?: {
  orderId?: string;
  status?: string;
  toolName?: string;
  route?: string;
  stepType?: string;
  modelId?: string;
  promptId?: string;
  minDurationMs?: number;
  startedFrom?: string;
  startedTo?: string;
  error?: string;
  limit?: number;
}) {
  const params = new URLSearchParams();
  if (input?.orderId) params.set("order_id", input.orderId);
  if (input?.status) params.set("status", input.status);
  if (input?.toolName) params.set("tool_name", input.toolName);
  if (input?.route) params.set("route", input.route);
  if (input?.stepType) params.set("step_type", input.stepType);
  if (input?.modelId) params.set("model_id", input.modelId);
  if (input?.promptId) params.set("prompt_id", input.promptId);
  if (input?.minDurationMs !== undefined) params.set("min_duration_ms", String(input.minDurationMs));
  if (input?.startedFrom) params.set("started_from", input.startedFrom);
  if (input?.startedTo) params.set("started_to", input.startedTo);
  if (input?.error) params.set("error", input.error);
  if (input?.limit) params.set("limit", String(input.limit));
  const query = params.toString();
  return request<{ traces: BusinessTrace[] }>(`/observability/traces${query ? `?${query}` : ""}`);
}

export function getBusinessTrace(traceId: string) {
  return request<BusinessTrace>(`/observability/traces/${encodeURIComponent(traceId)}`);
}
