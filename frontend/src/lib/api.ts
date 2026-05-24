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

const API_BASE = import.meta.env.VITE_API_BASE || "/api/v1";
const REQUEST_TIMEOUT_MS = 90_000;

async function request<T>(path: string, init?: RequestInit): Promise<ApiResponse<T>> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: {
        "Content-Type": "application/json",
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

export async function runHybrid(input: {
  orderId: string;
  question: string;
  threadId?: string;
}) {
  const params = new URLSearchParams({
    order_id: input.orderId,
    question: input.question,
  });
  if (input.threadId) {
    params.set("thread_id", input.threadId);
  }
  return request<HybridRunResult>(`/hybrid/run?${params.toString()}`, { method: "POST" });
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
