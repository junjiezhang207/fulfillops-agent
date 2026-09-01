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

export type ProposalAction = {
  action_id: string;
  action_type: string;
  sku_id?: string | null;
  quantity: number;
  from_warehouse?: string | null;
  to_warehouse?: string | null;
  carrier?: string | null;
  cost_delta: number;
  eta_hours: number;
  reason: string;
  reversible: boolean;
  depends_on?: string[];
  responsibility_domain?: string | null;
  business_evidence?: string[];
};

export type ExecutionProposal = {
  proposal_id: string;
  order_id: string;
  title: string;
  summary: string;
  status: string;
  capabilities: string[];
  actions: ProposalAction[];
  decision_context: Record<string, any>;
  inventory_snapshot: Array<Record<string, unknown>>;
  cost_breakdown: Record<string, unknown>;
  eta: Record<string, unknown>;
  rule_citations: string[];
  data_fingerprint: string;
  freshness: Record<string, unknown>;
  approval_required: boolean;
  preflight_checks: string[];
  invalidation_reason?: string | null;
  goal_type?: string;
  action_dag?: Record<string, unknown>;
  success_criteria?: Record<string, unknown>;
  plan_version?: number;
  context_version?: string;
  expires_at?: string | null;
};

export type PreflightValidation = {
  status: string;
  checked_at: string;
  checks: Array<Record<string, unknown>>;
  old_fingerprint?: string | null;
  new_fingerprint?: string | null;
  message: string;
  replacement_proposal?: ExecutionProposal | null;
};

export type RecentMessage = {
  role: string;
  content: string;
  created_at: string;
};

export type StructuredSessionMemory = {
  current_topic?: string;
  user_preferences?: Record<string, unknown>;
  confirmed_constraints?: Record<string, unknown>;
  plan_feedback?: Record<string, unknown>;
  references?: Record<string, unknown>;
};

export type SessionMemorySnapshot = {
  thread_id: string;
  order_id?: string;
  recent_messages?: RecentMessage[];
  structured?: StructuredSessionMemory;
  memory_use_case?: string;
  token_budget?: number;
  updated_at?: string | null;
};

export type RAGEvidenceItem = {
  source_type: string;
  source_file?: string;
  category?: string;
  title?: string;
  chunk_id?: string;
  source_case_id?: string;
  score?: number;
  text_excerpt?: string;
};

export type RAGSourceSet = {
  query?: string;
  applied_filters?: string[];
  key_points?: string[];
  coverage_note?: string;
  evidence?: RAGEvidenceItem[];
};

export type PlannerRAGContext = {
  order_id: string;
  question: string;
  vector_backend?: string;
  retrieval_strategy?: string;
  priority_rule?: string;
  sop_evidence?: RAGSourceSet;
  similar_cases?: RAGSourceSet;
  warnings?: string[];
};

export type RoutedExternalTask = {
  task_id: string;
  case_id: string;
  proposal_id: string;
  action_id: string;
  action_type: string;
  target_system: string;
  domain_service: string;
  collaborative_tool_name?: string;
  external_task_type: string;
  payload: Record<string, unknown>;
  status: string;
  case_version?: number;
  plan_version?: number;
  dependency_ids?: string[];
  idempotency_key?: string;
  created_at: string;
  updated_at: string;
  result: Record<string, unknown>;
};

export type FulfillmentCase = {
  case_id: string;
  order_id: string;
  proposal_id: string;
  case_status: string;
  case_version?: number;
  plan: ExecutionProposal;
  plan_version?: number;
  context_version?: string;
  action_dag?: Record<string, unknown>;
  success_criteria?: Record<string, unknown>;
  replan_count?: number;
  checkpoint: Record<string, unknown>;
  current_state?: Record<string, unknown>;
  tasks: RoutedExternalTask[];
  created_at: string;
  updated_at: string;
  verification: Record<string, unknown>;
};

export type FulfillmentCaseVerifyResult = {
  case_id: string;
  order_id: string;
  status: string;
  message: string;
  checks: Array<Record<string, unknown>>;
  replan_required: boolean;
  replacement_proposal?: ExecutionProposal | null;
};

export type ExcellentCaseRecord = {
  excellent_case_id: string;
  case_id: string;
  order_id_masked: string;
  scenario: string;
  key_conditions: string[];
  final_plan: Array<Record<string, unknown>>;
  sop_evidence: string[];
  execution_result: Record<string, unknown>;
  vector_backend: string;
  created_at: string;
};

export type CaseIngestionJob = {
  job_id: string;
  case_id: string;
  status: "PENDING" | "PROCESSING" | "SUCCESS" | "FAILED" | string;
  retry_count: number;
  operator_id: string;
  notes: string;
  excellent_case_id?: string | null;
  knowledge_path?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
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
  action_card?: ExecutionProposal | null;
  preflight_validation?: PreflightValidation | null;
  session_memory?: SessionMemorySnapshot | null;
  rag_context?: PlannerRAGContext | null;
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
  allocated_quantity?: number;
  shipped_quantity?: number;
  sku_status?: string;
  split_allowed?: boolean;
  special_storage?: string | null;
  sku_type?: string | null;
};

export type OrderRecord = {
  order_id: string;
  platform: string;
  order_time: string;
  order_status: string;
  region: string;
  priority: string;
  items: OrderItem[];
  created_at?: string | null;
  promise_delivery_time?: string | null;
  current_warehouse_id?: string | null;
  shipping_region?: string | null;
  fulfillment_type?: string;
  already_split?: boolean;
  inventory_reserved?: boolean;
  package_created?: boolean;
  waybill_created?: boolean;
  outbound_completed?: boolean;
  active_fulfillment_tasks?: string[];
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
  on_hand_stock?: number | null;
  reserved_stock?: number;
  inbound_stock?: number;
  expected_inbound_time?: string | null;
  inventory_version?: string | null;
  warehouse_status?: string;
  service_region?: string | null;
  supported_sku_types?: string[];
  cutoff_time?: string | null;
  capacity_status?: string;
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
  decision: "approved" | "rejected" | "modify" | "ask_followup";
  notes: string;
}) {
  const params = new URLSearchParams({
    thread_id: input.threadId,
    decision: input.decision,
    notes: input.notes,
  });
  return request<HybridRunResult>(`/hybrid/resume?${params.toString()}`, { method: "POST" });
}

export function confirmFulfillmentCase(input: {
  proposal: ExecutionProposal;
  preflightValidation: PreflightValidation;
  approverId?: string;
  notes?: string;
  checkpoint?: Record<string, unknown>;
}) {
  return request<{ case: FulfillmentCase }>("/fulfillment-cases/confirm", {
    method: "POST",
    body: JSON.stringify({
      proposal: input.proposal,
      preflight_validation: input.preflightValidation,
      approver_id: input.approverId || "operator",
      notes: input.notes || "",
      checkpoint: input.checkpoint || {},
    }),
  });
}

export function listFulfillmentCases(input?: { orderId?: string; limit?: number }) {
  const params = new URLSearchParams();
  if (input?.orderId) params.set("order_id", input.orderId);
  if (input?.limit) params.set("limit", String(input.limit));
  const query = params.toString();
  return request<{ cases: FulfillmentCase[] }>(`/fulfillment-cases${query ? `?${query}` : ""}`);
}

export function verifyFulfillmentCase(caseId: string) {
  return request<{ verification: FulfillmentCaseVerifyResult }>(`/fulfillment-cases/${encodeURIComponent(caseId)}/verify`, {
    method: "POST",
  });
}

export function updateFulfillmentTask(input: {
  taskId: string;
  status: "RUNNING" | "COMPLETED" | "FAILED" | "REJECTED";
  message?: string;
  result?: Record<string, unknown>;
  eventId?: string;
  caseVersion?: number;
  planVersion?: number;
  idempotencyKey?: string;
}) {
  return request<{ case: FulfillmentCase }>(`/fulfillment-cases/tasks/${encodeURIComponent(input.taskId)}/webhook`, {
    method: "POST",
    body: JSON.stringify({
      status: input.status,
      event_id: input.eventId,
      case_version: input.caseVersion,
      plan_version: input.planVersion,
      idempotency_key: input.idempotencyKey,
      message: input.message || "",
      result: input.result || {
        simulated_from: "operations_dashboard",
        simulated_at: new Date().toISOString(),
      },
    }),
  });
}

export function persistExcellentCase(input: { caseId: string; operatorId?: string; notes?: string }) {
  return request<{ ingestion_job: CaseIngestionJob; rebuild_scheduled: boolean }>(
    `/fulfillment-cases/${encodeURIComponent(input.caseId)}/excellent-case`,
    {
      method: "POST",
      body: JSON.stringify({
        operator_id: input.operatorId || "operator",
        notes: input.notes || "",
      }),
    },
  );
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
