# FulfillOps-Agent

电商履约运营智能协同 Agent，用来处理异常订单分析、履约规划、人工审核、外部协同任务、结果校验和案例沉淀。

![FulfillOps-Agent 架构图](docs/images/architecture.png)

## 项目简介

FulfillOps-Agent 面向电商多仓、多系统协同履约场景。系统会读取 OMS、WMS、TMS、ERP、PIM、CRM 等业务数据，结合 SOP 规则库和历史优秀案例，生成可审核、可追踪、可重规划的履约处理方案。

这个项目的核心不是让 AI 直接改订单、改库存或改物流数据，而是让 Agent 做分析、规划和协同任务创建；真实业务动作仍由原业务系统和人工流程完成。

当前基础设施已统一为 PostgreSQL：业务数据、短期会话状态、工具缓存、接口限流、Workflow 幂等、HITL、Trace、长期记忆元数据都写入 PostgreSQL；RAG 和长期记忆的语义向量使用同库 PGVector。

## 页面预览

异常订单处理页：

![异常订单处理页面](docs/images/ops-page-empty.png)

企业数据接入页：

![企业数据接入页面](docs/images/data-ingestion-page.png)

## 核心能力

- 异常订单分析：基于订单、库存、物流、商品限制和客户上下文判断履约风险。
- 短期记忆：保留用户明确表达的偏好、约束、方案反馈和指代关系。
- 最新业务上下文：每次规划和校验都重新读取业务系统，不复用旧实时数据。
- RAG 检索：同时检索当前 SOP 规则和历史优秀案例。
- Planner / Replanner：生成履约方案、Action DAG、Success Criteria 和证据链。
- HITL 人工审核：高风险或写入协同动作必须经过人工审批。
- 协同写入：只创建 WMS/TMS/ERP/CRM 的外部任务或申请，不直接修改核心业务数据。
- Workflow 状态机：支持 WAITING、Webhook/Status API 恢复、VERIFYING、REPLAN、MANUAL。
- Trace / Governance：记录模型、工具、RAG、审批、回调、校验、成本和异常。
- Agent Evaluation：用基准场景回归评估工具调用、事实准确性、RAG 依据和闭环流程。

## 简版业务流程

1. 用户提交异常订单或履约问题。
2. 系统抽取短期记忆，只保存偏好、约束和方案反馈，不保存旧业务事实。
3. 构建最新 OrderContext，从 OMS/WMS/TMS/ERP/PIM/CRM 读取实时数据。
4. 通过 Model Gateway 做意图识别和 Query Rewrite。
5. RAG 分别检索 SOP Knowledge Base 和 Case Library。
6. Planner Agent 基于当前业务数据、SOP 证据和历史案例生成方案。
7. 如果上下文不足，调用 6 个只读 Tool 渐进补齐数据。
8. Policy / Schema Validator 校验方案、Action DAG、动作参数和业务硬约束。
9. 进入 HITL，人工可以批准、拒绝或修改方案。
10. 批准后，Action Router 把动作转成外部协同任务。
11. Tool Gateway 创建 WMS/TMS/ERP/CRM 任务或申请，并记录幂等、权限、重试和回执。
12. Workflow 进入 WAITING，保存 Case 状态和 Checkpoint，Agent 释放执行资源。
13. 外部人员在原系统完成真实业务动作，通过 Webhook 或 Status API 回调。
14. 系统恢复 Workflow，进入 VERIFYING，重新读取最新业务数据并校验 Success Criteria。
15. 校验通过则 COMPLETED / CLOSED。
16. 校验失败则进入 REPLAN，最多重规划 3 次。
17. 多次失败后进入 MANUAL，生成包含最新业务状态、失败原因、已完成动作和未完成任务的人工接管包。
18. Case 关闭后，运营可主动沉淀为优秀案例。
19. 优秀案例写入 Case Library，后续相似异常可被召回。
20. 全链路 Trace 用于审计、排障、成本统计和版本回归评测。

决策优先级始终是：

```text
当前真实业务数据 > 当前有效 SOP > 历史优秀案例
```

历史案例只作为经验参考；如果和当前业务数据或 SOP 冲突，系统会忽略该案例。

## 技术栈

后端：

- FastAPI
- LangGraph
- LangChain
- LlamaIndex
- SQLAlchemy
- PostgreSQL
- PGVector

前端：

- React
- TypeScript
- Vite
- TanStack Query
- Zustand
- lucide-react

工程能力：

- Model Gateway
- Tool Gateway
- RAG 文档入库与切分
- Business Trace / Audit
- Prompt Injection 防护
- Agent Benchmark / Evaluation

## 目录结构

```text
app/
  api/                 FastAPI 接口
  application/          Workflow、HITL、异常案件、路由服务
  agents/               Agent 运行时、工具、MCP、评测
  domain/               订单、库存、履约上下文等领域逻辑
  infrastructure/       模型网关和基础设施适配
  memory/               短期记忆和长期记忆
  observability/        Trace、审计和指标
  rag/                  文档解析、索引、检索和重排
  schemas/              Pydantic 数据契约
  workflows/            LangGraph 履约工作流

frontend/
  src/                  前端页面、状态和 API 封装

config/
  model_gateway.yaml    模型路由、Prompt 和 fallback 配置

prompts/
  extraction/           记忆抽取、案例抽取 Prompt
  rag/                  Query Rewrite Prompt
  router/               意图识别 Prompt
  workflow/             Planner / Replanner Prompt

docs/images/            README 架构图和页面截图
```

## 部署方式

### 方式一：Docker Compose 部署

适合服务器或完整联调环境。Compose 会启动应用和带 PGVector 扩展的 PostgreSQL。

1. 复制环境变量：

```bash
cp .env.docker.example .env.docker
```

2. 修改 `.env.docker`，至少配置：

```env
APP_PORT=8000
POSTGRES_USER=fulfillops
POSTGRES_PASSWORD=change_this_postgres_password
POSTGRES_DB=fulfillops_agent
DATABASE_URL=postgresql+psycopg://fulfillops:change_this_postgres_password@postgres:5432/fulfillops_agent
VECTOR_STORE_TYPE=pgvector
PGVECTOR_HOST=postgres
PGVECTOR_TABLE=knowledge_base_vectors
MODEL_GATEWAY_CONFIG_PATH=config/model_gateway.yaml
LLM_API_KEY=sk-xxxxxxxx
```

3. 启动：

```bash
docker compose --env-file .env.docker up -d --build
```

4. 查看状态：

```bash
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker logs -f app
```

5. 访问：

```text
http://localhost:8000
http://localhost:8000/docs
```

停止：

```bash
docker compose --env-file .env.docker down
```

清空数据卷：

```bash
docker compose --env-file .env.docker down -v
```

### 方式二：本地源码运行

适合开发调试。需要本机准备 PostgreSQL，并启用 PGVector 扩展；短期记忆、工具缓存、限流、幂等、Trace 和长期记忆都写入 PostgreSQL。

1. 安装后端依赖：

```bash
uv sync
```

2. 安装前端依赖：

```bash
cd frontend
npm install
cd ..
```

3. 复制环境变量：

```bash
cp .env.example .env
```

常用本地配置：

```env
DATABASE_URL=postgresql+psycopg://fulfillops:fulfillops@localhost:5432/fulfillops_agent
POSTGRES_URL=postgresql+psycopg://fulfillops:fulfillops@localhost:5432/fulfillops_agent
VECTOR_STORE_TYPE=pgvector
PGVECTOR_HOST=localhost
PGVECTOR_TABLE=knowledge_base_vectors
MODEL_GATEWAY_CONFIG_PATH=config/model_gateway.yaml
LLM_API_KEY=sk-xxxxxxxx
```

4. 启动后端：

```bash
uv run uvicorn app.main:app --reload --port 8000
```

5. 启动前端：

```bash
cd frontend
npm run dev
```

默认访问：

```text
http://localhost:5173
```

### 方式三：Windows 一键脚本

项目内置了 Windows 启停脚本：

```cmd
start_windows.cmd
```

停止：

```cmd
stop_windows.cmd
```

## 数据接入

项目启动后，需要先接入业务数据和规则文档：

- 订单和库存：通过企业 API、Webhook、数据库 CDC 或补偿文件接入。
- SOP / 规则文档：支持 Markdown、TXT、PDF、DOCX、PPTX 入库。
- 历史优秀案例：由已关闭 Case 主动沉淀后进入 Case Library。

数据接入后，异常订单处理页才能展示待处理订单、AI 分析结果和外部任务状态。

## 常用地址

| 用途 | 地址 |
| --- | --- |
| 前端页面 | `http://localhost:5173` |
| Docker 应用入口 | `http://localhost:8000` |
| OpenAPI 文档 | `http://localhost:8000/docs` |
| 健康检查 | `/api/v1/health` |
| 指标 | `/metrics` |
| Trace Center | `/api/v1/observability/traces` |
| 待审批任务 | `/api/v1/workflow/approvals/pending` |

## 测试与检查

后端测试：

```bash
uv run pytest
```

后端编译检查：

```bash
python -m compileall app tests
```

前端类型检查和构建：

```bash
cd frontend
npm run build
```

前端运营台合同测试：

```bash
cd frontend
npm run test:ops-contract
```

## 说明

本项目强调可审计和可控的企业履约协同：Agent 负责分析、规划、检索、生成方案和创建协同任务；订单、库存、物流、采购和客服核心记录仍由原业务系统和人工流程完成。
