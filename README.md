# Multiship Agent

这是一个电商多仓履约场景的 AI 项目。

用户可以在前端输入订单履约问题，后端会结合订单、库存、规则知识库和 AI Agent，给出履约建议。如果订单风险较高，系统会暂停自动流程，进入人工审核。

简单说，这个项目做的是：

```text
订单 + 库存 + 业务规则 + AI Agent -> 履约判断 / 缺货处理 / 人工审核 / 决策链路追踪
```

## 主要功能

- 智能履约对话：输入订单问题，返回是否可履约、风险原因和处理建议。
- 企业数据接入：导入订单、库存和规则文档。
- RAG 规则检索：从履约规则、SOP、售后政策中检索依据。
- Agent 工具调用：动态查询订单、库存、知识库、替代 SKU、履约方案。
- Multi-Agent 分析：库存、履约、风险等多个专家协作。
- 人工审核 HITL：高风险订单会中断流程，等待人工通过或暂停。
- Trace Center：查看一次请求经过了哪些步骤、调用了哪些工具、用了哪些规则依据。

## 现在不是 mock 项目

这个版本已经去掉了主要的演示降级：

- Redis 不可用时不再降级到内存。
- Milvus 不可用时不再降级到本地向量库。
- 订单和库存不再使用前端 mock 或后端 InMemory 数据。
- HITL 审核状态写入 MySQL。
- Business Trace 和审计日志写入 MySQL。

所以你第一次启动后，需要先导入订单、库存和规则文档，系统才有真实数据可分析。

## 技术栈

后端：

- FastAPI
- LangGraph
- LangChain
- LlamaIndex
- SQLAlchemy
- Redis
- MySQL
- Milvus

前端：

- React
- TypeScript
- Vite
- TanStack Query
- Zustand

Docker 部署：

- app
- MySQL
- Redis
- Milvus
- etcd
- MinIO

说明：项目代码直接连接的是 MySQL、Redis、Milvus。`etcd` 和 `MinIO` 是 Milvus standalone 的内部依赖，业务代码不直接访问它们。

## 项目结构

```text
app/
  api/                 FastAPI 接口
  application/          应用服务：路由、Workflow、HITL
  agents/               ReAct Agent、Multi-Agent、工具系统
  core/                 配置、日志、限流、启动检查
  domain/               订单、库存、履约等业务逻辑
  memory/               短期记忆和长期记忆
  observability/        Trace 和审计日志
  rag/                  文档解析、索引、检索
  repositories/         MySQL 数据仓储

frontend/
  src/pages/            前端页面
  src/lib/api.ts        API 请求封装

config/
  model_gateway.yaml    模型网关配置
```

## Docker 部署

推荐在 Linux 服务器上用 Docker Compose 部署。

### 1. 准备环境变量

```bash
cp .env.docker.example .env.docker
```

编辑 `.env.docker`，至少修改这些值：

```env
MYSQL_ROOT_PASSWORD=change_this_mysql_password
MYSQL_URL=mysql+pymysql://root:change_this_mysql_password@mysql:3306/multiship_agent
LONG_TERM_MEMORY_MYSQL_URL=mysql+pymysql://root:change_this_mysql_password@mysql:3306/multiship_agent

MINIO_ROOT_PASSWORD=change_this_minio_password

LLM_API_KEY=sk-xxxxxxxx
```

建议 MySQL 密码先使用字母、数字、下划线，避免 URL 编码问题。

### 2. 启动

```bash
docker compose --env-file .env.docker up -d --build
```

首次启动 Milvus 会比较慢，可以看日志：

```bash
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker logs -f app
```

访问：

```text
http://服务器IP:8000
http://服务器IP:8000/docs
```

默认只需要开放 `8000` 端口。MySQL、Redis、Milvus、MinIO 默认只在 Docker 内部网络通信。

### 3. 停止

```bash
docker compose --env-file .env.docker down
```

如果要连数据卷一起清空：

```bash
docker compose --env-file .env.docker down -v
```

## 本地源码运行

本地运行需要自己准备 MySQL、Redis、Milvus。

### 1. 安装后端依赖

```bash
uv sync
```

### 2. 安装前端依赖

```bash
cd frontend
npm install
cd ..
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

重点检查：

```env
MYSQL_URL=mysql+pymysql://root:root@localhost:3306/multiship_agent
REDIS_URL=redis://localhost:6379/0
MILVUS_URI=http://localhost:19530
LLM_API_KEY=sk-xxxxxxxx
```

### 4. 启动后端

```bash
uv run uvicorn app.main:app --reload --port 8000
```

### 5. 启动前端

```bash
cd frontend
npm run dev
```

访问：

```text
http://localhost:5173
```


## 常用接口

| 用途 | 地址 |
| --- | --- |
| 前端页面 | `http://localhost:5173` |
| Docker 部署后的前端页面 | `http://服务器IP:8000` |
| API 健康检查 | `/api/v1/health` |
| OpenAPI 文档 | `/docs` |
| Trace 列表 | `/api/v1/observability/traces` |
| HITL 待审任务 | `/api/v1/workflow/approvals/pending` |

## 数据存在哪里

| 数据 | 存储 |
| --- | --- |
| 订单、库存 | MySQL |
| 人工审核任务 | MySQL |
| Trace 和审计日志 | MySQL |
| 短期会话状态 | Redis |
| 工具缓存和限流 | Redis |
| RAG 向量索引 | Milvus |
| 长期记忆元数据 | MySQL |
| 长期记忆向量 | Milvus |

## 测试

后端语法检查：

```bash
python -m compileall app
```

前端构建：

```bash
cd frontend
npm install
npm run build
```

完整测试：

```bash
uv run pytest
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
