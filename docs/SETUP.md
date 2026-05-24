# 启动与配置

## 1. 安装依赖

项目使用 `uv` 管理依赖。

```powershell
cd E:\multiship-agent
uv sync
```

## 2. 启动后端

推荐使用热启动命令，代码改动后自动重载：

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

后端默认 API 前缀：

```text
http://localhost:8000/api/v1
```

健康检查：

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

## 3. 启动前端

```powershell
cd frontend
npm install
npm run dev
```

前端默认地址：

```text
http://localhost:5173
```

开发环境会通过 Vite proxy 把 `/api` 转发到 FastAPI。后端端口不是 8000 时，在 `frontend/.env.local` 中配置 `VITE_API_TARGET`。

## 4. 模型环境变量

`.env` 中放密钥和运行时开关，`config/model_gateway.yaml` 中放模型能力、用途和路由。

最小可用配置示例：

```env
LLM_PROVIDER=openai_compatible
LLM_MODEL=deepseek-v4-pro
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-你的DeepSeekKey

DASHSCOPE_API_KEY=sk-你的阿里云百炼Key

VECTOR_STORE_TYPE=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=knowledge_base_1024
MILVUS_DIM=1024
VECTOR_STORE_FALLBACK_TO_LOCAL=true

LONG_TERM_MEMORY_VECTOR_DIMENSION=1024
```

注意：`.env` 修改后需要重启后端，运行中的进程不会自动重新读取旧环境变量。

## 5. 启动 Milvus

RAG 生产级向量库使用 Milvus。

```powershell
docker compose -f docker-compose.milvus.yml up -d
```

如果 Milvus 没启动，项目会在 `VECTOR_STORE_FALLBACK_TO_LOCAL=true` 时降级到本地索引，方便开发演示。

## 6. 启动 MySQL + Milvus 长期记忆

长期记忆生产推荐使用 MySQL + Milvus；当前 `docker-compose.milvus.yml` 已包含 MySQL 8.4 和 Milvus standalone。

```powershell
docker compose -f docker-compose.milvus.yml up -d
```

然后在 `.env` 中设置：

```env
LONG_TERM_MEMORY_BACKEND=mysql_milvus
LONG_TERM_MEMORY_MYSQL_URL=mysql+pymysql://root:root@localhost:3306/multiship_agent
LONG_TERM_MEMORY_MILVUS_COLLECTION=long_term_memory_vectors_1024
LONG_TERM_MEMORY_VECTOR_DIMENSION=1024
```

## 7. 常见问题

### 后端 Ctrl+C 没反应

`uvicorn --reload` 会有父子两个进程。Windows 下偶尔 Ctrl+C 只停掉其中一个。可以新开 PowerShell 查找端口：

```powershell
netstat -ano | findstr :8000
```

然后结束对应 PID：

```powershell
Stop-Process -Id <PID> -Force
```

### Embedding 维度报错

如果看到类似：

```text
shapes (512,) and (256,), not aligned
```

通常是旧索引里存的是旧维度，新模型生成的是新维度。当前阿里云 `text-embedding-v4` 配置为 1024 维，需要：

- `MILVUS_DIM=1024`
- `MILVUS_COLLECTION=knowledge_base_1024`
- `LONG_TERM_MEMORY_VECTOR_DIMENSION=1024`

如果切换模型维度，建议新建 collection，不要混用旧向量。

### 后端显示缺少 API Key

先确认 `.env` 已保存，然后重启后端。可以用下面命令验证配置是否被读取：

```powershell
uv run python -c "from app.core.config import get_settings; s=get_settings(); print(bool(s.dashscope_api_key))"
```
