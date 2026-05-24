"""Project configuration center.

Learning notes:
- Settings loads .env values through pydantic-settings.
- Model gateway, Milvus, Redis, memory and Langfuse settings are centralized here.
- Business code should call get_settings() instead of reading environment variables directly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目全局配置。

    设计说明：
    1. 使用 pydantic-settings 统一读取环境变量。
    2. 当前阶段先把项目级配置和基础设施配置边界定义清楚。
    3. 即使 MySQL、Redis、Milvus 还没有真正连接，本阶段也先把配置入口准备好。

    这样做的好处是：
    - 后续新增数据库连接时，不需要再回头重构配置体系。
    - 配置项集中、清晰，便于学习整个项目的基础设施边界。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="全域电商供应链智能履约多Agent平台")
    app_version: str = Field(default="0.1.0")
    api_v1_prefix: str = Field(default="/api/v1")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    frontend_cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated origins allowed to call the API from the browser.",
    )

    mysql_url: str = Field(
        default="mysql+pymysql://root:root@localhost:3306/multiship_agent"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")
    short_term_memory_ttl_seconds: int = Field(
        default=86400,
        description="短期会话记忆 TTL，默认 24 小时。",
    )
    long_term_memory_db_path: str = Field(
        default=str(Path("storage") / "long_term_memory.sqlite3"),
        description="长期记忆 SQLite 落盘路径。",
    )
    long_term_memory_backend: str = Field(
        default="sqlite",
        description="长期记忆后端：sqlite | mysql_milvus。mysql_milvus 使用 MySQL + Milvus。",
    )
    long_term_memory_mysql_url: str = Field(
        default="",
        description="长期记忆 MySQL 连接串；留空时复用 MYSQL_URL。",
    )
    long_term_memory_vector_dimension: int = Field(
        default=512,
        description="长期记忆 Milvus 向量维度，需要和模型网关 embedding 模型输出维度一致。",
    )
    long_term_memory_ttl_days: int | None = Field(
        default=180,
        description="长期记忆默认 TTL 天数；None 表示不过期。",
    )
    long_term_memory_milvus_collection: str = Field(
        default="long_term_memory_vectors",
        description="长期记忆 Milvus collection；与 RAG 知识库 collection 分开，避免索引污染。",
    )
    long_term_memory_milvus_alias: str = Field(
        default="ltm_milvus",
        description="长期记忆 Milvus 连接别名。",
    )
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    knowledge_dir: str = Field(
        default=str(Path("app") / "data" / "knowledge")
    )
    knowledge_index_cache_dir: str = Field(
        default=str(Path("storage") / "knowledge_index")
    )
    enterprise_data_dir: str = Field(
        default=str(Path("storage") / "enterprise_data"),
        description="企业后台导入的订单、库存等结构化业务数据目录。",
    )

    # ------------------------------------------------------------------
    # LLM 配置
    # ------------------------------------------------------------------
    # 企业级主路径：后端模型网关从 YAML/配置中心读取模型清单。
    # 旧版 llm_provider / llm_model / llm_api_key 字段保留为本地 Demo fallback。
    model_gateway_config_path: str = Field(
        default=str(Path("config") / "model_gateway.yaml"),
        description="模型网关配置文件路径。",
    )
    default_llm_model_id: str = Field(default="", description="默认模型 ID，优先级高于配置文件 default_model_id。")

    # llm_provider 决定使用哪个大模型厂商，留空时自动降级为 NoopLLM。
    # 支持的值：anthropic / openai / openai_compatible /
    #           tongyi / zhipuai / baidu
    llm_provider: str = Field(default="", description="LLM 提供商标识。")

    # 模型 ID，各厂商填法不同，参见 .env.example 注释。
    llm_model: str = Field(default="", description="LLM 模型 ID。")

    # 通用 API Key：openai / openai_compatible / tongyi / zhipuai / baidu 均使用此字段。
    llm_api_key: str = Field(default="", description="LLM 通用 API Key。")

    # OpenAI 兼容接口的自定义 base_url（DeepSeek/Kimi/Qwen 等国内模型使用）。
    # openai / anthropic 等官方接口留空即可。
    llm_base_url: str = Field(default="", description="OpenAI 兼容接口 base_url。")

    # Anthropic 专用 API Key（llm_provider=anthropic 时使用）。
    anthropic_api_key: str = Field(default="", description="Anthropic API Key。")

    # 模型网关可选供应商 Key。生产环境通常由环境变量/密钥管理系统注入；
    # 本地开发时 pydantic-settings 会从 .env 读取这些字段。
    openai_api_key: str = Field(default="", description="OpenAI API Key。")
    qwen_api_key: str = Field(default="", description="通义千问 API Key。")
    dashscope_api_key: str = Field(default="", description="阿里云百炼 DashScope API Key，用于 Embedding / Reranker。")
    kimi_api_key: str = Field(default="", description="Moonshot Kimi API Key。")
    jina_api_key: str = Field(default="", description="Jina AI Reranker API Key。")
    local_llm_api_key: str = Field(default="", description="本地 OpenAI-compatible 服务可选 API Key。")

    # ------------------------------------------------------------------
    # Agent 质量门配置
    # ------------------------------------------------------------------
    # 开启后，Agent 会在低质量回答上自动带反馈重试；评分逻辑是规则型的，
    # 不额外调用 judge 模型，但重试本身会增加一次主模型调用。
    agent_enable_reflection: bool = Field(default=True, description="是否启用 Agent 自反思质量门。")
    agent_reflection_threshold: float = Field(default=0.65, description="反思质量门阈值，0-1。")
    agent_max_reflection_retries: int = Field(default=1, description="单轮回答最多反思重试次数。")

    # ------------------------------------------------------------------
    # 向量存储配置（RAG 索引后端）
    # ------------------------------------------------------------------
    # vector_store_type 决定 RAG 向量索引存储在哪里：
    #   milvus — Milvus 向量数据库（默认，生产推荐）
    #   local  — 本地文件系统（无外部服务时的开发兜底）
    vector_store_type: str = Field(default="milvus", description="RAG 向量存储后端：milvus | local")
    vector_store_fallback_to_local: bool = Field(
        default=True,
        description="Milvus 不可用时是否自动回退本地向量存储；生产环境建议设为 false。",
    )
    milvus_uri: str = Field(
        default="",
        description="Milvus URI；优先级高于 host/port，例如 http://localhost:19530 或 Milvus Cloud URI。",
    )
    milvus_token: str = Field(default="", description="Milvus Cloud token，本地 Milvus 可留空。")
    milvus_database: str = Field(default="default", description="Milvus database 名称；本地单库可保持 default。")
    milvus_alias: str = Field(default="", description="pymilvus 连接别名；留空时按 collection 自动生成。")
    # Milvus 集合名称（vector_store_type=milvus 时生效）
    milvus_collection: str = Field(default="knowledge_base", description="Milvus 集合名称")
    milvus_upsert_mode: bool = Field(default=True, description="写入 Milvus 时使用 upsert，避免重复 chunk。")
    milvus_overwrite: bool = Field(default=False, description="启动时是否覆盖 Milvus 集合；生产环境应保持 false。")
    milvus_batch_size: int = Field(default=100, description="Milvus 批量写入大小。")
    milvus_timeout_seconds: float = Field(default=10.0, description="pymilvus 连接和 collection 检查超时时间。")
    milvus_similarity_metric: str = Field(default="COSINE", description="Milvus 向量相似度指标。")
    milvus_consistency_level: str = Field(default="Session", description="Milvus 一致性级别。")
    # 向量维度，需与 embed_model 输出维度一致：
    #   BAAI/bge-small-zh-v1.5 → 512
    #   text-embedding-3-small  → 1536
    milvus_dim: int = Field(default=512, description="向量维度（需与 embed_model 匹配）")

    # ------------------------------------------------------------------
    # Embedding 配置（RAG 向量检索，默认使用本地 BGE 模型）
    # ------------------------------------------------------------------
    # embed_provider: local（HuggingFace BGE，无需 API Key）| openai | mock（仅测试）
    embed_provider: str = Field(default="local", description="Embedding 提供方。")
    # 本地模型名（embed_provider=local 时使用），建议 BAAI/bge-small-zh-v1.5（中文优化）
    embed_model_name: str = Field(default="BAAI/bge-small-zh-v1.5", description="Embedding 模型名。")

    # ------------------------------------------------------------------
    # Langfuse 可观测性配置（可选，留空时自动跳过）
    # ------------------------------------------------------------------
    # 从 Langfuse 项目设置 → API Keys 获取
    langfuse_public_key: str = Field(default="", description="Langfuse Public Key。")
    langfuse_secret_key: str = Field(default="", description="Langfuse Secret Key。")
    # 自托管时填写，使用 cloud.langfuse.com 时留空
    langfuse_host: str = Field(default="", description="Langfuse 服务地址（自托管时填写）。")
    # 兼容 Langfuse / OpenTelemetry 常见命名；优先级低于 LANGFUSE_HOST
    langfuse_base_url: str = Field(default="", description="Langfuse 服务地址别名。")

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_flag(cls, value):
        """兼容常见部署环境写法。

        本地开发常写 DEBUG=true/false，但有些部署环境会写 release/prod。
        Pydantic 默认不能把 release 解析成 bool，这里统一转成 False。
        """
        if isinstance(value, str) and value.strip().lower() in {"release", "prod", "production"}:
            return False
        return value


@lru_cache
def get_settings() -> Settings:
    """获取单例配置对象。

    使用缓存的原因：
    - 避免应用运行期间重复解析环境变量。
    - 在 FastAPI 项目中，这是很常见且清晰的配置组织方式。
    """

    return Settings()
