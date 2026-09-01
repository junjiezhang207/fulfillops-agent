"""Project configuration center.

Learning notes:
- Settings loads .env values through pydantic-settings.
- Model gateway, PostgreSQL, PGVector and memory settings are centralized here.
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
    3. 即使 PostgreSQL、PGVector 或业务系统 API 还没有真正连接，本阶段也先把配置入口准备好。

    这样做的好处是：
    - 后续新增数据库连接时，不需要再回头重构配置体系。
    - 配置项集中、清晰，便于学习整个项目的基础设施边界。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="电商履约运营智能协同 Agent")
    app_version: str = Field(default="0.1.0")
    api_v1_prefix: str = Field(default="/api/v1")
    debug: bool = Field(default=True)
    app_env: str = Field(
        default="production",
        description="运行环境：production 会强制依赖 PostgreSQL/PGVector。",
    )
    allow_infra_fallback: bool = Field(
        default=False,
        description="是否允许基础设施降级。生产模式必须保持 false，避免故障被本地内存/文件掩盖。",
    )
    allow_demo_data: bool = Field(
        default=False,
        description="是否允许使用内置 demo 订单和库存。生产模式必须保持 false。",
    )
    require_postgres: bool = Field(default=True, description="启动与运行时是否强制要求 PostgreSQL 可用。")
    require_pgvector: bool = Field(default=True, description="启动与运行时是否强制要求 PGVector 扩展可用。")
    log_level: str = Field(default="INFO")
    frontend_cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated origins allowed to call the API from the browser.",
    )

    database_url: str = Field(
        default="",
        description="主业务 PostgreSQL 连接串。",
    )
    postgres_url: str = Field(
        default="postgresql+psycopg://fulfillops:fulfillops@localhost:5432/fulfillops_agent",
        description="兼容部署平台命名；留空时使用 DATABASE_URL。",
    )
    short_term_memory_ttl_seconds: int = Field(
        default=86400,
        description="短期会话记忆 TTL，默认 24 小时。",
    )
    short_term_memory_model_extraction_enabled: bool = Field(
        default=False,
        description="是否通过 Model Gateway use_case=memory_extraction 抽取短期结构化记忆。",
    )
    long_term_memory_database_url: str = Field(
        default="",
        description="长期记忆 PostgreSQL 连接串；留空时复用 DATABASE_URL。",
    )
    long_term_memory_vector_dimension: int = Field(
        default=1024,
        description="长期记忆向量维度，需要和模型网关 embedding 模型输出维度一致。",
    )
    long_term_memory_ttl_days: int | None = Field(
        default=180,
        description="长期记忆默认 TTL 天数；None 表示不过期。",
    )
    long_term_memory_pgvector_table: str = Field(
        default="long_term_memory_vectors",
        description="长期记忆 PGVector 表名。",
    )
    long_term_memory_vector_store_type: str = Field(
        default="pgvector",
        description="长期记忆向量后端：pgvector。",
    )
    knowledge_dir: str = Field(
        default=str(Path("app") / "data" / "knowledge")
    )
    knowledge_extra_dirs: str = Field(
        default="knowledge_base",
        description="逗号分隔的附加只读知识目录；默认把根目录 knowledge_base 纳入 RAG。",
    )
    knowledge_recursive: bool = Field(
        default=True,
        description="是否递归扫描知识目录，便于接入按主题分层的知识库。",
    )
    knowledge_index_cache_dir: str = Field(
        default=str(Path("storage") / "knowledge_index")
    )
    knowledge_warmup_on_startup: bool = Field(
        default=True,
        description="启动时是否预热 RAG 索引，避免第一次用户提问时触发全量建库。",
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
    # vector_store_type 决定 RAG 向量索引存储在哪里。默认使用 PostgreSQL PGVector。
    vector_store_type: str = Field(default="pgvector", description="RAG 向量存储后端：pgvector | local")
    vector_store_fallback: str = Field(default="local", description="PGVector 不可用时的 fallback：local")
    vector_store_fallback_to_local: bool = Field(default=False, description="PGVector 不可用时是否允许降级到 local 后端。")
    sop_collection: str = Field(default="sop_collection", description="SOP 规则库逻辑 collection 名称。")
    case_collection: str = Field(default="case_collection", description="优秀案例库逻辑 collection 名称。")
    pgvector_database: str = Field(default="fulfillops_agent", description="PGVector 所在数据库名。")
    pgvector_host: str = Field(default="localhost", description="PGVector 所在 PostgreSQL 主机。")
    pgvector_port: int = Field(default=5432, description="PGVector 所在 PostgreSQL 端口。")
    pgvector_user: str = Field(default="fulfillops", description="PGVector 连接用户名。")
    pgvector_password: str = Field(default="fulfillops", description="PGVector 连接密码。")
    pgvector_table: str = Field(default="knowledge_base_vectors", description="RAG PGVector 表名。")
    # 向量维度，需与当前默认 embedding 输出维度一致：
    #   text-embedding-v4       → 1024（当前默认）
    #   BAAI/bge-small-zh-v1.5 → 512（本地候选）
    #   text-embedding-3-small  → 1536（OpenAI 候选）
    pgvector_dim: int = Field(default=1024, description="向量维度（需与当前 embedding 模型匹配）")

    @property
    def effective_database_url(self) -> str:
        """主数据库连接串，兼容 DATABASE_URL 和 POSTGRES_URL 两种环境变量名。"""
        return self.database_url or self.postgres_url

    # ------------------------------------------------------------------
    # Embedding 配置（RAG 向量检索，默认使用本地 BGE 模型）
    # ------------------------------------------------------------------
    # embed_provider: local（HuggingFace BGE，无需 API Key）| openai | mock（仅测试）
    embed_provider: str = Field(default="local", description="Embedding 提供方。")
    # 本地模型名（embed_provider=local 时使用），建议 BAAI/bge-small-zh-v1.5（中文优化）
    embed_model_name: str = Field(default="BAAI/bge-small-zh-v1.5", description="Embedding 模型名。")

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
