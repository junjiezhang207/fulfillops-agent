from types import SimpleNamespace

from app.infrastructure.llm.model_gateway import ModelGateway


def test_model_gateway_lists_public_model_metadata_without_secret(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_model_id: fast-agent
models:
  - id: fast-agent
    display_name: Fast Agent
    provider: openai_compatible
    model: deepseek-chat
    base_url: https://api.deepseek.com/v1
    api_key_env: TEST_LLM_KEY
    enabled: true
    priority: 10
    input_price_per_1m_tokens: 0.5
    output_price_per_1m_tokens: 1.5
    input_cache_hit_price_per_1m_tokens: 0.05
    input_cache_miss_price_per_1m_tokens: 0.5
    use_cases: ["agent", "workflow"]
  - id: judge-model
    display_name: Judge Model
    provider: openai
    model: gpt-4o
    api_key_env: TEST_JUDGE_KEY
    enabled: false
    priority: 20
    use_cases: ["judge"]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_LLM_KEY", "secret-value")

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))
    models = gateway.list_models(use_case="agent")

    assert len(models) == 1
    assert models[0]["id"] == "fast-agent"
    assert models[0]["api_key_configured"] is True
    assert models[0]["input_price_per_1m_tokens"] == 0.5
    assert models[0]["output_price_per_1m_tokens"] == 1.5
    assert models[0]["input_cache_hit_price_per_1m_tokens"] == 0.05
    assert models[0]["input_cache_miss_price_per_1m_tokens"] == 0.5
    assert "secret-value" not in str(models[0])


def test_model_gateway_resolves_default_model_by_use_case(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_model_id: agent-model
models:
  - id: agent-model
    display_name: Agent Model
    provider: openai_compatible
    model: deepseek-chat
    api_key_env: TEST_LLM_KEY
    enabled: true
    priority: 10
    use_cases: ["agent"]
  - id: workflow-model
    display_name: Workflow Model
    provider: openai_compatible
    model: qwen-max
    api_key_env: TEST_QWEN_KEY
    enabled: true
    priority: 5
    use_cases: ["workflow"]
""",
        encoding="utf-8",
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))

    assert gateway.resolve_profile(use_case="agent").id == "agent-model"
    assert gateway.resolve_profile(use_case="workflow").id == "workflow-model"


def test_model_gateway_reads_provider_key_from_settings_when_not_exported(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_model_id: qwen-model
models:
  - id: qwen-model
    display_name: Qwen Model
    provider: openai_compatible
    model: qwen-max
    api_key_env: QWEN_API_KEY
    enabled: true
    priority: 10
    use_cases: ["agent"]
""",
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        model_gateway_config_path=str(config),
        default_llm_model_id="",
        qwen_api_key="qwen-secret",
    )

    gateway = ModelGateway(settings)
    active = gateway.active_model(use_case="agent")

    assert active["id"] == "qwen-model"
    assert active["api_key_configured"] is True
    assert "qwen-secret" not in str(active)


def test_model_gateway_uses_task_default_before_global_default(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_model_id: strong-agent
default_models:
  rag_rewrite: small-rewriter
models:
  - id: strong-agent
    display_name: Strong Agent
    provider: openai_compatible
    model: deepseek-chat
    api_key_env: TEST_LLM_KEY
    enabled: true
    priority: 10
    use_cases: ["agent", "rag_rewrite"]
  - id: small-rewriter
    display_name: Small Rewriter
    provider: openai_compatible
    model: qwen-turbo
    api_key_env: TEST_QWEN_KEY
    enabled: true
    priority: 20
    tier: small
    use_cases: ["rag_rewrite"]
""",
        encoding="utf-8",
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id="strong-agent"))

    assert gateway.resolve_profile(use_case="agent").id == "strong-agent"
    assert gateway.resolve_profile(use_case="rag_rewrite").id == "small-rewriter"


def test_model_gateway_marks_local_model_configured_without_api_key(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_models:
  rag_compress: local-small
models:
  - id: local-small
    display_name: Local Small
    provider: openai_compatible
    model: qwen2.5:7b-instruct
    base_url: http://localhost:8003/v1
    requires_api_key: false
    enabled: true
    tier: small
    deployment: local
    use_cases: ["rag_compress"]
""",
        encoding="utf-8",
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))
    active = gateway.active_model(use_case="rag_compress")

    assert active["id"] == "local-small"
    assert active["deployment"] == "local"
    assert active["requires_api_key"] is False
    assert active["api_key_configured"] is True


def test_model_gateway_resolves_embedding_and_reranker_profiles(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_models:
  embedding: bge-small
  reranker: bge-reranker
models:
  - id: bge-small
    display_name: BGE Small
    model_type: embedding
    provider: huggingface
    model: BAAI/bge-small-zh-v1.5
    requires_api_key: false
    enabled: true
    dimension: 512
    use_cases: ["embedding"]
  - id: bge-reranker
    display_name: BGE Reranker
    model_type: reranker
    provider: sentence_transformers
    model: BAAI/bge-reranker-base
    requires_api_key: false
    enabled: true
    top_k: 5
    use_cases: ["reranker"]
""",
        encoding="utf-8",
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))

    embedding = gateway.active_model(use_case="embedding", model_type="embedding")
    reranker = gateway.active_model(use_case="reranker", model_type="reranker")

    assert embedding["id"] == "bge-small"
    assert embedding["model_type"] == "embedding"
    assert embedding["dimension"] == 512
    assert embedding["api_key_configured"] is True
    assert reranker["id"] == "bge-reranker"
    assert reranker["model_type"] == "reranker"
    assert reranker["top_k"] == 5


def test_model_gateway_passes_openai_compatible_extra_body(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_model_id: deepseek-v4-pro
models:
  - id: deepseek-v4-pro
    display_name: DeepSeek V4 Pro
    provider: openai_compatible
    model: deepseek-v4-pro
    base_url: https://api.deepseek.com/v1
    api_key_env: TEST_LLM_KEY
    enabled: true
    use_cases: ["agent"]
    extra_body:
      thinking:
        type: disabled
    streaming: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_LLM_KEY", "secret-value")

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))
    chat_model = gateway.create_chat_model(use_case="agent")

    assert chat_model.extra_body == {"thinking": {"type": "disabled"}}
    assert chat_model.streaming is True


def test_model_gateway_falls_back_to_legacy_env_config_when_yaml_missing():
    settings = SimpleNamespace(
        model_gateway_config_path="missing.yaml",
        default_llm_model_id="",
        llm_provider="openai_compatible",
        llm_model="deepseek-chat",
        llm_base_url="https://api.deepseek.com/v1",
        llm_api_key="legacy-key",
        anthropic_api_key="",
    )

    gateway = ModelGateway(settings)
    active = gateway.active_model(use_case="agent")

    assert active["id"] == "legacy-env"
    assert active["provider"] == "openai_compatible"
    assert active["api_key_configured"] is True


def test_model_gateway_disables_thinking_for_legacy_deepseek_config():
    settings = SimpleNamespace(
        model_gateway_config_path="missing.yaml",
        default_llm_model_id="",
        llm_provider="openai_compatible",
        llm_model="deepseek-v4-pro",
        llm_base_url="https://api.deepseek.com/v1",
        llm_api_key="legacy-key",
        anthropic_api_key="",
    )

    gateway = ModelGateway(settings)
    chat_model = gateway.create_chat_model(use_case="agent")

    assert chat_model.extra_body == {"thinking": {"type": "disabled"}}
