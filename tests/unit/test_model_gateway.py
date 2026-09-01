from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

import app.infrastructure.llm.model_gateway as model_gateway_module
from app.infrastructure.llm.model_gateway import FallbackChatModel, ModelGateway


class _FailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing-test-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("primary down")


class _AuthFailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "auth-failing-test-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("401 unauthorized api key")


class _UnknownFailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "unknown-failing-test-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("vendor exploded in a surprising way")


class _StaticChatModel(BaseChatModel):
    text: str

    @property
    def _llm_type(self) -> str:
        return "static-test-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])


class _FailingRunnable(Runnable[Any, Any]):
    def __init__(self, message: str):
        self.message = message

    def invoke(self, input: Any, config=None, **kwargs: Any) -> Any:
        raise RuntimeError(self.message)


class _StaticRunnable(Runnable[Any, Any]):
    def __init__(self, value: Any):
        self.value = value

    def invoke(self, input: Any, config=None, **kwargs: Any) -> Any:
        return self.value


class _ToolBindingChatModel(BaseChatModel):
    bound_runnable: Any

    @property
    def _llm_type(self) -> str:
        return "tool-binding-test-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="unused"))])

    def bind_tools(self, tools, *, tool_choice: str | None = None, **kwargs: Any):
        return self.bound_runnable


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


def test_model_gateway_resolves_configured_fallback_chain(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_models:
  agent: primary-agent
fallback_models:
  agent: [backup-agent, ignored-disabled]
models:
  - id: primary-agent
    display_name: Primary
    provider: openai_compatible
    model: primary
    api_key_env: TEST_LLM_KEY
    enabled: true
    priority: 10
    use_cases: ["agent"]
  - id: backup-agent
    display_name: Backup
    provider: openai_compatible
    model: backup
    api_key_env: TEST_BACKUP_KEY
    enabled: true
    priority: 20
    use_cases: ["agent"]
  - id: ignored-disabled
    display_name: Disabled
    provider: openai_compatible
    model: disabled
    api_key_env: TEST_DISABLED_KEY
    enabled: false
    priority: 30
    use_cases: ["agent"]
""",
        encoding="utf-8",
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))
    chain = gateway.resolve_profile_chain(use_case="agent")
    active = gateway.active_model(use_case="agent")

    assert [profile.id for profile in chain] == ["primary-agent", "backup-agent"]
    assert active["id"] == "primary-agent"
    assert active["fallback_model_ids"] == ["backup-agent"]


def test_fallback_chat_model_uses_backup_when_primary_fails():
    model = FallbackChatModel(
        models=[_FailingChatModel(), _StaticChatModel(text="backup ok")],
        model_ids=["primary", "backup"],
        use_case="agent",
    )

    result = model.invoke("hello")

    assert result.content == "backup ok"
    assert model.fallback_model_ids == ["backup"]


def test_fallback_chat_model_does_not_fallback_for_auth_error():
    model = FallbackChatModel(
        models=[_AuthFailingChatModel(), _StaticChatModel(text="backup ok")],
        model_ids=["primary", "backup"],
        use_case="agent",
    )

    with pytest.raises(RuntimeError, match="401 unauthorized"):
        model.invoke("hello")


def test_fallback_chat_model_keeps_unknown_error_strict_by_default():
    model = FallbackChatModel(
        models=[_UnknownFailingChatModel(), _StaticChatModel(text="backup ok")],
        model_ids=["primary", "backup"],
        use_case="agent",
    )

    with pytest.raises(RuntimeError, match="surprising way"):
        model.invoke("hello")


def test_fallback_chat_model_can_fallback_for_unknown_error_when_configured():
    model = FallbackChatModel(
        models=[_UnknownFailingChatModel(), _StaticChatModel(text="backup ok")],
        model_ids=["primary", "backup"],
        use_case="agent",
        fallback_on_unknown_error=True,
    )

    result = model.invoke("hello")

    assert result.content == "backup ok"


def test_bound_tool_runnable_uses_retryable_fallback_policy():
    model = FallbackChatModel(
        models=[
            _ToolBindingChatModel(bound_runnable=_FailingRunnable("503 service unavailable")),
            _ToolBindingChatModel(bound_runnable=_StaticRunnable("backup ok")),
        ],
        model_ids=["primary", "backup"],
        use_case="agent",
    )

    bound = model.bind_tools([])
    result = bound.invoke("hello")

    assert result == "backup ok"
    assert bound.fallback_model_ids == ["backup"]


def test_bound_tool_runnable_does_not_fallback_for_auth_error():
    model = FallbackChatModel(
        models=[
            _ToolBindingChatModel(bound_runnable=_FailingRunnable("401 unauthorized api key")),
            _ToolBindingChatModel(bound_runnable=_StaticRunnable("backup ok")),
        ],
        model_ids=["primary", "backup"],
        use_case="agent",
    )

    bound = model.bind_tools([])

    with pytest.raises(RuntimeError, match="401 unauthorized"):
        bound.invoke("hello")


def test_model_gateway_skips_unconfigured_primary_and_builds_backup(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_models:
  agent: primary-agent
fallback_models:
  agent: [backup-agent]
models:
  - id: primary-agent
    display_name: Primary
    provider: openai_compatible
    model: primary
    api_key_env: MISSING_PRIMARY_KEY
    enabled: true
    use_cases: ["agent"]
  - id: backup-agent
    display_name: Backup
    provider: openai_compatible
    model: backup
    api_key_env: BACKUP_KEY
    enabled: true
    use_cases: ["agent"]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("BACKUP_KEY", "backup-secret")
    monkeypatch.setattr(
        model_gateway_module,
        "_build_chat_model",
        lambda **_kwargs: _StaticChatModel(text="backup ready"),
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))
    model = gateway.create_chat_model(use_case="agent")

    assert model.invoke("hello").content == "backup ready"


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


def test_model_gateway_parses_fallback_policy_and_passes_it_to_chat_wrapper(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_models:
  agent: primary-agent
fallback_policy:
  fallback_on_unknown_error: true
  retryable_error_types: [timeout]
fallback_models:
  agent: [backup-agent]
models:
  - id: primary-agent
    display_name: Primary
    provider: openai_compatible
    model: primary
    api_key_env: TEST_LLM_KEY
    enabled: true
    use_cases: ["agent"]
  - id: backup-agent
    display_name: Backup
    provider: openai_compatible
    model: backup
    api_key_env: TEST_LLM_KEY
    enabled: true
    use_cases: ["agent"]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_LLM_KEY", "secret-value")
    monkeypatch.setattr(
        model_gateway_module,
        "_build_chat_model",
        lambda **_kwargs: _StaticChatModel(text="ready"),
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))
    chat_model = gateway.create_chat_model(use_case="agent")

    assert isinstance(chat_model, FallbackChatModel)
    assert chat_model.fallback_on_unknown_error is True
    assert chat_model.retryable_error_types == ["timeout"]


def test_model_gateway_loads_prompt_by_use_case(tmp_path):
    prompt_file = tmp_path / "agent_prompt.yaml"
    prompt_file.write_text(
        """
version: "test-v1"
name: agent_prompt
description: Test prompt
system: |
  你是测试 Agent。
""",
        encoding="utf-8",
    )
    config = tmp_path / "models.yaml"
    config.write_text(
        f"""
default_prompts:
  agent: agent_prompt
prompt_profiles:
  agent_prompt:
    use_case: agent
    path: {prompt_file.name}
    version: test-v1
    fallback_builtin: true
models: []
""",
        encoding="utf-8",
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))
    prompt = gateway.load_prompt(use_case="agent")

    assert prompt.id == "agent_prompt"
    assert prompt.source == "yaml"
    assert prompt.version == "test-v1"
    assert "测试 Agent" in prompt.system


def test_model_gateway_prompt_profile_use_case_overrides_reused_yaml(tmp_path):
    prompt_file = tmp_path / "replan_prompt.yaml"
    prompt_file.write_text(
        """
version: "test-v1"
name: shared_replan_prompt
use_case: replan
system: |
  你是重新规划 Agent。
""",
        encoding="utf-8",
    )
    config = tmp_path / "models.yaml"
    config.write_text(
        f"""
default_prompts:
  replanner: replanner_prompt
prompt_profiles:
  replanner_prompt:
    use_case: replanner
    path: {prompt_file.name}
    version: test-v1
    fallback_builtin: true
models: []
""",
        encoding="utf-8",
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))
    prompt = gateway.load_prompt(use_case="replanner")

    assert prompt.source == "yaml"
    assert prompt.use_case == "replanner"


def test_model_gateway_prompt_falls_back_to_builtin_when_file_missing(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_prompts:
  agent: fulfillment_agent
prompt_profiles:
  fulfillment_agent:
    use_case: agent
    path: missing.yaml
    fallback_builtin: true
models: []
""",
        encoding="utf-8",
    )

    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))
    prompt = gateway.load_prompt(use_case="agent")

    assert prompt.source == "builtin"
    assert "履约" in prompt.system


def test_project_model_gateway_config_covers_known_use_cases():
    gateway = ModelGateway(SimpleNamespace(model_gateway_config_path="config/model_gateway.yaml", default_llm_model_id=""))
    report = gateway.validate_use_case_coverage()

    assert report["missing_default_models"] == []
    assert report["missing_supported_profiles"] == []
    assert report["unknown_default_use_cases"] == []
    assert report["unknown_fallback_use_cases"] == []
    assert report["unknown_profile_use_cases"] == []
    assert report["unknown_model_refs"] == []
    assert report["default_model_use_case_mismatches"] == []
    assert report["fallback_model_use_case_mismatches"] == []
    assert report["disabled_default_models"] == []
    assert report["missing_default_prompts"] == []
    assert report["missing_prompt_profiles"] == []
    assert report["unknown_default_prompt_use_cases"] == []
    assert report["unknown_prompt_profile_use_cases"] == []
    assert report["unknown_prompt_refs"] == []
    assert report["prompt_use_case_mismatches"] == []
    assert report["prompt_file_missing"] == []
    assert report["prompt_variable_mismatches"] == []


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
