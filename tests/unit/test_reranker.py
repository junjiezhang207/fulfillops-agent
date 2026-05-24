from types import SimpleNamespace

import httpx

from app.rag.reranker import DashScopeReranker, create_reranker


def test_dashscope_reranker_uses_openai_compatible_reranks_api(monkeypatch):
    """阿里云 qwen3-rerank 走 /reranks 接口，并按 relevance_score 重排。"""

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.32},
                ]
            }

    def fake_post(url: str, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    reranker = DashScopeReranker(api_key="dashscope-key", top_k=2)
    result = reranker.rerank_with_scores("库存不足怎么办", ["普通规则", "缺货履约规则"])

    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    assert captured["headers"]["Authorization"] == "Bearer dashscope-key"
    assert captured["json"] == {
        "model": "qwen3-rerank",
        "query": "库存不足怎么办",
        "documents": ["普通规则", "缺货履约规则"],
        "top_n": 2,
    }
    assert [item.text for item in result] == ["缺货履约规则", "普通规则"]
    assert result[0].rerank_score == 0.91


def test_create_reranker_supports_dashscope_provider(tmp_path, monkeypatch):
    """模型网关配置 provider=dashscope 时，能创建阿里云 Reranker 适配器。"""

    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_models:
  reranker: aliyun-qwen3-rerank
models:
  - id: aliyun-qwen3-rerank
    display_name: Aliyun Qwen3 Rerank
    model_type: reranker
    provider: dashscope
    model: qwen3-rerank
    base_url: https://dashscope.aliyuncs.com/compatible-api/v1/reranks
    api_key_env: DASHSCOPE_API_KEY
    enabled: true
    top_k: 7
    use_cases: ["reranker"]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")

    reranker = create_reranker(SimpleNamespace(model_gateway_config_path=str(config), default_llm_model_id=""))

    assert isinstance(reranker, DashScopeReranker)
