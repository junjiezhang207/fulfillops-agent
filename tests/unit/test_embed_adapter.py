import httpx

from app.graph.embed_adapter import OpenAICompatibleEmbedding


def test_openai_compatible_embedding_posts_to_embeddings_endpoint(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }

    def fake_post(url: str, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    embed = OpenAICompatibleEmbedding(
        model_name="text-embedding-v4",
        api_key="dashscope-key",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions=1024,
    )

    result = embed._get_text_embeddings(["a", "b"])

    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer dashscope-key"
    assert captured["json"] == {
        "model": "text-embedding-v4",
        "input": ["a", "b"],
        "dimensions": 1024,
    }
    assert result == [[0.1, 0.2], [0.3, 0.4]]
