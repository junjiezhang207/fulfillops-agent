import httpx

from app.infrastructure.llm.embedding_adapter import OpenAICompatibleEmbedding


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


def test_openai_compatible_embedding_batches_large_inputs(monkeypatch):
    captured_inputs: list[list[str]] = []

    class FakeResponse:
        def __init__(self, inputs: list[str]) -> None:
            self.inputs = inputs

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {"index": index, "embedding": [float(len(value))]}
                    for index, value in enumerate(self.inputs)
                ]
            }

    def fake_post(url: str, **kwargs):
        inputs = kwargs["json"]["input"]
        captured_inputs.append(inputs)
        return FakeResponse(inputs)

    monkeypatch.setattr(httpx, "post", fake_post)

    embed = OpenAICompatibleEmbedding(
        model_name="text-embedding-v4",
        api_key="dashscope-key",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions=1024,
        batch_size=2,
    )

    result = embed._get_text_embeddings(["a", "bb", "ccc", "dddd", "eeeee"])

    assert captured_inputs == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_openai_compatible_embedding_retries_transient_errors(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"index": 0, "embedding": [1.0]}]}

    def fake_post(url: str, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("ssl eof")
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("app.infrastructure.llm.embedding_adapter.time.sleep", sleeps.append)

    embed = OpenAICompatibleEmbedding(
        model_name="text-embedding-v4",
        api_key="dashscope-key",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        retry_attempts=2,
        retry_backoff_seconds=0.5,
    )

    assert embed._get_text_embedding("a") == [1.0]
    assert calls["count"] == 2
    assert sleeps == [0.5]
