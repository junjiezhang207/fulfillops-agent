from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.runtime.context_manager import ContextWindowConfig, ContextWindowManager, TrimResult


class _CountingLLM:
    def get_num_tokens_from_messages(self, messages):
        return sum(len(str(message.content).split()) for message in messages)


def test_context_window_without_token_counter_never_trims():
    messages = [HumanMessage(content="hello"), AIMessage(content="world")]
    manager = ContextWindowManager(llm=None)

    trimmed, result = manager.trim(messages)

    assert trimmed is messages
    assert result.action_taken == "none"
    assert result.messages_dropped == 0
    assert manager.should_trim(messages) is False


def test_context_window_should_trim_when_threshold_exceeded():
    manager = ContextWindowManager(
        ContextWindowConfig(max_tokens=10, trim_threshold=0.5),
        llm=_CountingLLM(),
    )
    messages = [HumanMessage(content="one two three"), AIMessage(content="four five six")]

    assert manager.count_tokens(messages) == 6
    assert manager.should_trim(messages) is True


def test_context_window_trims_old_messages_and_preserves_recent_context():
    manager = ContextWindowManager(
        ContextWindowConfig(max_tokens=8, trim_threshold=0.5, min_messages_to_keep=2),
        llm=_CountingLLM(),
    )
    messages = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="old question words"),
        AIMessage(content="old answer words"),
        HumanMessage(content="new question words"),
        AIMessage(content="new answer words"),
    ]

    trimmed, result = manager.trim(messages)

    assert result.action_taken == "trim_messages"
    assert result.original_count == 5
    assert result.final_count < result.original_count
    assert result.messages_dropped == result.original_count - result.final_count
    assert any(message.content == "new question words" for message in trimmed)
    assert any(message.content == "new answer words" for message in trimmed)


def test_context_window_fallback_keeps_minimum_tail_when_langchain_trim_fails(monkeypatch):
    manager = ContextWindowManager(
        ContextWindowConfig(max_tokens=2, trim_threshold=0.5, min_messages_to_keep=3),
        llm=_CountingLLM(),
    )
    messages = [
        HumanMessage(content="first message"),
        AIMessage(content="second message"),
        HumanMessage(content="third message"),
        AIMessage(content="fourth message"),
    ]

    def _raise(*args, **kwargs):
        raise ValueError("bad message shape")

    monkeypatch.setattr("app.agents.runtime.context_manager.trim_messages", _raise)
    trimmed, result = manager.trim(messages)

    assert trimmed == messages[-3:]
    assert result.final_count == 3
    assert result.messages_dropped == 1


def test_trim_result_reduction_rate_handles_zero_and_nonzero_tokens():
    assert TrimResult(1, 1, 0, 0, "none", 0).reduction_rate == 0.0
    assert TrimResult(4, 2, 100, 25, "trim", 2).reduction_rate == 0.75
