"""Agent capability package.

The enterprise layout separates Agent code into:

- runtime: external service entrypoints and session/runtime helpers.
- tools: tool factories, guardrails, cache, and resilience wrappers.
- orchestration: ReAct, Plan-and-Execute, Multi-Agent, and parallel tools.
- quality: evaluation, Langfuse tracing, and golden datasets.
"""
