"""旧 Agent 路径兼容包。

新的企业级目录是 ``app/agents``：
- ``app/agents/orchestration``：ReAct、Plan-Execute、Multi-Agent 编排实现
- ``app/agents/runtime``：AgentService、checkpointer、上下文窗口管理
- ``app/agents/tools``：工具工厂、工具包装、缓存、Guardrails
- ``app/agents/quality``：Langfuse、RAGAS、Golden Dataset 等评测观测能力

这个 ``app/agent`` 包只保留兼容导出，避免旧测试、脚本或文档示例立刻失效。
新代码请优先导入 ``app.agents``。
"""
