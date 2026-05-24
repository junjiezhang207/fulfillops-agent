"""旧服务层兼容包。

新的企业级目录：
- ``app.application``：Hybrid Routing、Workflow facade、会话记忆、响应缓存等用例编排。
- ``app.domain``：订单、库存、履约、仓库、替代 SKU、业务规则等确定性业务能力。
- ``app.agents``：Agent 编排、运行时、工具和质量观测。
- ``app.rag``：RAG 检索、重写、重排、答案构建。
- ``app.infrastructure``：模型网关、LLM/Embedding 等外部技术适配。

``app.services`` 目前只保留旧导入路径兼容，新代码请优先使用上述新目录。
"""
