"""已废弃 — LLMPort / NoopLLM 已替换为 LangChain BaseChatModel | None。

迁移说明：
  旧版：WorkflowNodes(llm=LangChainLLMAdapter(chat_model))
  新版：WorkflowNodes(chat_model=chat_model)

  旧版：LLMFactory.create(settings) → LLMPort
  新版：LLMFactory.create_chat_model(settings) → BaseChatModel | None

为什么保留空文件：
  1. 避免旧文档、旧 demo 或外部导入马上报 ImportError。
  2. 给读代码的人一个迁移路标，知道 LLM 抽象已经收敛到 LangChain 标准接口。
  3. 后续如果确认没有任何旧引用，可以安全删除这个文件并更新文档。
"""
