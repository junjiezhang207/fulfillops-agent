"""文件作用摘要：Agent 评测与可观测子模块。

这个子目录放的是“怎么观察和评估 Agent / RAG 质量”的辅助能力，不属于
用户请求主链路。面试时可以把它作为企业级加分项：不是只演示一次回答，
而是用 trace、评分和 Golden Dataset 观察系统是否稳定。

包含文件：
- ``langfuse_tracer.py``：把 LangChain / Agent 调用链路上报到 Langfuse。
- ``ragas_evaluator.py``：用 Ragas 评估 RAG 回答的忠实度和相关性。
- ``golden_dataset.py``：保存人工整理的标准问题和期望结果，用于回归测试。
"""

# 面试官可能问：为什么评测代码不放在业务 service 里？
# 回答：评测是质量保障和离线分析能力，不应该污染线上主链路。单独放在
# evaluation 子包里，既能复用线上 trace / case，也能避免业务代码变重。
