"""文件作用摘要：Agent 包的目录说明。

这个目录集中放大模型智能体相关能力。面试复习时不要把所有文件同等看待：
先看普通 ReAct Agent 主链路，再看工具、安全、记忆；Multi-Agent 和
Plan-and-Execute 属于扩展能力，被问到再讲。

主线文件：
- ``agent.py``：创建 ReAct Agent，并提供可选反思质量门。
- ``tools.py``：把订单、库存、知识库等业务服务封装成工具。
- ``tool_wrapper.py``：给工具加缓存、超时、重试、熔断和输出净化。
- ``guardrails.py``：处理用户输入和工具输出中的 Prompt Injection 风险。
- ``checkpointer.py`` / ``context_manager.py``：处理短期记忆和上下文裁剪。

扩展文件：
- ``parallel_tools.py``：并行执行多个独立工具，降低延迟。
- ``plan_execute.py``：先规划再执行的 Agent 模式。
- ``multi_agent.py``：Supervisor 调度多个专家 Agent 的模式。

评测目录：
- ``evaluation/``：Langfuse、Ragas、Golden Dataset 等质量评测相关能力。
"""

# 面试官可能问：app/agent 目录里哪些是主线，哪些是扩展？
# 回答：主线是 agent.py + tools.py + tool_wrapper.py + guardrails.py；
# checkpointer/context_manager 支撑记忆和上下文；plan_execute/multi_agent/parallel_tools
# 是扩展能力，面试时先讲主线，被追问复杂任务时再讲扩展。
