# Agent 与 Workflow

## 1. 为什么同时保留两条链路

Workflow 和 Agent 不是重复功能，而是解决不同问题。

| 链路 | 适合场景 | 优点 | 风险 |
| --- | --- | --- | --- |
| Workflow | 标准履约判断、路径稳定 | 可控、可解释、好测试 | 灵活性较低 |
| Agent | 开放问题、多工具推理 | 灵活、能处理追问 | 成本更高、需要防护 |

面试时可以说：企业里不会所有事情都让 Agent 自由发挥，关键业务链路仍然需要固定流程兜底。

## 2. Workflow 链路

核心文件：

| 文件 | 职责 |
| --- | --- |
| `app/workflows/fulfillment/` | LangGraph 节点、状态和履约 Workflow 编排 |
| `app/services/workflow_service.py` | Workflow 应用服务入口 |
| `app/domain/orders/analysis.py` | 订单分析 |
| `app/domain/inventory/analysis.py` | 库存分析 |
| `app/domain/fulfillment/plan_service.py` | 履约方案生成 |
| `app/domain/rules/business_rule_engine.py` | 业务规则引擎 |

Workflow 的输出更结构化，适合展示“为什么这么判断”。

## 3. Agent 链路

核心文件：

| 文件 | 职责 |
| --- | --- |
| `app/agents/orchestration/react_agent.py` | ReAct Agent 主编排 |
| `app/agents/runtime/agent_service.py` | Agent API 服务封装 |
| `app/agents/tools/factory.py` | Agent 可调用工具 |
| `app/agents/tools/wrapper.py` | 工具统一包装、错误处理、观测 |
| `app/agents/tools/guardrails.py` | Prompt Injection 和敏感信息防护 |
| `app/agents/runtime/context_manager.py` | 上下文压缩和历史消息管理 |
| `app/agents/runtime/checkpointer.py` | Agent 状态检查点 |

Agent 可以调用订单、库存、知识库、履约建议等工具。

## 4. 流式输出

Agent 支持流式接口：

```text
POST /api/v1/agent/chat/stream
```

前端会边接收 token 边展示，降低首字等待时间。普通非流式接口仍然保留：

```text
POST /api/v1/agent/chat
```

## 5. 反思机制

Agent 中的反思不是每轮都触发，而是在以下情况更应该触发：

- 工具执行失败。
- 回答置信度较低。
- 结果缺少依据。
- 用户问题涉及复杂业务判断。
- 多工具结果之间存在冲突。

这样反思更像质量兜底，而不是无意义地增加成本和延迟。

## 6. 前端聊天记录

React 前端为订单处理和人工审查保留操作上下文。用户触发高风险中断后，可以在审查队列或独立审查页继续处理，不会丢失当前 `thread_id`。
