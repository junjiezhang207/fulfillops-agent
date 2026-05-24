"""执行轨迹（Trace）事件与工具。

这里的两个模型直接复用在 GraphState 里，也会被 API 响应原样返回。
原因：学习型项目，可观察性必须"显式可见"。

后续接入 LangSmith / OpenTelemetry 时，这两个模型的字段可以平滑映射：
    TraceEvent   ≈ OTel Span
    ErrorEvent   ≈ OTel Span Event (exception)
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """节点级执行轨迹事件。

    每个节点进入/退出都会产生一条 TraceEvent，
    最终以列表形式累加到 state["trace"] 里。
    """

    node: str = Field(..., description="节点名。")
    start_ts: datetime = Field(..., description="节点开始执行的时间。")
    end_ts: datetime = Field(..., description="节点结束执行的时间。")
    elapsed_ms: int = Field(..., description="节点耗时，单位毫秒。")
    status: Literal["ok", "error", "skipped"] = Field(
        ..., description="节点执行状态。"
    )
    note: str | None = Field(
        default=None,
        description="可选的附加说明，例如条件分支结果、命中类别等。",
    )


class ErrorEvent(BaseModel):
    """节点执行过程中产生的错误事件。

    注意：节点捕获异常后只写 ErrorEvent，不 raise。
    主链路继续向后推进（除非是 dispatch_node 的参数校验错误）。
    """

    node: str = Field(..., description="出错节点名。")
    message: str = Field(..., description="错误信息。")
    exception_type: str = Field(..., description="异常类型名。")


def build_trace_event(
    node: str,
    start_ts: datetime,
    end_ts: datetime,
    status: Literal["ok", "error", "skipped"],
    note: str | None = None,
) -> TraceEvent:
    """统一的 TraceEvent 构造工具。

    把 elapsed_ms 的计算封在一个地方，避免各节点重复实现。
    """

    elapsed_ms = int((end_ts - start_ts).total_seconds() * 1000)
    return TraceEvent(
        node=node,
        start_ts=start_ts,
        end_ts=end_ts,
        elapsed_ms=elapsed_ms,
        status=status,
        note=note,
    )
