"""Common API response schemas.

Learning notes:
- ApiResponse gives all endpoints the same success/message/data shape.
- A consistent shape makes frontend handling and API documentation simpler.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查接口响应模型。"""

    status: str = Field(..., description="当前服务状态。")
    app_name: str = Field(..., description="当前应用名称。")
    version: str = Field(..., description="当前应用版本。")


class ApiResponse(BaseModel):
    """通用 API 响应模型。

    说明：
    - 当前阶段先保持简单，只提供最基础的成功标记、消息和数据结构。
    - 后续进入核心模块阶段后，再根据任务接口需要扩展更细的响应结构。
    """

    success: bool = Field(..., description="本次请求是否成功。")
    message: str = Field(..., description="接口返回消息。")
    data: dict = Field(default_factory=dict, description="接口返回数据。")
