"""LLM 适配入口。

历史上这里直接根据 `.env` 构造各家 ChatModel。
现在主路径已经升级为后端模型网关：模型清单、用途、密钥来源由
``app.services.model_gateway.ModelGateway`` 管理。

保留 ``LLMFactory`` 是为了兼容项目里已有调用点。
"""

from langchain_core.language_models import BaseChatModel

from app.services.model_gateway import ModelGateway


class LLMFactory:
    """兼容旧调用点的轻量工厂，内部委托给 ModelGateway。

    教学理解：
      workflow 节点只需要一个 BaseChatModel | None，不应该知道 DeepSeek、
      OpenAI、Qwen 或本地模型分别怎么初始化。模型选择、API Key、base_url 等
      基础设施细节都收口在 ModelGateway 里。

    因此这个类不是新的抽象层，而是“旧接口适配器”：
      老代码仍然可以调用 LLMFactory.create_chat_model(settings)，
      新逻辑则统一落到 ModelGateway.create_chat_model()。
    """

    @staticmethod
    def create_chat_model(
        settings: object,
        use_case: str = "agent",
        model_id: str | None = None,
    ) -> BaseChatModel | None:
        return ModelGateway(settings).create_chat_model(use_case=use_case, model_id=model_id)

    @staticmethod
    def create(settings: object) -> BaseChatModel | None:
        """兼容旧 demo 脚本里的 LLMFactory.create(settings)。"""
        return LLMFactory.create_chat_model(settings)
