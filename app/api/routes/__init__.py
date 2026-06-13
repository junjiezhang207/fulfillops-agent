"""API 路由包。

``routes`` 目录按业务能力拆分多个 FastAPI ``APIRouter``：
- ``health.py``：健康检查，给前端和部署平台判断后端是否可用。
- ``orders.py`` / ``inventory.py`` / ``knowledge.py``：基础业务能力接口。
- ``workflow.py`` / ``agent.py`` / ``hybrid.py``：不同智能决策链路的入口。
- ``enterprise_data.py`` / ``knowledge_mgmt.py``：企业数据和知识库管理接口。
- ``models.py`` / ``metrics.py``：模型网关和可观测性接口。
"""
