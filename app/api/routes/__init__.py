"""API 路由包。

文件作用摘要：
``routes`` 目录按业务能力拆分多个 FastAPI ``APIRouter``：
- ``health.py``：健康检查，给前端和部署平台判断后端是否可用。
- ``orders.py`` / ``inventory.py`` / ``knowledge.py``：基础业务能力接口。
- ``workflow.py`` / ``agent.py`` / ``hybrid.py``：不同智能决策链路的入口。
- ``enterprise_data.py`` / ``knowledge_mgmt.py``：企业数据和知识库管理接口。
- ``models.py`` / ``metrics.py``：模型网关和可观测性接口。

学习时先看 ``router.py`` 如何把这些子路由挂到一起，再看每个路由文件如何把 HTTP
请求转交给 service 层。
"""
