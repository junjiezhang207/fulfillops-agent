"""API boundary package.

This package only owns FastAPI HTTP boundary code: request parameters, status
codes, response models, and route wiring.

Current backend layering:
1. ``app/api``: HTTP entrypoints.
2. ``app/application``: use-case orchestration such as Hybrid Routing and Workflow facade.
3. ``app/domain``: deterministic OMS/WMS business rules and fulfillment logic.
4. ``app/repositories`` / ``app/infrastructure``: data sources and external adapters.
"""
