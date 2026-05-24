"""Compatibility module for ``app.agents.runtime.plan_execute_service``."""

import sys

from app.agents.runtime import plan_execute_service as _module

sys.modules[__name__] = _module
