"""Compatibility module for ``app.agents.orchestration.plan_execute``."""

import sys

from app.agents.orchestration import plan_execute as _module

sys.modules[__name__] = _module
