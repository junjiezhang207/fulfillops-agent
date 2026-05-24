"""Compatibility module for ``app.agents.orchestration.multi_agent``."""

import sys

from app.agents.orchestration import multi_agent as _module

sys.modules[__name__] = _module
