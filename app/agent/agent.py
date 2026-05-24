"""Compatibility module for ``app.agents.orchestration.react_agent``."""

import sys

from app.agents.orchestration import react_agent as _module

sys.modules[__name__] = _module
