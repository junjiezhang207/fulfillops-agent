"""Compatibility module for ``app.agents.runtime.multi_agent_service``."""

import sys

from app.agents.runtime import multi_agent_service as _module

sys.modules[__name__] = _module
