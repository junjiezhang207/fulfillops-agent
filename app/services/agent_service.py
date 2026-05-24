"""Compatibility module for ``app.agents.runtime.agent_service``."""

import sys

from app.agents.runtime import agent_service as _module

sys.modules[__name__] = _module
