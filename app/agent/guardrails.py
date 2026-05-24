"""Compatibility module for ``app.agents.tools.guardrails``."""

import sys

from app.agents.tools import guardrails as _module

sys.modules[__name__] = _module
