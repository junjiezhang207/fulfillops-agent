"""Compatibility module for ``app.agents.runtime.context_manager``."""

import sys

from app.agents.runtime import context_manager as _module

sys.modules[__name__] = _module
