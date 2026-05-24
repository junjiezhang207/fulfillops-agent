"""Compatibility module for ``app.agents.runtime.checkpointer``."""

import sys

from app.agents.runtime import checkpointer as _module

sys.modules[__name__] = _module
