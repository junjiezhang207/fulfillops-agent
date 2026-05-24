"""Compatibility module for ``app.agents.tools.cache``."""

import sys

from app.agents.tools import cache as _module

sys.modules[__name__] = _module
