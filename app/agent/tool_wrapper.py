"""Compatibility module for ``app.agents.tools.wrapper``."""

import sys

from app.agents.tools import wrapper as _module

sys.modules[__name__] = _module
