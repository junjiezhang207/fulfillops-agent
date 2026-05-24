"""Compatibility module for ``app.agents.orchestration.parallel_tools``."""

import sys

from app.agents.orchestration import parallel_tools as _module

sys.modules[__name__] = _module
