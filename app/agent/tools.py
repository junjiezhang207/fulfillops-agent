"""Compatibility module for ``app.agents.tools.factory``."""

import sys

from app.agents.tools import factory as _module

sys.modules[__name__] = _module
