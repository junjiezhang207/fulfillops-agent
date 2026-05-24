"""Compatibility module for ``app.infrastructure.llm.model_gateway``."""

import sys

from app.infrastructure.llm import model_gateway as _module

sys.modules[__name__] = _module
