"""Compatibility module for ``app.application.routing.advanced_hybrid_service``."""

import sys

from app.application.routing import advanced_hybrid_service as _module

sys.modules[__name__] = _module
