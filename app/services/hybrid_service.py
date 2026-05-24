"""Compatibility module for ``app.application.routing.hybrid_service``."""

import sys

from app.application.routing import hybrid_service as _module

sys.modules[__name__] = _module
