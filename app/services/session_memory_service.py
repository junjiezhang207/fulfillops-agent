"""Compatibility module for ``app.application.memory.session_memory_service``."""

import sys

from app.application.memory import session_memory_service as _module

sys.modules[__name__] = _module
