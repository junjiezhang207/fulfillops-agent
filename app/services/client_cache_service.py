"""Compatibility module for ``app.application.cache.client_cache_service``."""

import sys

from app.application.cache import client_cache_service as _module

sys.modules[__name__] = _module
