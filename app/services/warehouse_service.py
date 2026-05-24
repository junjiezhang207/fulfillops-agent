"""Compatibility module for ``app.domain.inventory.warehouse_service``."""

import sys

from app.domain.inventory import warehouse_service as _module

sys.modules[__name__] = _module
