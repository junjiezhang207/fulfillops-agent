"""Compatibility module for ``app.domain.fulfillment.substitute_sku``."""

import sys

from app.domain.fulfillment import substitute_sku as _module

sys.modules[__name__] = _module
