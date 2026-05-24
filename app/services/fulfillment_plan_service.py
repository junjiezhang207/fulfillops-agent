"""Compatibility module for ``app.domain.fulfillment.plan_service``."""

import sys

from app.domain.fulfillment import plan_service as _module

sys.modules[__name__] = _module
