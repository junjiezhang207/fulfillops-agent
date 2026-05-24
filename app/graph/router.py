"""Compatibility module for ``app.workflows.fulfillment.router``."""

import sys

from app.workflows.fulfillment import router as _module

sys.modules[__name__] = _module
