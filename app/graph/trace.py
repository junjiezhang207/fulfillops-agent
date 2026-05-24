"""Compatibility module for ``app.workflows.fulfillment.trace``."""

import sys

from app.workflows.fulfillment import trace as _module

sys.modules[__name__] = _module
