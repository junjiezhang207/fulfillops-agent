"""Compatibility module for ``app.workflows.fulfillment.ports``."""

import sys

from app.workflows.fulfillment import ports as _module

sys.modules[__name__] = _module
