"""Compatibility module for ``app.workflows.fulfillment.state``."""

import sys

from app.workflows.fulfillment import state as _module

sys.modules[__name__] = _module
