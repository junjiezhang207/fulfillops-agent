"""Compatibility module for ``app.workflows.fulfillment.nodes``."""

import sys

from app.workflows.fulfillment import nodes as _module

sys.modules[__name__] = _module
