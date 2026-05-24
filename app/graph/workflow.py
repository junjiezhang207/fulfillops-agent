"""Compatibility module for ``app.workflows.fulfillment.graph``."""

import sys

from app.workflows.fulfillment import graph as _module

sys.modules[__name__] = _module
