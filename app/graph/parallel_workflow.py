"""Compatibility module for ``app.workflows.fulfillment.parallel_graph``."""

import sys

from app.workflows.fulfillment import parallel_graph as _module

sys.modules[__name__] = _module
