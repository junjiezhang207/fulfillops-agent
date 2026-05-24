"""Compatibility module for ``app.domain.orders.analysis``."""

import sys

from app.domain.orders import analysis as _module

sys.modules[__name__] = _module
