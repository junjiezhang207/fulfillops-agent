"""Compatibility module for ``app.domain.fulfillment.options_generator``."""

import sys

from app.domain.fulfillment import options_generator as _module

sys.modules[__name__] = _module
