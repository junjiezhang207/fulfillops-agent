"""Compatibility module for ``app.domain.inventory.analysis``."""

import sys

from app.domain.inventory import analysis as _module

sys.modules[__name__] = _module
