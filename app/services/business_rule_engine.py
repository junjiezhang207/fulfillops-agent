"""Compatibility module for ``app.domain.rules.business_rule_engine``."""

import sys

from app.domain.rules import business_rule_engine as _module

sys.modules[__name__] = _module
