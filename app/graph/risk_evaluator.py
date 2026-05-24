"""Compatibility module for ``app.workflows.fulfillment.risk_evaluator``."""

import sys

from app.workflows.fulfillment import risk_evaluator as _module

sys.modules[__name__] = _module
