"""Compatibility module for ``app.application.routing.intent_classifier``."""

import sys

from app.application.routing import intent_classifier as _module

sys.modules[__name__] = _module
