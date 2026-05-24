"""Compatibility module for ``app.agents.quality.evaluation.golden_dataset``."""

import sys

from app.agents.quality.evaluation import golden_dataset as _module

sys.modules[__name__] = _module
