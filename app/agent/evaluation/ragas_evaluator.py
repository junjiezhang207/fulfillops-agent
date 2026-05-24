"""Compatibility module for ``app.agents.quality.evaluation.ragas_evaluator``."""

import sys

from app.agents.quality.evaluation import ragas_evaluator as _module

sys.modules[__name__] = _module
