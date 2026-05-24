"""Compatibility module for ``app.agents.quality.evaluation.langfuse_tracer``."""

import sys

from app.agents.quality.evaluation import langfuse_tracer as _module

sys.modules[__name__] = _module
