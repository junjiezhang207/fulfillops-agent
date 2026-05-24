"""Compatibility module for ``app.infrastructure.llm.embedding_adapter``."""

import sys

from app.infrastructure.llm import embedding_adapter as _module

sys.modules[__name__] = _module
