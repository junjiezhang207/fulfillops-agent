"""Compatibility module for ``app.infrastructure.llm.chat_adapter``."""

import sys

from app.infrastructure.llm import chat_adapter as _module

sys.modules[__name__] = _module
