"""Compatibility module for ``app.application.workflow.workflow_service``."""

import sys

from app.application.workflow import workflow_service as _module

sys.modules[__name__] = _module
