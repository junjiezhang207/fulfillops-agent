"""根级 pytest 配置 — 注册自定义选项和 marker。

--slow 选项：
  默认情况下跳过所有 @pytest.mark.slow 测试（需要 LLM 的 DeepEval 评测）。
  本地深度评测时加 --slow 参数运行：
    pytest tests/eval/ --slow
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="运行需要 LLM 的慢速评测测试（DeepEval LLM-as-judge）",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--slow"):
        return  # --slow 时全部运行

    skip_slow = pytest.mark.skip(
        reason="需要 LLM 的慢速测试，加 --slow 参数运行：pytest --slow"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
