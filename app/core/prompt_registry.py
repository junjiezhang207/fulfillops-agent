"""Prompt 版本化注册表 — 降级链：Langfuse → YAML → 内置 hardcoded。

为什么 Prompt 不应该硬编码在代码里？
  - 改 Prompt 要改代码 → 提 PR → 等 CI → 重新部署，无法快速迭代
  - 没有版本记录，不知道某次 Prompt 改动是变好还是变差
  - 无法 A/B 测试两个 Prompt 版本

Langfuse Prompt Management 解决这些问题：
  - 在 Langfuse UI 里直接编辑 Prompt，立即生效（无需部署）
  - 每个 Prompt 自动版本化（v1, v2, v3...）
  - 每个 trace 自动关联当时使用的 Prompt 版本（可以看出哪个版本效果最好）
  - 支持按 label 发布（production / staging / experiment）

降级链（三层兜底，生产不中断）：
  1. Langfuse Prompt Management（在线，版本管理，trace 关联）
  2. prompts/*.yaml 文件（本地，有版本号，热加载）
  3. 内置 hardcoded 字符串（最后兜底，服务永不中断）

Langfuse Prompt 命名约定：
  代码名（下划线）    →  Langfuse 名（连字符）
  fulfillment_agent  →  fulfillment-agent
  supervisor_agent   →  supervisor-agent

在 Langfuse UI 创建 Prompt 步骤：
  1. 进入 Langfuse → Prompts → Create
  2. 名称填 "fulfillment-agent"（类型选 Text）
  3. 把 prompts/fulfillment_agent_v1.yaml 里的 system 内容粘进去
  4. 发布（Publish）→ 自动成为 production label
  5. 服务启动时自动拉取，旧版本作为降级保底
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

# 内置 hardcoded（第三层兜底，确保服务不因 Langfuse/文件问题中断）
_BUILTIN_PROMPTS: dict[str, str] = {
    "fulfillment_agent": """\
你是一个高级供应链履约决策助手，专业领域是订单分析、库存优化和履约方案设计。

## 核心职责
根据用户的订单和库存问题，自主决定使用哪些工具来收集信息，然后提供：
1. 明确的问题诊断
2. 可操作的解决方案
3. 关键风险提醒

## 履约优先级规则
1. 现货充足（≥订单量）→ 直接发货
2. 现货不足但有替代品 → 询问或推荐
3. 多个仓库有库存 → 跨仓调配
4. 库存严重不足 → 考虑延期或分批
5. 无可用库存 → 建议停单或长期规划

记住：你的最终目标是最小化客户影响，同时优化成本和库存周转。\
""",
    "supervisor_agent": (
        "你是一个多 Agent 编排系统的 Supervisor。"
        "你的职责是根据用户问题和已收集的信息，决定下一步调用哪个专业 Agent。"
    ),
    "synthesizer_agent": (
        "你是最终答案汇总器。根据各专业 Agent 提供的信息，"
        "生成综合性的、可操作的最终答案。答案要简洁（不超过 300 字）、准确、可操作。"
    ),
}


@dataclass
class PromptEntry:
    name: str
    system: str
    version: str = "unknown"
    description: str = ""
    changelog: str = ""
    source: str = "builtin"   # "langfuse" | "yaml" | "builtin"


class PromptRegistry:
    """三层降级 Prompt 注册表：Langfuse → YAML → 内置 hardcoded。

    使用方式：
        registry = get_prompt_registry()
        system_prompt = registry.get("fulfillment_agent")
        # 指定 label（Langfuse 标签）或版本：
        system_prompt = registry.get("fulfillment_agent", label="staging")
    """

    def __init__(
        self,
        langfuse_client: Any | None = None,
        prompts_dir: Path | None = None,
        langfuse_cache_ttl: int = 300,
    ) -> None:
        self._langfuse = langfuse_client
        self._dir = prompts_dir or _PROMPTS_DIR
        self._cache: dict[str, PromptEntry] = {}
        self._langfuse_cache_ttl = langfuse_cache_ttl

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def get(self, name: str, version: str = "latest", label: str = "production") -> str:
        """获取 system prompt 文本。

        降级顺序：Langfuse → YAML → 内置 hardcoded

        Args:
            name:    prompt 名称（下划线格式），如 "fulfillment_agent"
            version: 版本号（Langfuse 整数版本，或 YAML 版本标识如 "v1"）
            label:   Langfuse 标签（"production" / "staging" / "experiment"）
        """
        return self.get_entry(name, version=version, label=label).system

    def get_entry(self, name: str, version: str = "latest", label: str = "production") -> PromptEntry:
        """获取完整 PromptEntry（含来源、版本、描述）。"""
        return self._get_cached_entry(name, version, label)

    def _get_cached_entry(self, name: str, version: str, label: str) -> PromptEntry:
        """缓存只做一件事：没有就按降级链加载，有就直接返回。"""
        cache_key = f"{name}:{version}:{label}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self._load(name, version, label)
        entry = self._cache[cache_key]
        logger.debug("Prompt '%s' 来源=%s 版本=%s", name, entry.source, entry.version)
        return entry

    def reload(self) -> None:
        """清空缓存，强制下次重新拉取（热更新用）。"""
        self._cache.clear()
        logger.info("PromptRegistry 缓存已清空。")

    def list_available(self) -> list[str]:
        """列出本地 prompts/ 目录下的 YAML 文件。"""
        if not self._dir.exists():
            return []
        return [f.stem for f in self._dir.glob("*.yaml")]

    # ── 内部加载（三层降级）──────────────────────────────────────────────────

    def _load(self, name: str, version: str, label: str) -> PromptEntry:
        # 层 1：Langfuse Prompt Management
        if self._langfuse is not None:
            entry = self._load_from_langfuse(name, version, label)
            if entry is not None:
                return entry

        # 层 2：本地 YAML 文件
        for path in self._get_yaml_candidates(name, version):
            entry = self._load_from_yaml(path, name)
            if entry is not None:
                return entry

        # 层 3：内置 hardcoded（保底）
        text = _BUILTIN_PROMPTS.get(name, "")
        if not text:
            logger.warning("Prompt '%s' 在所有来源中均未找到，返回空字符串。", name)
        else:
            logger.info(
                "Prompt '%s' 使用内置 hardcoded（Langfuse 未配置或 YAML 未找到）。", name
            )
        return PromptEntry(name=name, system=text, version="builtin", source="builtin")

    def _load_from_langfuse(self, name: str, version: str, label: str) -> PromptEntry | None:
        """从 Langfuse Prompt Management 拉取 Prompt。

        Langfuse 命名：下划线 → 连字符（fulfillment_agent → fulfillment-agent）。
        拉取到的 Prompt 自动被 Langfuse 追踪，每个 trace 都会记录使用了哪个版本。
        """
        langfuse_name = name.replace("_", "-")
        try:
            kwargs: dict[str, Any] = {
                "cache_ttl_seconds": self._langfuse_cache_ttl,
                "label": label,
            }
            if version not in ("latest", "production"):
                try:
                    kwargs["version"] = int(version.lstrip("v"))
                    kwargs.pop("label", None)  # 指定版本号时不用 label
                except ValueError:
                    kwargs["label"] = version  # 当作 label 处理

            prompt_obj = self._langfuse.get_prompt(langfuse_name, **kwargs)
            text = getattr(prompt_obj, "prompt", None) or ""
            if not text:
                return None

            prompt_version = getattr(prompt_obj, "version", "?")
            logger.info(
                "Prompt '%s' 已从 Langfuse 加载（v%s, label=%s）", name, prompt_version, label
            )
            return PromptEntry(
                name=name,
                system=text,
                version=f"langfuse-v{prompt_version}",
                source="langfuse",
            )
        except Exception as exc:
            logger.debug(
                "Langfuse prompt '%s' 加载失败（将降级到 YAML）：%s", langfuse_name, exc
            )
            return None

    def _get_yaml_candidates(self, name: str, version: str) -> list[Path]:
        if not self._dir.exists():
            return []
        paths = []
        if version not in ("latest", "production"):
            paths.append(self._dir / f"{name}_{version}.yaml")
        paths.append(self._dir / f"{name}_latest.yaml")
        paths.append(self._dir / f"{name}.yaml")
        # 兜底：扫描最新的版本化文件（如 fulfillment_agent_v2.yaml > v1.yaml）
        versioned = sorted(self._dir.glob(f"{name}_v*.yaml"), reverse=True)
        paths.extend(versioned)
        return paths

    def _load_from_yaml(self, path: Path, name: str) -> PromptEntry | None:
        if not path.exists():
            return None
        try:
            import yaml
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict) or "system" not in data:
                logger.warning("YAML Prompt 格式错误（缺 system 字段）：%s", path)
                return None
            logger.info("Prompt '%s' 已从 YAML 加载：%s", name, path.name)
            return PromptEntry(
                name=data.get("name", name),
                system=data["system"],
                version=data.get("version", "unknown"),
                description=data.get("description", ""),
                changelog=data.get("changelog", ""),
                source="yaml",
            )
        except Exception as exc:
            logger.warning("YAML Prompt 加载失败 %s：%s", path, exc)
            return None


# ── 全局单例工厂 ──────────────────────────────────────────────────────────────

_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """获取全局 PromptRegistry 单例。

    自动尝试连接 Langfuse（已配置时），未配置时降级 YAML → 内置。
    """
    global _registry
    if _registry is None:
        _langfuse_client = _try_create_langfuse_client()
        _registry = PromptRegistry(langfuse_client=_langfuse_client)
        if _langfuse_client:
            logger.info("PromptRegistry: Langfuse Prompt Management 已启用")
        else:
            logger.info("PromptRegistry: 使用本地 YAML 文件（Langfuse 未配置）")
    return _registry


def _try_create_langfuse_client() -> Any | None:
    """尝试创建 Langfuse 客户端，用于 Prompt Management。未配置或失败时返回 None。"""
    try:
        from app.core.config import get_settings
        settings = get_settings()
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            return None
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host or "https://cloud.langfuse.com",
        )
    except Exception as exc:
        logger.debug("Langfuse 客户端创建失败（非致命）：%s", exc)
        return None
