"""RAG 知识文档入库流水线。

这个模块放在 ``app/rag`` 下，是因为它服务的是 RAG 知识库“入库前”的治理流程：

1. 把运营人员上传的 PDF / Word / Markdown / TXT 统一解析成 Markdown。
2. 在进入向量索引前做清洗、脱敏、去重和质量校验。
3. 先写入 pending 区，等待人工确认；确认后才进入正式知识目录。
4. 为每次确认入库保存版本快照和解析报告，便于审计和回滚。

为什么不把这些逻辑直接写在 ``knowledge_mgmt.py`` 路由里？
- 路由层应该只负责 HTTP 参数、异常映射和后台任务调度。
- 文档解析/清洗/质检是可独立测试、可替换、可扩展的一套 pipeline。
- 未来如果接对象存储、知识管理系统或异步队列，这个模块仍然可以复用。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any


# 这些后缀是当前 pipeline 真正能处理的源格式。
# .md/.txt 是已有能力；.pdf/.docx 是本次新增能力。
SUPPORTED_SOURCE_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}


@dataclass
class ParsedDocument:
    """解析阶段的输出。

    注意：这里的 ``markdown`` 还不是最终入库版本，只是“从源文件抽出的可读 Markdown”。
    后面还会经过清洗、脱敏、质量校验和 front matter 注入。
    """

    markdown: str
    source_type: str
    parser: str
    pages: int = 0
    tables: int = 0
    repaired_encoding: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class CleanResult:
    """清洗阶段的输出和统计信息。

    这些统计会进入解析报告，方便运维人员查看清洗阶段实际做了哪些处理。
    """

    markdown: str
    removed_header_footer_lines: int = 0
    removed_duplicate_blocks: int = 0
    redacted_sensitive_items: int = 0
    repaired_encoding: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class QualityResult:
    """文档质量校验结果。

    ``score`` 不是学术指标，而是工程上的健康分：
    - 1.0 表示文档很适合入库；
    - 0.0 表示基本不应该进入知识库；
    - warnings/errors 给人工确认页面展示。
    """

    score: float
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    expired: bool = False


@dataclass
class IngestionReport:
    """一次文档入库解析的完整报告。

    报告会同时保存在 pending 目录和 version 目录：
    - pending 报告用于人工确认；
    - version 报告用于后续审计、排障和面试讲解。
    """

    ingestion_id: str
    document_id: str
    source_filename: str
    source_type: str
    status: str
    version: str
    created_at: str
    parser: str
    pages: int
    tables: int
    content_hash: str
    markdown_chars: int
    quality_score: float
    quality_passed: bool
    expired: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cleaning: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """把不同源格式解析成 Markdown。

    这里优先选主流、维护稳定的库：
    - PDF：``pdfplumber``，能抽文本，也能抽取简单表格。
    - Word：``python-docx``，能读取段落和表格。

    依赖缺失时不会静默失败，而是抛出明确错误，便于部署时排查。
    """

    def parse(self, filename: str, raw: bytes) -> ParsedDocument:
        """根据文件后缀选择解析器。"""

        suffix = Path(filename).suffix.lower()
        if suffix == ".md":
            return self._parse_text(filename, raw, source_type="markdown")
        if suffix == ".txt":
            return self._parse_text(filename, raw, source_type="text")
        if suffix == ".pdf":
            return self._parse_pdf(raw)
        if suffix == ".docx":
            return self._parse_docx(raw)
        raise ValueError(f"不支持的知识文档类型：{suffix}")

    def _parse_text(self, filename: str, raw: bytes, source_type: str) -> ParsedDocument:
        """解析 Markdown/TXT。

        TXT 会被包一层标题，避免纯文本进入知识库后没有任何章节锚点。
        """

        text, repaired = TextCleaner.decode_text(raw)
        if source_type == "text":
            title = Path(filename).stem
            text = f"# {title}\n\n{text.strip()}\n"
        warnings = ["检测到非 UTF-8 或疑似乱码，已尝试修复。"] if repaired else []
        return ParsedDocument(
            markdown=text,
            source_type=source_type,
            parser="builtin-text",
            repaired_encoding=repaired,
            warnings=warnings,
        )

    def _parse_pdf(self, raw: bytes) -> ParsedDocument:
        """解析 PDF。

        ``pdfplumber`` 是 RAG 入库里常见的 PDF 解析方案之一：
        - ``extract_text`` 用于正文；
        - ``extract_tables`` 用于简单表格；
        - 每页之间加分页标记，后续清洗器可以识别页眉页脚。

        复杂扫描件 PDF 需要 OCR，本次没有默认引入，避免本地部署依赖过重。
        """

        try:
            import pdfplumber
        except ImportError as exc:  # pragma: no cover - 只有缺依赖时触发
            raise RuntimeError("解析 PDF 需要安装 pdfplumber。") from exc

        pages: list[str] = []
        table_count = 0
        warnings: list[str] = []
        with pdfplumber.open(BytesIO(raw)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                page_parts: list[str] = []
                text = page.extract_text() or ""
                if text.strip():
                    page_parts.append(text.strip())
                tables = page.extract_tables() or []
                for table in tables:
                    markdown_table = MarkdownTableCleaner.table_to_markdown(table)
                    if markdown_table:
                        table_count += 1
                        page_parts.append(markdown_table)
                if not page_parts:
                    warnings.append(f"第 {index} 页未提取到文本，可能是扫描件，需要 OCR。")
                pages.append("\n\n".join(page_parts).strip())
        markdown = "\n\n\f\n\n".join(page for page in pages if page)
        return ParsedDocument(
            markdown=markdown,
            source_type="pdf",
            parser="pdfplumber",
            pages=len(pages),
            tables=table_count,
            warnings=warnings,
        )

    def _parse_docx(self, raw: bytes) -> ParsedDocument:
        """解析 Word ``.docx``。

        Word 文档里的规则经常写在表格中，比如“客户等级 -> SLA -> 处理动作”。
        所以这里不只是抽段落，还会把表格转成 Markdown table，方便后续切片和检索。
        """

        try:
            from docx import Document
        except ImportError as exc:  # pragma: no cover - 只有缺依赖时触发
            raise RuntimeError("解析 Word 需要安装 python-docx。") from exc

        document = Document(BytesIO(raw))
        parts: list[str] = []
        table_count = 0
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
            if "heading 1" in style_name or "标题 1" in style_name:
                parts.append(f"# {text}")
            elif "heading 2" in style_name or "标题 2" in style_name:
                parts.append(f"## {text}")
            else:
                parts.append(text)
        for table in document.tables:
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            markdown_table = MarkdownTableCleaner.table_to_markdown(rows)
            if markdown_table:
                table_count += 1
                parts.append(markdown_table)
        return ParsedDocument(
            markdown="\n\n".join(parts),
            source_type="docx",
            parser="python-docx",
            tables=table_count,
        )


class MarkdownTableCleaner:
    """表格清洗工具。

    这里选择 Markdown table 作为统一格式，是因为：
    - 人可以直接审查；
    - Git diff 友好；
    - RAG 切片时表头和单元格能保留在同一段上下文里。
    """

    @staticmethod
    def table_to_markdown(table: list[list[Any]]) -> str:
        """把二维表格转成 Markdown table。"""

        normalized_rows: list[list[str]] = []
        for row in table:
            cells = [MarkdownTableCleaner._clean_cell(cell) for cell in row]
            if any(cells):
                normalized_rows.append(cells)
        if not normalized_rows:
            return ""

        width = max(len(row) for row in normalized_rows)
        padded = [row + [""] * (width - len(row)) for row in normalized_rows]
        header = padded[0]
        separator = ["---"] * width
        body = padded[1:] or [[""] * width]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(lines)

    @staticmethod
    def _clean_cell(value: Any) -> str:
        """清洗单元格。

        单元格里经常有换行、制表符、多个空格；直接进 Markdown 会破坏表格结构。
        """

        text = "" if value is None else str(value)
        text = re.sub(r"\s+", " ", text.replace("|", "\\|")).strip()
        return text


class TextCleaner:
    """解析后的 Markdown 清洗器。

    这些清洗动作都尽量保持“保守”：
    - 删除明显无意义的页眉页脚和页码；
    - 修复常见编码问题；
    - 去掉重复段落；
    - 对敏感信息做脱敏；
    - 不改写业务规则含义。
    """

    # 常见敏感信息正则。进入向量库前脱敏，是企业 RAG 的基本安全边界。
    SENSITIVE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[EMAIL]"),
        (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE]"),
        (re.compile(r"\b\d{17}[\dXx]\b"), "[ID_CARD]"),
        (re.compile(r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+", re.I), "[SECRET]"),
    )

    @staticmethod
    def decode_text(raw: bytes) -> tuple[str, bool]:
        """把字节流解码成文本，并尽量修复乱码。

        返回值第二项表示是否发生过“非标准解码/乱码修复”，会写入报告。
        """

        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = raw.decode(encoding)
                repaired = encoding not in {"utf-8", "utf-8-sig"}
                return TextCleaner._repair_mojibake(text), repaired
            except UnicodeDecodeError:
                continue

        try:
            from charset_normalizer import from_bytes

            best = from_bytes(raw).best()
            if best is not None:
                return TextCleaner._repair_mojibake(str(best)), True
        except Exception:
            pass
        return raw.decode("utf-8", errors="replace"), True

    @staticmethod
    def clean(markdown: str) -> CleanResult:
        """执行完整清洗流程。"""

        result = CleanResult(markdown=markdown)
        text = TextCleaner._normalize_text(markdown)
        text, header_footer_removed = TextCleaner._remove_repeated_headers_and_footers(text)
        result.removed_header_footer_lines = header_footer_removed
        text, duplicate_removed = TextCleaner._dedupe_blocks(text)
        result.removed_duplicate_blocks = duplicate_removed
        text, redacted = TextCleaner._redact_sensitive(text)
        result.redacted_sensitive_items = redacted
        text = TextCleaner._normalize_markdown_spacing(text)
        result.markdown = text.strip() + "\n" if text.strip() else ""
        return result

    @staticmethod
    def _repair_mojibake(text: str) -> str:
        """修复一类常见 mojibake。

        例如 UTF-8 文本被错误当成 latin1 打开时，中文或符号会出现 ``Ã``、``Â``。
        这个修复很保守：只有检测到典型 mojibake 标记时才尝试。
        """

        if not any(marker in text for marker in ("Ã", "Â", "â€", "�")):
            return text
        try:
            repaired = text.encode("latin1", errors="ignore").decode("utf-8")
            if repaired.count("�") <= text.count("�"):
                return repaired
        except UnicodeError:
            return text
        return text

    @staticmethod
    def _normalize_text(text: str) -> str:
        """统一换行、去掉控制字符、压缩奇怪空白。"""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = text.replace("\u00a0", " ")
        return text

    @staticmethod
    def _remove_repeated_headers_and_footers(text: str) -> tuple[str, int]:
        """删除 PDF 多页重复出现的页眉页脚。

        PDF 解析后会用 ``\f`` 标记分页。我们统计每页首尾两行，如果同一行在多页重复出现，
        就把它视为页眉/页脚候选并删除。
        """

        pages = [page for page in text.split("\f") if page.strip()]
        if len(pages) < 2:
            return text, 0

        candidates: dict[str, int] = {}
        page_lines: list[list[str]] = []
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            page_lines.append(lines)
            for line in [*lines[:2], *lines[-2:]]:
                normalized = TextCleaner._line_key(line)
                if normalized:
                    candidates[normalized] = candidates.get(normalized, 0) + 1

        threshold = max(2, int(len(pages) * 0.6))
        repeated = {line for line, count in candidates.items() if count >= threshold}
        removed = 0
        cleaned_pages: list[str] = []
        for lines in page_lines:
            kept: list[str] = []
            for line in lines:
                if TextCleaner._line_key(line) in repeated or re.fullmatch(r"(第\s*)?\d+\s*(页|/ \d+)?", line):
                    removed += 1
                    continue
                kept.append(line)
            cleaned_pages.append("\n".join(kept))
        return "\n\n".join(cleaned_pages), removed

    @staticmethod
    def _line_key(line: str) -> str:
        """页眉页脚候选归一化。"""

        key = re.sub(r"\d+", "{n}", line.strip().lower())
        key = re.sub(r"\s+", " ", key)
        return key if 4 <= len(key) <= 80 else ""

    @staticmethod
    def _dedupe_blocks(text: str) -> tuple[str, int]:
        """删除重复段落。

        这里按空行分块，而不是逐行去重。逐行去重容易误删表格里的相同值；
        按块去重更适合清理“免责声明、页脚、重复目录段落”。
        """

        blocks = re.split(r"\n{2,}", text)
        seen: set[str] = set()
        kept: list[str] = []
        removed = 0
        for block in blocks:
            normalized = re.sub(r"\s+", " ", block).strip().lower()
            if not normalized:
                continue
            # 中文规则文档里的重复声明往往很短，例如“本文档仅供内部使用，请勿外传”。
            # 阈值设为 8 可以清理这类重复块，同时仍然避免把表格里的单字/短值误删。
            if len(normalized) > 8 and normalized in seen:
                removed += 1
                continue
            seen.add(normalized)
            kept.append(block.strip())
        return "\n\n".join(kept), removed

    @staticmethod
    def _redact_sensitive(text: str) -> tuple[str, int]:
        """脱敏邮箱、手机号、身份证号、密钥等敏感信息。"""

        total = 0
        for pattern, replacement in TextCleaner.SENSITIVE_PATTERNS:
            text, count = pattern.subn(replacement, text)
            total += count
        return text, total

    @staticmethod
    def _normalize_markdown_spacing(text: str) -> str:
        """规范 Markdown 空行，让后续切片更稳定。"""

        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text


class DocumentQualityChecker:
    """文档质量校验器。

    质检不是为了阻止所有低质量文档，而是把风险显式暴露给人工确认流程。
    只有“空文档、极短文档、已过期文档”这类硬问题会判为不通过。
    """

    MIN_CHARS = 80

    def check(self, markdown: str, metadata: dict[str, Any], parser_warnings: list[str]) -> QualityResult:
        """返回质量评分、警告和错误。"""

        warnings = list(parser_warnings)
        errors: list[str] = []
        score = 1.0
        stripped = markdown.strip()

        if not stripped:
            errors.append("文档正文为空，不能进入知识库。")
            score -= 0.8
        elif len(stripped) < self.MIN_CHARS:
            errors.append(f"文档正文过短，少于 {self.MIN_CHARS} 个字符。")
            score -= 0.4

        if "#" not in stripped:
            warnings.append("文档缺少 Markdown 标题，建议人工补充章节结构。")
            score -= 0.1

        if "category" not in metadata or not str(metadata.get("category") or "").strip():
            warnings.append("缺少 category metadata，系统会退化为按文件名推断分类。")
            score -= 0.08

        replacement_ratio = stripped.count("�") / max(len(stripped), 1)
        if replacement_ratio > 0.01:
            warnings.append("文档包含较多替换字符，可能仍有乱码。")
            score -= 0.2

        expired = self._is_expired(metadata.get("expires_at"))
        if expired:
            errors.append("文档 expires_at 已过期，默认不建议进入可检索知识库。")
            score -= 0.5

        score = max(round(score, 2), 0.0)
        return QualityResult(
            score=score,
            passed=not errors and score >= 0.55,
            warnings=warnings,
            errors=errors,
            expired=expired,
        )

    @staticmethod
    def _is_expired(value: Any) -> bool:
        """判断 front matter 中的 ``expires_at`` 是否已经过期。"""

        if not value:
            return False
        try:
            return date.fromisoformat(str(value)[:10]) < datetime.now(timezone.utc).date()
        except ValueError:
            return False


class DocumentIngestionService:
    """文档入库应用服务。

    存储布局：
    ``storage/knowledge_ingestion/pending/{ingestion_id}``
        人工确认前的 Markdown、原始文件和 report。

    ``storage/knowledge_ingestion/versions/{document_id}/{version}``
        已确认入库的版本快照和 report。

    ``app/data/knowledge/{document_id}.md``
        RAG 正式知识目录，只有这里的 Markdown 会被现有索引重建流程读取。
    """

    def __init__(self, knowledge_dir: str, storage_dir: str | Path | None = None) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.storage_dir = Path(storage_dir or Path("storage") / "knowledge_ingestion")
        self.pending_dir = self.storage_dir / "pending"
        self.versions_dir = self.storage_dir / "versions"
        self.parser = DocumentParser()
        self.quality_checker = DocumentQualityChecker()

    def prepare_upload(
        self,
        *,
        filename: str,
        raw: bytes,
        document_id: str,
        metadata: dict[str, Any],
        require_confirmation: bool,
        replace_existing: bool,
        allow_low_quality_direct_commit: bool = False,
    ) -> IngestionReport:
        """解析、清洗、质检，并根据参数决定是否进入 pending。

        这个方法不直接触发 RAG rebuild；路由层在确认提交后统一调度后台重建。
        """

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SOURCE_SUFFIXES:
            raise ValueError("仅支持上传 .md、.txt、.pdf 或 .docx 知识文档。")

        parsed = self.parser.parse(filename, raw)
        cleaned = TextCleaner.clean(parsed.markdown)
        metadata = self._normalize_metadata(metadata, filename)
        markdown = self._inject_front_matter(cleaned.markdown, document_id, metadata)
        quality = self.quality_checker.check(markdown, metadata, parsed.warnings + cleaned.warnings)
        ingestion_id = self._new_ingestion_id(document_id, raw)
        version = str(metadata.get("version") or self._new_version())
        # 默认仍保留治理保护；显式直入库的调用方可以把质检问题作为 warning 返回。
        force_pending = require_confirmation or (not quality.passed and not allow_low_quality_direct_commit)
        if not quality.passed and not require_confirmation and not allow_low_quality_direct_commit:
            quality.warnings.append("质检未通过，系统已强制转入人工确认流程。")
        elif not quality.passed and allow_low_quality_direct_commit:
            quality.warnings.append("质检未通过，但当前入口配置为跳过人工确认，已直接入库。")
        status = "pending_confirmation" if force_pending else "ready_to_commit"
        report = IngestionReport(
            ingestion_id=ingestion_id,
            document_id=document_id,
            source_filename=filename,
            source_type=parsed.source_type,
            status=status,
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            parser=parsed.parser,
            pages=parsed.pages,
            tables=parsed.tables,
            content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            markdown_chars=len(markdown),
            quality_score=quality.score,
            quality_passed=quality.passed,
            expired=quality.expired,
            warnings=quality.warnings,
            errors=quality.errors,
            cleaning={
                "removed_header_footer_lines": cleaned.removed_header_footer_lines,
                "removed_duplicate_blocks": cleaned.removed_duplicate_blocks,
                "redacted_sensitive_items": cleaned.redacted_sensitive_items,
                "repaired_encoding": parsed.repaired_encoding or cleaned.repaired_encoding,
            },
            metadata=metadata,
        )

        if force_pending:
            self._write_pending(report, markdown, raw)
        else:
            self.commit_markdown(
                document_id=document_id,
                markdown=markdown,
                report=report,
                replace_existing=replace_existing,
            )
            report.status = "indexed_source_ready"
            self._write_json(self._version_path(document_id, report.version) / "report.json", asdict(report))
        return report

    def list_pending(self) -> list[dict[str, Any]]:
        """列出所有等待人工确认的解析任务。"""

        reports: list[dict[str, Any]] = []
        if not self.pending_dir.exists():
            return reports
        for report_path in sorted(self.pending_dir.glob("*/report.json")):
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        return reports

    def get_pending(self, ingestion_id: str) -> dict[str, Any]:
        """读取 pending 任务详情，包含 Markdown 预览。"""

        pending_path = self._pending_path(ingestion_id)
        report_path = pending_path / "report.json"
        markdown_path = pending_path / "content.md"
        if not report_path.exists() or not markdown_path.exists():
            raise FileNotFoundError(f"待确认解析任务不存在：{ingestion_id}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")
        report["markdown_preview"] = markdown[:4000]
        report["markdown_total_chars"] = len(markdown)
        return report

    def confirm_pending(self, ingestion_id: str, *, replace_existing: bool) -> dict[str, Any]:
        """人工确认 pending 任务，并写入正式知识目录。"""

        pending_path = self._pending_path(ingestion_id)
        report_path = pending_path / "report.json"
        markdown_path = pending_path / "content.md"
        if not report_path.exists() or not markdown_path.exists():
            raise FileNotFoundError(f"待确认解析任务不存在：{ingestion_id}")

        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")
        document_id = str(report_data["document_id"])
        report = IngestionReport(**report_data)
        existed = self.commit_markdown(
            document_id=document_id,
            markdown=markdown,
            report=report,
            replace_existing=replace_existing,
        )
        report.status = "confirmed"
        self._write_json(self._version_path(document_id, report.version) / "report.json", asdict(report))
        shutil.rmtree(pending_path)
        return {
            "document_id": document_id,
            "filename": f"{document_id}.md",
            "version": report.version,
            "replaced": existed,
        }

    def reject_pending(self, ingestion_id: str) -> None:
        """拒绝 pending 任务并删除暂存文件。"""

        pending_path = self._pending_path(ingestion_id)
        if not pending_path.exists():
            raise FileNotFoundError(f"待确认解析任务不存在：{ingestion_id}")
        shutil.rmtree(pending_path)

    def list_versions(self, document_id: str) -> list[dict[str, Any]]:
        """列出某个文档的所有版本报告。"""

        root = self.versions_dir / document_id
        if not root.exists():
            return []
        versions: list[dict[str, Any]] = []
        for report_path in sorted(root.glob("*/report.json"), reverse=True):
            versions.append(json.loads(report_path.read_text(encoding="utf-8")))
        return versions

    def commit_markdown(
        self,
        *,
        document_id: str,
        markdown: str,
        report: IngestionReport,
        replace_existing: bool,
    ) -> bool:
        """把确认后的 Markdown 写入正式知识目录，并保存版本快照。

        返回值表示正式文档是否已经存在，路由层用它生成响应。
        """

        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        target = self.knowledge_dir / f"{document_id}.md"
        existed = target.exists()
        if existed and not replace_existing:
            raise FileExistsError(f"知识文档已存在：{document_id}")

        temp = target.with_suffix(".md.tmp")
        temp.write_text(markdown, encoding="utf-8")
        temp.replace(target)

        version_path = self._version_path(document_id, report.version)
        version_path.mkdir(parents=True, exist_ok=True)
        (version_path / "content.md").write_text(markdown, encoding="utf-8")
        self._write_json(version_path / "report.json", asdict(report))
        return existed

    def _write_pending(self, report: IngestionReport, markdown: str, raw: bytes) -> None:
        """写入 pending 目录，等待人工确认。"""

        pending_path = self._pending_path(report.ingestion_id)
        pending_path.mkdir(parents=True, exist_ok=True)
        (pending_path / "content.md").write_text(markdown, encoding="utf-8")
        (pending_path / "source.bin").write_bytes(raw)
        self._write_json(pending_path / "report.json", asdict(report))

    def _normalize_metadata(self, metadata: dict[str, Any], filename: str) -> dict[str, Any]:
        """补齐上传时缺省的 metadata。"""

        normalized = {key: value for key, value in metadata.items() if value not in (None, "")}
        normalized.setdefault("title", Path(filename).stem)
        normalized.setdefault("source_filename", filename)
        return normalized

    def _inject_front_matter(self, markdown: str, document_id: str, metadata: dict[str, Any]) -> str:
        """给 Markdown 注入 front matter。

        现有 ``KnowledgeFrontMatterExtractor`` 会读取 front matter，
        所以这里把上传表单里的 category/version/expires_at 等治理字段写进去。
        """

        if markdown.startswith("---\n"):
            return markdown
        frontmatter = {"document_id": document_id, **metadata}
        lines = ["---"]
        for key, value in frontmatter.items():
            if isinstance(value, list):
                rendered = "[" + ", ".join(str(item) for item in value) + "]"
            else:
                rendered = str(value)
            lines.append(f"{key}: {rendered}")
        lines.append("---")
        return "\n".join(lines) + "\n\n" + markdown

    def _pending_path(self, ingestion_id: str) -> Path:
        return self.pending_dir / ingestion_id

    def _version_path(self, document_id: str, version: str) -> Path:
        return self.versions_dir / document_id / version

    @staticmethod
    def _new_ingestion_id(document_id: str, raw: bytes) -> str:
        digest = hashlib.sha1(raw).hexdigest()[:10]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{document_id}-{stamp}-{digest}"

    @staticmethod
    def _new_version() -> str:
        return datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
