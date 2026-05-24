"""Enterprise data repository.

Learning notes:
- Stores structured orders and inventory imported from ERP/OMS/WMS or offline files.
- Current storage is local JSON for demo purposes; the interface can later map to SQL/API storage.
- Orders and inventory are structured business data, not RAG documents.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from app.repositories.inventory_repository import InventoryRepository
from app.repositories.order_repository import OrderRepository
from app.schemas.enterprise_data import (
    EnterpriseDataSourceCreate,
    EnterpriseDataSourceInfo,
    EnterpriseDataStats,
)
from app.schemas.inventory import InventoryRecord
from app.schemas.orders import OrderRecord


class EnterpriseDataRepository(OrderRepository, InventoryRepository):
    """企业业务数据的本地持久化仓库。

    这层不要和 RAG 混在一起：
    - RAG 适合存制度、规则、说明文档这类非结构化知识。
    - 订单、库存是强结构化业务数据，应该走仓库接口，被服务层直接查询。

    当前实现用 JSON 文件落盘，方便面试演示和本地调试；接口边界已经按仓库层
    设计好，后续替换成 MySQL/ERP API 时，上层服务基本不用改。
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._sources_file = self.data_dir / "sources.json"
        self._orders_file = self.data_dir / "orders.json"
        self._inventory_file = self.data_dir / "inventory.json"
        self._ensure_files()

    def create_source(
        self,
        request: EnterpriseDataSourceCreate,
        *,
        allow_update: bool = False,
    ) -> EnterpriseDataSourceInfo:
        """登记一个企业数据源。

        allow_update 给后台配置页使用：同一个 source_id 再提交时可以更新名称、
        描述和连接配置，而不会误删已经导入的业务数据。
        """

        with self._lock:
            sources = self._read_sources()
            source_id = (
                self.normalize_source_id(request.source_id)
                if request.source_id is not None
                else self._build_source_id(request.name, sources)
            )

            now = datetime.now(timezone.utc)
            if source_id in sources and not allow_update:
                raise ValueError(f"数据源已存在：{source_id}")

            existing = sources.get(source_id)
            info = EnterpriseDataSourceInfo(
                source_id=source_id,
                name=request.name,
                source_type=request.source_type,
                description=request.description,
                config=request.config,
                enabled=request.enabled,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            sources[source_id] = self._attach_counts(info)
            self._write_sources(sources)
            return sources[source_id]

    def list_sources(self) -> list[EnterpriseDataSourceInfo]:
        with self._lock:
            sources = self._read_sources()
            return [self._attach_counts(source) for source in sources.values()]

    def delete_source(self, source_id: str) -> bool:
        """删除数据源，同时删除该来源下已经导入的订单和库存。"""

        with self._lock:
            source_id = self.normalize_source_id(source_id)
            sources = self._read_sources()
            if source_id not in sources:
                return False

            sources.pop(source_id)
            orders = {
                order_id: entry
                for order_id, entry in self._read_orders().items()
                if entry["source_id"] != source_id
            }
            inventory = self._remove_inventory_by_source(source_id)

            self._write_sources(sources)
            self._write_json(self._orders_file, orders)
            self._write_json(self._inventory_file, inventory)
            return True

    def import_orders(
        self,
        source_id: str,
        orders: list[OrderRecord],
        *,
        replace_source: bool = False,
    ) -> int:
        """导入企业订单。

        同一个 order_id 后导入会覆盖先导入的数据，这符合大多数后台同步语义：
        企业系统里的订单状态、优先级、明细可能发生变化，分析时应取最新快照。
        """

        with self._lock:
            source_id = self.normalize_source_id(source_id)
            self._ensure_source_exists(source_id)
            current = self._read_orders()
            if replace_source:
                current = {
                    order_id: entry
                    for order_id, entry in current.items()
                    if entry["source_id"] != source_id
                }

            for order in orders:
                current[order.order_id] = {
                    "source_id": source_id,
                    "record": order.model_dump(mode="json"),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }

            self._write_json(self._orders_file, current)
            self._refresh_source_counts()
            return len(orders)

    def import_inventory(
        self,
        source_id: str,
        records: list[InventoryRecord],
        *,
        replace_source: bool = False,
    ) -> int:
        """导入企业库存快照。"""

        with self._lock:
            source_id = self.normalize_source_id(source_id)
            self._ensure_source_exists(source_id)
            inventory = self._read_inventory()
            if replace_source:
                inventory = self._remove_inventory_by_source(source_id)

            for record in records:
                inventory.setdefault(record.sku_id, [])
                inventory[record.sku_id] = [
                    entry
                    for entry in inventory[record.sku_id]
                    if not (
                        entry["source_id"] == source_id
                        and entry["record"]["warehouse_id"] == record.warehouse_id
                    )
                ]
                inventory[record.sku_id].append(
                    {
                        "source_id": source_id,
                        "record": record.model_dump(mode="json"),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            self._write_json(self._inventory_file, inventory)
            self._refresh_source_counts()
            return len(records)

    def get_order_by_id(self, order_id: str) -> OrderRecord | None:
        with self._lock:
            entry = self._read_orders().get(order_id)
            if entry is None:
                return None
            return OrderRecord.model_validate(entry["record"])

    def list_inventory_by_sku(self, sku_id: str) -> list[InventoryRecord]:
        with self._lock:
            entries = self._read_inventory().get(sku_id, [])
            return [InventoryRecord.model_validate(entry["record"]) for entry in entries]

    def stats(self) -> EnterpriseDataStats:
        with self._lock:
            orders = self._read_orders()
            inventory = self._read_inventory()
            return EnterpriseDataStats(
                source_count=len(self._read_sources()),
                order_count=len(orders),
                inventory_record_count=sum(len(entries) for entries in inventory.values()),
                inventory_sku_count=len(inventory),
            )

    def _ensure_files(self) -> None:
        if not self._sources_file.exists():
            manual = EnterpriseDataSourceInfo(
                source_id="manual",
                name="手动导入",
                source_type="manual",
                description="后台手动 JSON 导入的数据源。",
            )
            self._write_json(self._sources_file, {"manual": manual.model_dump(mode="json")})
        if not self._orders_file.exists():
            self._write_json(self._orders_file, {})
        if not self._inventory_file.exists():
            self._write_json(self._inventory_file, {})

    def _ensure_source_exists(self, source_id: str) -> None:
        source_id = self.normalize_source_id(source_id)
        sources = self._read_sources()
        if source_id in sources:
            return
        self.create_source(
            EnterpriseDataSourceCreate(
                source_id=source_id,
                name=source_id,
                source_type="manual",
                description="导入时自动创建的数据源。",
            ),
            allow_update=True,
        )

    def _read_sources(self) -> dict[str, EnterpriseDataSourceInfo]:
        raw = self._read_json(self._sources_file)
        return {
            source_id: EnterpriseDataSourceInfo.model_validate(payload)
            for source_id, payload in raw.items()
        }

    def _write_sources(self, sources: dict[str, EnterpriseDataSourceInfo]) -> None:
        self._write_json(
            self._sources_file,
            {
                source_id: source.model_dump(mode="json")
                for source_id, source in sources.items()
            },
        )

    def _read_orders(self) -> dict[str, dict]:
        return self._read_json(self._orders_file)

    def _read_inventory(self) -> dict[str, list[dict]]:
        return self._read_json(self._inventory_file)

    def _read_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    def _build_source_id(
        self,
        name: str,
        sources: dict[str, EnterpriseDataSourceInfo],
    ) -> str:
        base = "-".join(name.strip().lower().split()) or "source"
        candidate = base
        index = 2
        while candidate in sources:
            candidate = f"{base}-{index}"
            index += 1
        return candidate

    def normalize_source_id(self, source_id: str | None) -> str:
        if source_id is None:
            return "manual"
        normalized = source_id.strip().lower().replace(" ", "-")
        return normalized or "manual"

    def _remove_inventory_by_source(self, source_id: str) -> dict[str, list[dict]]:
        inventory = self._read_inventory()
        cleaned: dict[str, list[dict]] = {}
        for sku_id, entries in inventory.items():
            kept = [entry for entry in entries if entry["source_id"] != source_id]
            if kept:
                cleaned[sku_id] = kept
        return cleaned

    def _attach_counts(self, source: EnterpriseDataSourceInfo) -> EnterpriseDataSourceInfo:
        orders = self._read_orders()
        inventory = self._read_inventory()
        return source.model_copy(
            update={
                "order_count": sum(
                    1 for entry in orders.values() if entry["source_id"] == source.source_id
                ),
                "inventory_record_count": sum(
                    1
                    for entries in inventory.values()
                    for entry in entries
                    if entry["source_id"] == source.source_id
                ),
            }
        )

    def _refresh_source_counts(self) -> None:
        sources = self._read_sources()
        self._write_sources(
            {
                source_id: self._attach_counts(source)
                for source_id, source in sources.items()
            }
        )
