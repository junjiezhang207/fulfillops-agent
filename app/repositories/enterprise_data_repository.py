"""企业结构化数据 PostgreSQL 仓储。

生产边界：
- 订单和库存是实时/准实时结构化业务数据，不能放进 RAG。
- 本仓储只读写 PostgreSQL，不再使用本地 JSON 文件。
- 没有查到订单或库存时直接返回空结果，不再回退 demo 数据。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.repositories.inventory_repository import InventoryRepository
from app.repositories.order_repository import OrderRepository
from app.schemas.enterprise_data import (
    EnterpriseDataSourceCreate,
    EnterpriseDataSourceInfo,
    EnterpriseDataStats,
)
from app.schemas.inventory import InventoryRecord
from app.schemas.orders import OrderRecord


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _json_load(payload: str | bytes | None) -> Any:
    if not payload:
        return {}
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


class EnterpriseDataRepository(OrderRepository, InventoryRepository):
    """企业订单和库存的 PostgreSQL 仓储。

    这里是生产版数据入口。上层 Workflow、Agent 和 RAG query planning 都只依赖
    OrderRepository / InventoryRepository 接口，不关心数据来自 OMS、WMS 还是导入表。
    """

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise RuntimeError("企业数据仓储必须配置 DATABASE_URL/POSTGRES_URL，生产模式不允许使用本地文件。")
        self.database_url = database_url
        self._engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=1800, future=True)
        self._init_schema()

    @property
    def engine(self) -> Engine:
        return self._engine

    def create_source(
        self,
        request: EnterpriseDataSourceCreate,
        *,
        allow_update: bool = False,
    ) -> EnterpriseDataSourceInfo:
        source_id = (
            self.normalize_source_id(request.source_id)
            if request.source_id is not None
            else self._build_source_id(request.name)
        )
        now = _utc_now()
        existing = self._get_source_row(source_id)
        if existing and not allow_update:
            raise ValueError(f"数据源已存在：{source_id}")

        with self._engine.begin() as conn:
            if existing:
                conn.execute(
                    text(
                        """
                        UPDATE enterprise_data_sources
                        SET name=:name, source_type=:source_type, description=:description,
                            config_json=:config_json, enabled=:enabled, updated_at=:updated_at
                        WHERE source_id=:source_id
                        """
                    ),
                    {
                        "source_id": source_id,
                        "name": request.name,
                        "source_type": str(request.source_type),
                        "description": request.description,
                        "config_json": _json_dump(request.config),
                        "enabled": bool(request.enabled),
                        "updated_at": now,
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO enterprise_data_sources (
                            source_id, name, source_type, description, config_json,
                            enabled, created_at, updated_at
                        ) VALUES (
                            :source_id, :name, :source_type, :description, :config_json,
                            :enabled, :created_at, :updated_at
                        )
                        """
                    ),
                    {
                        "source_id": source_id,
                        "name": request.name,
                        "source_type": str(request.source_type),
                        "description": request.description,
                        "config_json": _json_dump(request.config),
                        "enabled": bool(request.enabled),
                        "created_at": now,
                        "updated_at": now,
                    },
                )
        return self._attach_counts(self._row_to_source(self._get_source_row(source_id)))

    def list_sources(self) -> list[EnterpriseDataSourceInfo]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM enterprise_data_sources ORDER BY created_at DESC")
            ).mappings().all()
        return [self._attach_counts(self._row_to_source(row)) for row in rows]

    def delete_source(self, source_id: str) -> bool:
        source_id = self.normalize_source_id(source_id)
        with self._engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM enterprise_data_sources WHERE source_id=:source_id"),
                {"source_id": source_id},
            )
        return bool(result.rowcount)

    def import_orders(
        self,
        source_id: str,
        orders: list[OrderRecord],
        *,
        replace_source: bool = False,
    ) -> int:
        source_id = self.normalize_source_id(source_id)
        self._ensure_source_exists(source_id)
        now = _utc_now()
        with self._engine.begin() as conn:
            if replace_source:
                conn.execute(text("DELETE FROM enterprise_orders WHERE source_id=:source_id"), {"source_id": source_id})
            for order in orders:
                conn.execute(
                    text(
                        """
                        INSERT INTO enterprise_orders (
                            order_id, source_id, platform, order_time, order_status,
                            region, priority, record_json, updated_at
                        ) VALUES (
                            :order_id, :source_id, :platform, :order_time, :order_status,
                            :region, :priority, :record_json, :updated_at
                        )
                        ON CONFLICT (order_id) DO UPDATE SET
                            source_id=EXCLUDED.source_id,
                            platform=EXCLUDED.platform,
                            order_time=EXCLUDED.order_time,
                            order_status=EXCLUDED.order_status,
                            region=EXCLUDED.region,
                            priority=EXCLUDED.priority,
                            record_json=EXCLUDED.record_json,
                            updated_at=EXCLUDED.updated_at
                        """
                    ),
                    {
                        "order_id": order.order_id,
                        "source_id": source_id,
                        "platform": order.platform,
                        "order_time": order.order_time,
                        "order_status": order.order_status,
                        "region": order.region,
                        "priority": order.priority,
                        "record_json": _json_dump(order.model_dump(mode="json")),
                        "updated_at": now,
                    },
                )
        return len(orders)

    def import_inventory(
        self,
        source_id: str,
        records: list[InventoryRecord],
        *,
        replace_source: bool = False,
    ) -> int:
        source_id = self.normalize_source_id(source_id)
        self._ensure_source_exists(source_id)
        now = _utc_now()
        with self._engine.begin() as conn:
            if replace_source:
                conn.execute(
                    text("DELETE FROM enterprise_inventory WHERE source_id=:source_id"),
                    {"source_id": source_id},
                )
            for record in records:
                conn.execute(
                    text(
                        """
                        INSERT INTO enterprise_inventory (
                            source_id, warehouse_id, warehouse_name, region, sku_id,
                            available_stock, locked_stock, record_json, updated_at
                        ) VALUES (
                            :source_id, :warehouse_id, :warehouse_name, :region, :sku_id,
                            :available_stock, :locked_stock, :record_json, :updated_at
                        )
                        ON CONFLICT (source_id, warehouse_id, sku_id) DO UPDATE SET
                            warehouse_name=EXCLUDED.warehouse_name,
                            region=EXCLUDED.region,
                            available_stock=EXCLUDED.available_stock,
                            locked_stock=EXCLUDED.locked_stock,
                            record_json=EXCLUDED.record_json,
                            updated_at=EXCLUDED.updated_at
                        """
                    ),
                    {
                        "source_id": source_id,
                        "warehouse_id": record.warehouse_id,
                        "warehouse_name": record.warehouse_name,
                        "region": record.region,
                        "sku_id": record.sku_id,
                        "available_stock": record.available_stock,
                        "locked_stock": record.locked_stock,
                        "record_json": _json_dump(record.model_dump(mode="json")),
                        "updated_at": now,
                    },
                )
        return len(records)

    def get_order_by_id(self, order_id: str) -> OrderRecord | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT record_json FROM enterprise_orders WHERE order_id=:order_id"),
                {"order_id": order_id},
            ).mappings().first()
        if row is None:
            return None
        return OrderRecord.model_validate(_json_load(row["record_json"]))

    def list_orders(self, limit: int = 100) -> list[OrderRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT record_json FROM enterprise_orders ORDER BY updated_at DESC LIMIT :limit"),
                {"limit": int(limit)},
            ).mappings().all()
        return [OrderRecord.model_validate(_json_load(row["record_json"])) for row in rows]

    def list_inventory_by_sku(self, sku_id: str) -> list[InventoryRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT record_json FROM enterprise_inventory
                    WHERE sku_id=:sku_id
                    ORDER BY updated_at DESC
                    """
                ),
                {"sku_id": sku_id},
            ).mappings().all()
        return [InventoryRecord.model_validate(_json_load(row["record_json"])) for row in rows]

    def stats(self) -> EnterpriseDataStats:
        with self._engine.connect() as conn:
            source_count = conn.execute(text("SELECT COUNT(*) FROM enterprise_data_sources")).scalar_one()
            order_count = conn.execute(text("SELECT COUNT(*) FROM enterprise_orders")).scalar_one()
            inventory_count = conn.execute(text("SELECT COUNT(*) FROM enterprise_inventory")).scalar_one()
            sku_count = conn.execute(text("SELECT COUNT(DISTINCT sku_id) FROM enterprise_inventory")).scalar_one()
        return EnterpriseDataStats(
            source_count=int(source_count),
            order_count=int(order_count),
            inventory_record_count=int(inventory_count),
            inventory_sku_count=int(sku_count),
        )

    def normalize_source_id(self, source_id: str | None) -> str:
        if source_id is None:
            return "manual"
        normalized = source_id.strip().lower().replace(" ", "-")
        return normalized or "manual"

    def _init_schema(self) -> None:
        """初始化生产表结构。

        这里直接在应用启动时建表，便于单体项目部署。更严格的生产环境可以把这些
        DDL 迁移到 Alembic，但运行时仍保持同一套 Repository 接口。
        """
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS enterprise_data_sources (
                    source_id VARCHAR(128) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    source_type VARCHAR(64) NOT NULL,
                    description TEXT,
                    config_json JSONB NOT NULL,
                    enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS enterprise_orders (
                    order_id VARCHAR(128) PRIMARY KEY,
                    source_id VARCHAR(128) NOT NULL,
                    platform VARCHAR(128),
                    order_time TIMESTAMP,
                    order_status VARCHAR(128),
                    region VARCHAR(128),
                    priority VARCHAR(64),
                    record_json JSONB NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    CONSTRAINT fk_enterprise_orders_source
                        FOREIGN KEY (source_id) REFERENCES enterprise_data_sources(source_id)
                        ON DELETE CASCADE
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_enterprise_orders_source ON enterprise_orders (source_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_enterprise_orders_updated ON enterprise_orders (updated_at)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS enterprise_inventory (
                    id BIGSERIAL PRIMARY KEY,
                    source_id VARCHAR(128) NOT NULL,
                    warehouse_id VARCHAR(128) NOT NULL,
                    warehouse_name VARCHAR(255),
                    region VARCHAR(128),
                    sku_id VARCHAR(128) NOT NULL,
                    available_stock INT NOT NULL,
                    locked_stock INT NOT NULL,
                    record_json JSONB NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    CONSTRAINT uq_enterprise_inventory_source_warehouse_sku UNIQUE (source_id, warehouse_id, sku_id),
                    CONSTRAINT fk_enterprise_inventory_source
                        FOREIGN KEY (source_id) REFERENCES enterprise_data_sources(source_id)
                        ON DELETE CASCADE
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_enterprise_inventory_sku ON enterprise_inventory (sku_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_enterprise_inventory_updated ON enterprise_inventory (updated_at)"))

    def _ensure_source_exists(self, source_id: str) -> None:
        if self._get_source_row(source_id) is not None:
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

    def _get_source_row(self, source_id: str):
        with self._engine.connect() as conn:
            return conn.execute(
                text("SELECT * FROM enterprise_data_sources WHERE source_id=:source_id"),
                {"source_id": source_id},
            ).mappings().first()

    def _build_source_id(self, name: str) -> str:
        base = "-".join(name.strip().lower().split()) or "source"
        candidate = base
        index = 2
        while self._get_source_row(candidate) is not None:
            candidate = f"{base}-{index}"
            index += 1
        return candidate

    def _row_to_source(self, row) -> EnterpriseDataSourceInfo:
        if row is None:
            raise ValueError("数据源不存在。")
        return EnterpriseDataSourceInfo(
            source_id=row["source_id"],
            name=row["name"],
            source_type=row["source_type"],
            description=row["description"] or "",
            config=_json_load(row["config_json"]),
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _attach_counts(self, source: EnterpriseDataSourceInfo) -> EnterpriseDataSourceInfo:
        with self._engine.connect() as conn:
            order_count = conn.execute(
                text("SELECT COUNT(*) FROM enterprise_orders WHERE source_id=:source_id"),
                {"source_id": source.source_id},
            ).scalar_one()
            inventory_count = conn.execute(
                text("SELECT COUNT(*) FROM enterprise_inventory WHERE source_id=:source_id"),
                {"source_id": source.source_id},
            ).scalar_one()
        return source.model_copy(
            update={
                "order_count": int(order_count),
                "inventory_record_count": int(inventory_count),
            }
        )
