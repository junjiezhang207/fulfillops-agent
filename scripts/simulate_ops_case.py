"""模拟 OMS/WMS/SOP 接入后的运营异常案件分析。

运行：
    python scripts/simulate_ops_case.py

这个脚本不用真实 MySQL，使用内存假仓储模拟订单系统和库存系统，方便快速验证
Agent 的价值边界：它不发货、不锁库、不退款，只产出运营案件材料。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.application.routing.ops_case_service import OpsCaseAnalysisService
from app.domain.inventory.analysis import InventoryAnalysisService
from app.domain.orders.analysis import OrderAnalysisService
from app.schemas.inventory import InventoryRecord
from app.schemas.orders import OrderItem, OrderRecord


class DemoOrderRepository:
    """模拟 OMS 订单库。"""

    def __init__(self) -> None:
        self._orders = {
            "SO-DEMO-OPS-001": OrderRecord(
                order_id="SO-DEMO-OPS-001",
                platform="天猫旗舰店",
                order_time=datetime(2026, 8, 22, 9, 15, 0),
                order_status="paid_waiting_fulfillment",
                region="上海",
                priority="vip_urgent",
                items=[
                    OrderItem(sku_id="SKU-PRO-SET", product_name="高端礼盒套装", quantity=2, unit_price=58_800),
                    OrderItem(sku_id="SKU-GIFT", product_name="赠品包", quantity=1, unit_price=399),
                ],
            )
        }

    def get_order_by_id(self, order_id: str) -> OrderRecord | None:
        return self._orders.get(order_id)


class DemoInventoryRepository:
    """模拟 WMS 库存库。"""

    def __init__(self) -> None:
        updated_at = datetime(2026, 8, 22, 9, 20, 0)
        self._records = {
            "SKU-PRO-SET": [
                InventoryRecord(
                    warehouse_id="WH-SH",
                    warehouse_name="上海中心仓",
                    region="华东",
                    sku_id="SKU-PRO-SET",
                    available_stock=0,
                    locked_stock=3,
                    updated_at=updated_at,
                ),
                InventoryRecord(
                    warehouse_id="WH-GZ",
                    warehouse_name="广州备货仓",
                    region="华南",
                    sku_id="SKU-PRO-SET",
                    available_stock=1,
                    locked_stock=0,
                    updated_at=updated_at,
                ),
            ],
            "SKU-GIFT": [
                InventoryRecord(
                    warehouse_id="WH-SH",
                    warehouse_name="上海中心仓",
                    region="华东",
                    sku_id="SKU-GIFT",
                    available_stock=30,
                    locked_stock=2,
                    updated_at=updated_at,
                )
            ],
        }

    def list_inventory_by_sku(self, sku_id: str) -> list[InventoryRecord]:
        return self._records.get(sku_id, [])


class DemoKnowledgeService:
    """模拟 SOP/RAG 命中结果。"""

    def retrieve(self, **kwargs):
        return SimpleNamespace(
            order_id=kwargs["order_id"],
            answer_summary=SimpleNamespace(
                conclusion="该订单应按缺货 SOP + VIP 急单升级流程处理。",
                key_rules=[
                    "缺货 SKU 先检查同区域仓库，再评估跨仓调拨或替代 SKU。",
                    "VIP 急单不得在未确认客户接受前直接拆单或替代。",
                    "高价值订单涉及延期承诺时，需要运营主管确认并保留审批记录。",
                ],
                suggested_actions=[
                    "仓配运营确认广州备货仓是否可跨仓补足剩余 1 件。",
                    "客服先向客户说明库存缺口，并确认是否接受延期或分批发货。",
                    "运营主管确认是否承诺新时效，避免平台处罚或 VIP 投诉。",
                ],
                coverage_note="命中缺货处理、VIP 急单和高价值订单复核 SOP。",
            ),
            hits=[
                SimpleNamespace(source_file="缺货订单处理-sop.md", category="stockout_rule", score=0.93),
                SimpleNamespace(source_file="高优先级订单处理流程.md", category="priority_rule", score=0.89),
                SimpleNamespace(source_file="风险订单审核制度.md", category="risk_rule", score=0.82),
            ],
        )


def build_service() -> OpsCaseAnalysisService:
    order_service = OrderAnalysisService(DemoOrderRepository())
    inventory_service = InventoryAnalysisService(DemoInventoryRepository(), order_service)
    return OpsCaseAnalysisService(order_service, inventory_service, DemoKnowledgeService())


def main() -> None:
    service = build_service()
    result = service.analyze(
        "SO-DEMO-OPS-001",
        "请把这个订单整理成异常案件摘要、SOP 适配、分流建议、客服沟通草稿和复盘建议。",
    )
    print(result.to_markdown())


if __name__ == "__main__":
    main()
