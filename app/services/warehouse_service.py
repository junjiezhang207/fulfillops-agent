"""仓库库存服务（学习版注释）。

这个文件模拟企业里的“库存可视化/仓库库存查询”能力。
它和 InventoryAnalysisService 的区别：
- InventoryAnalysisService 是按订单判断“够不够履约”。
- WarehouseService 是按 SKU 查看“库存分布在哪些仓库”。

当前数据是内存模拟数据，真实企业里可以替换为 WMS/库存中心查询。
"""

from dataclasses import dataclass


@dataclass
class WarehouseInventory:
    """单个仓库的库存信息。

    这是最小库存明细单位：一个 SKU 在一个仓库里的库存状态。
    """

    # 仓库编号，适合系统内部使用。
    warehouse_id: str
    # 仓库中文名，适合前端展示。
    warehouse_name: str
    # 仓库所在区域，后续可用于就近履约。
    region: str
    # 商品 SKU。
    sku_id: str
    # 可售库存，可以用于新订单履约。
    available_quantity: int
    # 预留库存，通常已经被其他订单占用。
    reserved_quantity: int
    # 总库存 = 可售 + 预留。
    total_quantity: int

    @property
    def can_fulfill(self) -> bool:
        """该仓库是否可对该 SKU 进行单件履约。

        这里判断 > 0 是因为这个属性只表达“是否有货”，
        不是判断能否满足某个具体订单数量。
        """
        return self.available_quantity > 0


@dataclass
class WarehouseSearchResult:
    """仓库库存查询结果。

    search_sku_inventory 的统一返回结构。
    """

    sku_id: str
    total_available: int
    warehouse_list: list[WarehouseInventory]
    summary: str


class WarehouseService:
    """仓库库存查询服务。

    这个服务没有依赖外部 repository，是为了演示方便。
    如果要做成生产级，可以把 _warehouse_data 和 _warehouse_info 抽到仓库层。
    """

    def __init__(self) -> None:
        # 模拟库存表：键是 (sku_id, warehouse_id)，值是 (可用库存, 预留库存)。
        self._warehouse_data = {
            ("SKU-IPHONE-CASE-001", "WH-SH-001"): (20, 5),  # (可用, 预留)
            ("SKU-IPHONE-CASE-001", "WH-GZ-001"): (15, 2),
            ("SKU-IPHONE-CASE-001", "WH-BJ-001"): (8, 1),
            ("SKU-CHARGER-020W-002", "WH-SH-001"): (50, 10),
            ("SKU-CHARGER-020W-002", "WH-GZ-001"): (30, 5),
            ("SKU-CHARGER-020W-002", "WH-BJ-001"): (20, 3),
            ("SKU-BOTTLE-INS-001", "WH-SH-001"): (5, 1),
            ("SKU-BOTTLE-INS-001", "WH-GZ-001"): (12, 2),
            ("SKU-BOTTLE-INS-001", "WH-BJ-001"): (0, 0),
            ("SKU-ROUTER-WIFI7-001", "WH-SH-001"): (3, 1),
            ("SKU-ROUTER-WIFI7-001", "WH-GZ-001"): (2, 0),
            ("SKU-ROUTER-WIFI7-001", "WH-BJ-001"): (4, 1),
            ("SKU-CABLE-TYPEC-003", "WH-SH-001"): (100, 20),
            ("SKU-CABLE-TYPEC-003", "WH-GZ-001"): (80, 15),
            ("SKU-CABLE-TYPEC-003", "WH-BJ-001"): (60, 10),
        }

        # 模拟仓库主数据：仓库 ID -> (仓库 ID, 仓库名, 区域)。
        self._warehouse_info = {
            "WH-SH-001": ("WH-SH-001", "上海金桥仓", "华东-上海"),
            "WH-GZ-001": ("WH-GZ-001", "广州南沙仓", "华南-广州"),
            "WH-BJ-001": ("WH-BJ-001", "北京大兴仓", "华北-北京"),
        }

    def search_sku_inventory(self, sku_id: str) -> WarehouseSearchResult:
        """查询某个SKU在全国仓库的库存分布。

        Args:
            sku_id: 商品SKU编码

        Returns:
            包含各仓库库存详情的结果对象
        """
        warehouse_list: list[WarehouseInventory] = []
        total_available = 0

        # 遍历所有仓库，保证即使某仓没有该 SKU，也会返回 0 库存记录。
        for warehouse_id, warehouse_info in self._warehouse_info.items():
            warehouse_name, region = warehouse_info[1], warehouse_info[2]
            # 没找到库存记录时兜底为 0，表示该仓没有这个 SKU。
            available, reserved = self._warehouse_data.get(
                (sku_id, warehouse_id), (0, 0)
            )
            total_available += available
            # 组装单仓库存明细。
            warehouse_list.append(
                WarehouseInventory(
                    warehouse_id=warehouse_id,
                    warehouse_name=warehouse_name,
                    region=region,
                    sku_id=sku_id,
                    available_quantity=available,
                    reserved_quantity=reserved,
                    total_quantity=available + reserved,
                )
            )

        # 按可用量降序排列，让库存最多的仓库排在最前面。
        warehouse_list.sort(key=lambda w: w.available_quantity, reverse=True)

        # 摘要给前端/Agent 直接展示。
        summary = self._build_summary(sku_id, warehouse_list, total_available)
        return WarehouseSearchResult(
            sku_id=sku_id,
            total_available=total_available,
            warehouse_list=warehouse_list,
            summary=summary,
        )

    def _build_summary(
        self, sku_id: str, warehouse_list: list[WarehouseInventory], total_available: int
    ) -> str:
        """生成库存分布摘要。

        这里输出的是短文本，不替代结构化字段。
        真正做判断时应该读 total_available 和 warehouse_list。
        """
        if total_available == 0:
            return f"SKU {sku_id}：全国仓库均无现货。"

        # 只统计有可售库存的仓库，预留库存不算可履约能力。
        available_warehouses = [w for w in warehouse_list if w.available_quantity > 0]
        if not available_warehouses:
            return f"SKU {sku_id}：全国仓库均无现货。"

        # warehouse_list 已经按可用库存降序排过，所以第一个就是库存最充足仓。
        top_warehouse = available_warehouses[0]
        summary_lines = [
            f"SKU {sku_id} 全国库存：{total_available} 件",
            f"  最充足：{top_warehouse.warehouse_name}（{top_warehouse.region}）{top_warehouse.available_quantity} 件",
        ]

        if len(available_warehouses) > 1:
            summary_lines.append(
                f"  可用仓库：{len(available_warehouses)} 个"
            )

        return "\n".join(summary_lines)
