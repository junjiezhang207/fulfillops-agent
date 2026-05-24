"""替代 SKU 服务（学习版注释）。

这个文件模拟企业缺货场景里的“替代商品推荐”。
典型使用方式：
1. 库存分析发现某个 SKU 缺货。
2. 调用 SubstituteSkuService 查可替代商品。
3. Workflow/Agent 再结合价格、库存、兼容度给运营建议。

当前实现是内存规则表，真实企业里可以替换为商品知识图谱、商品主数据或推荐系统。
"""

from dataclasses import dataclass


@dataclass
class SubstituteSku:
    """替代方案 SKU 信息。"""

    # 替代商品 SKU。
    sku_id: str
    # 替代商品名称。
    product_name: str
    # 单价，用于判断替代方案是否会导致客诉或补差价。
    unit_price: float
    # 0-100，越高表示和原商品越接近。
    compatibility_score: float  # 0-100，兼容度评分
    # 为什么推荐这个替代品，给客服/运营解释用。
    reason: str
    # 替代品当前可用库存。
    available_quantity: int


@dataclass
class SubstituteResult:
    """替代 SKU 查询结果。"""

    original_sku_id: str
    original_product_name: str
    substitutes: list[SubstituteSku]
    summary: str


class SubstituteSkuService:
    """替代 SKU 查询服务。

    这个服务只负责“找替代品并排序”，不负责自动改订单。
    自动替换商品通常需要客户确认或运营审批。
    """

    def __init__(self) -> None:
        # 原 SKU -> [替代 SKU 列表]。
        # 这里写成内存字典是演示数据；生产中通常来自商品主数据或替代关系表。
        self._substitute_map = {
            "SKU-IPHONE-CASE-001": [
                SubstituteSku(
                    sku_id="SKU-PHONE-CASE-BASIC-002",
                    product_name="硅胶手机壳（通用）",
                    unit_price=49.0,
                    compatibility_score=85,
                    reason="材质相同，防护能力相近",
                    available_quantity=50,
                ),
                SubstituteSku(
                    sku_id="SKU-PHONE-CASE-PREMIUM-003",
                    product_name="皮革手机壳（高端）",
                    unit_price=149.0,
                    compatibility_score=90,
                    reason="更高级材质，防护更好，用户通常接受升级",
                    available_quantity=20,
                ),
            ],
            "SKU-CHARGER-020W-002": [
                SubstituteSku(
                    sku_id="SKU-CHARGER-30W-004",
                    product_name="30W 快充头",
                    unit_price=149.0,
                    compatibility_score=95,
                    reason="兼容性强，功率更高，性能升级",
                    available_quantity=80,
                ),
                SubstituteSku(
                    sku_id="SKU-CHARGER-GENERIC-005",
                    product_name="18W 通用充电头",
                    unit_price=79.0,
                    compatibility_score=70,
                    reason="功率略低但广泛兼容",
                    available_quantity=100,
                ),
            ],
            "SKU-BOTTLE-INS-001": [
                SubstituteSku(
                    sku_id="SKU-BOTTLE-GLASS-006",
                    product_name="玻璃保温杯（400ml）",
                    unit_price=159.0,
                    compatibility_score=80,
                    reason="保温性能相当，材质升级",
                    available_quantity=15,
                ),
            ],
            "SKU-ROUTER-WIFI7-001": [
                SubstituteSku(
                    sku_id="SKU-ROUTER-WIFI6-007",
                    product_name="WiFi 6 双频路由器",
                    unit_price=599.0,
                    compatibility_score=75,
                    reason="性能接近，功能完整，价格更优",
                    available_quantity=12,
                ),
            ],
            "SKU-CABLE-TYPEC-003": [
                SubstituteSku(
                    sku_id="SKU-CABLE-TYPEC-FAST-008",
                    product_name="Type-C 快充数据线（2.4A）",
                    unit_price=49.9,
                    compatibility_score=95,
                    reason="性能升级，传输速度更快",
                    available_quantity=200,
                ),
                SubstituteSku(
                    sku_id="SKU-CABLE-MICRO-009",
                    product_name="Micro-USB 数据线",
                    unit_price=29.9,
                    compatibility_score=40,
                    reason="兼容性差，仅推荐给兼容设备用户",
                    available_quantity=150,
                ),
            ],
        }

    def search_substitutes(self, sku_id: str) -> SubstituteResult:
        """查询某个SKU的替代方案。

        Args:
            sku_id: 原始商品SKU编码

        Returns:
            包含替代方案列表的结果对象
        """
        # 没有配置替代关系时返回空列表，不抛异常。
        substitutes = self._substitute_map.get(sku_id, [])

        # 按兼容度降序排列，优先推荐最接近原商品的方案。
        substitutes_sorted = sorted(
            substitutes, key=lambda s: s.compatibility_score, reverse=True
        )

        # 构建给人读的摘要；结构化替代列表仍然会完整返回。
        summary = self._build_summary(sku_id, substitutes_sorted)

        # original_product_name 用映射兜底，便于前端展示“原商品 -> 替代商品”。
        return SubstituteResult(
            original_sku_id=sku_id,
            original_product_name=self._get_product_name(sku_id),
            substitutes=substitutes_sorted,
            summary=summary,
        )

    def _get_product_name(self, sku_id: str) -> str:
        """获取 SKU 的产品名称（硬编码映射）。

        真实项目里这一步通常会查商品主数据服务。
        """
        mapping = {
            "SKU-IPHONE-CASE-001": "磁吸手机壳",
            "SKU-CHARGER-020W-002": "20W 快充头",
            "SKU-BOTTLE-INS-001": "316 不锈钢保温杯",
            "SKU-ROUTER-WIFI7-001": "WiFi 7 路由器",
            "SKU-CABLE-TYPEC-003": "Type-C 数据线",
        }
        return mapping.get(sku_id, sku_id)

    def _build_summary(
        self, sku_id: str, substitutes: list[SubstituteSku]
    ) -> str:
        """生成替代方案摘要。

        摘要最多展示前 3 个，避免 Agent/前端把一长串替代品直接塞给用户。
        """
        if not substitutes:
            return f"SKU {sku_id}：暂无推荐的替代方案。"

        lines = [f"SKU {sku_id} 的替代方案："]
        # 只展示前三个最优方案；完整 substitutes 列表仍然在结构化字段里。
        for sub in substitutes[:3]:  # 只展示前3个
            lines.append(
                f"  • {sub.product_name}"
                f"（￥{sub.unit_price}，兼容度 {sub.compatibility_score}%，现货 {sub.available_quantity}）"
            )
            lines.append(f"    理由：{sub.reason}")

        if len(substitutes) > 3:
            lines.append(f"  ... 还有 {len(substitutes) - 3} 个替代方案")

        return "\n".join(lines)
