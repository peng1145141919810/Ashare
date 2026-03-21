from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import gmtrade.api as gm

from ..models import AccountState, FillRecord, OrderIntent, Position
from ..utils import from_gm_symbol, safe_float, safe_int, to_gm_symbol
from .base import BaseBroker


class GMTradeSimBroker(BaseBroker):
    """掘金仿真券商适配器。

    Args:
        token: 掘金 token。
        account_id: 账户编号。
        account_alias: 账户别名。
        endpoint: 仿真服务地址。
        buy_price_ratio: 买单价格调整比例。
        sell_price_ratio: 卖单价格调整比例。
        order_wait_seconds: 下单后等待回报秒数。
        sell_by_available: 卖出时是否按可卖数量约束。

    Returns:
        GMTradeSimBroker: 券商实例。
    """

    def __init__(
        self,
        token: str,
        account_id: str,
        account_alias: str,
        endpoint: str,
        buy_price_ratio: float,
        sell_price_ratio: float,
        order_wait_seconds: float,
        sell_by_available: bool,
    ) -> None:
        self.token = token
        self.account_id = account_id
        self.account_alias = account_alias
        self.endpoint = endpoint
        self.buy_price_ratio = float(buy_price_ratio)
        self.sell_price_ratio = float(sell_price_ratio)
        self.order_wait_seconds = float(order_wait_seconds)
        self.sell_by_available = bool(sell_by_available)
        self._account = None

    def _login(self):
        """登录掘金账户。

        Args:
            None

        Returns:
            Any: 账户对象。
        """
        if self._account is None:
            gm.set_token(self.token)
            gm.set_endpoint(self.endpoint)
            self._account = gm.account(account_id=self.account_id, account_alias=self.account_alias)
            gm.login(self._account)
        return self._account

    def _pick_attr(self, candidates: List[str]) -> Any:
        """按顺序寻找常量名。

        Args:
            candidates: 候选常量名列表。

        Returns:
            Any: 找到的常量值。
        """
        for name in candidates:
            if hasattr(gm, name):
                return getattr(gm, name)
        raise AttributeError(f"未找到任何可用常量: {candidates}")

    def _get_positions_raw(self):
        """兼容不同 gmtrade 版本的持仓查询函数。

        Args:
            None

        Returns:
            Any: 原始持仓列表。
        """
        if hasattr(gm, "get_positions"):
            return gm.get_positions()
        if hasattr(gm, "get_position"):
            return gm.get_position()
        raise AttributeError("当前 gmtrade.api 中既没有 get_positions，也没有 get_position")

    def load_account_state(self) -> AccountState:
        """加载账户状态。

        Args:
            None

        Returns:
            AccountState: 当前账户状态。
        """
        self._login()
        cash = gm.get_cash()
        positions_raw = self._get_positions_raw()
        positions: List[Position] = []
        for item in positions_raw:
            symbol = from_gm_symbol(getattr(item, "symbol", ""))
            if not symbol:
                continue
            positions.append(
                Position(
                    symbol=symbol,
                    shares=safe_int(getattr(item, "volume", 0)),
                    avg_cost=safe_float(getattr(item, "vwap", 0.0)),
                    last_price=safe_float(getattr(item, "price", getattr(item, "vwap", 0.0))),
                    available_shares=safe_int(getattr(item, "available", getattr(item, "volume", 0))),
                )
            )
        return AccountState(
            account_id=str(getattr(cash, "account_id", self.account_id)),
            cash=safe_float(getattr(cash, "available", 0.0)),
            nav_value=safe_float(getattr(cash, "nav", 0.0), 0.0) or None,
            positions=positions,
        )

    def _build_order_kwargs(self, order: OrderIntent) -> Dict[str, Any]:
        """构建 order_volume 参数。

        Args:
            order: 订单意图。

        Returns:
            Dict[str, Any]: 下单参数。
        """
        self._login()
        side = self._pick_attr(["OrderSide_Buy", "OrderSide_Sell"] if order.side == "BUY" else ["OrderSide_Sell", "OrderSide_Buy"])
        if order.side != "BUY" and hasattr(gm, "OrderSide_Sell"):
            side = getattr(gm, "OrderSide_Sell")
        if order.side == "BUY" and hasattr(gm, "OrderSide_Buy"):
            side = getattr(gm, "OrderSide_Buy")

        order_type = self._pick_attr(["OrderType_Limit", "OrderType_LimitOrder"])
        if order.side == "BUY":
            price = round(float(order.ref_price) * self.buy_price_ratio, 3)
            position_effect = getattr(gm, "PositionEffect_Open", None)
        else:
            price = round(float(order.ref_price) * self.sell_price_ratio, 3)
            position_effect = getattr(gm, "PositionEffect_Close", None)

        kwargs: Dict[str, Any] = {
            "symbol": to_gm_symbol(order.symbol),
            "volume": int(order.delta_shares),
            "side": side,
            "order_type": order_type,
            "price": price,
            "account": self._account,
        }
        if position_effect is not None:
            kwargs["position_effect"] = position_effect
        return kwargs

    def _extract_order_ids(self, order_result: Any) -> List[Tuple[str, str, str]]:
        """从委托返回值中提取订单标识。

        Args:
            order_result: order_volume 返回结果。

        Returns:
            List[Tuple[str, str, str]]: (cl_ord_id, order_id, symbol) 列表。
        """
        result: List[Tuple[str, str, str]] = []
        if isinstance(order_result, list):
            items = order_result
        else:
            items = [order_result]
        for item in items:
            result.append(
                (
                    str(getattr(item, "cl_ord_id", "")),
                    str(getattr(item, "order_id", "")),
                    from_gm_symbol(getattr(item, "symbol", "")),
                )
            )
        return result

    def execute_orders(
        self,
        order_intents: List[OrderIntent],
        price_map: Dict[str, float],
    ) -> Tuple[AccountState, List[FillRecord], List[dict]]:
        """执行订单并回收成交回报。

        Args:
            order_intents: 订单意图列表。
            price_map: 最新价格映射。

        Returns:
            Tuple[AccountState, List[FillRecord], List[dict]]: 账户状态、成交记录和委托摘要。
        """
        self._login()
        raw_orders: List[dict] = []
        submitted_ids: List[Tuple[str, str, str]] = []

        for order in order_intents:
            kwargs = self._build_order_kwargs(order)
            result = gm.order_volume(**kwargs)
            raw_orders.append(
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "target_shares": order.target_shares,
                    "delta_shares": order.delta_shares,
                    "ref_price": order.ref_price,
                    "submit_price": kwargs["price"],
                    "reason": order.reason,
                    "raw_text": str(result),
                }
            )
            submitted_ids.extend(self._extract_order_ids(result))
            time.sleep(0.2)

        time.sleep(self.order_wait_seconds)

        reports = gm.get_execution_reports() if hasattr(gm, "get_execution_reports") else []
        report_rows = []
        fills: List[FillRecord] = []
        id_set = {(cl_id, order_id) for cl_id, order_id, _ in submitted_ids}
        order_side_map = {(
            cl_id,
            order_id,
        ): next((x.side for x in order_intents if x.symbol == symbol), "") for cl_id, order_id, symbol in submitted_ids}

        for item in reports:
            cl_ord_id = str(getattr(item, "cl_ord_id", ""))
            order_id = str(getattr(item, "order_id", ""))
            if id_set and (cl_ord_id, order_id) not in id_set:
                continue
            symbol = from_gm_symbol(getattr(item, "symbol", ""))
            price = safe_float(getattr(item, "price", 0.0))
            volume = safe_int(getattr(item, "volume", 0))
            amount = safe_float(getattr(item, "amount", price * volume))
            cost = safe_float(getattr(item, "cost", 0.0))
            side = order_side_map.get((cl_ord_id, order_id), "")
            fee = max(cost - amount, 0.0) if side == "BUY" else max(amount - cost, 0.0)
            net_cash_flow = -(cost if cost > 0 else amount + fee) if side == "BUY" else (cost if cost > 0 else amount - fee)
            report_rows.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "volume": volume,
                    "amount": amount,
                    "cost": cost,
                    "cl_ord_id": cl_ord_id,
                    "order_id": order_id,
                    "exec_id": str(getattr(item, "exec_id", "")),
                    "raw_text": str(item),
                }
            )
            fills.append(
                FillRecord(
                    symbol=symbol,
                    side=side,
                    shares=volume,
                    price=price,
                    gross_amount=amount,
                    fee=fee,
                    net_cash_flow=net_cash_flow,
                    order_id=order_id,
                    exec_id=str(getattr(item, "exec_id", "")),
                )
            )

        after_state = self.load_account_state()
        return after_state, fills, raw_orders + report_rows
