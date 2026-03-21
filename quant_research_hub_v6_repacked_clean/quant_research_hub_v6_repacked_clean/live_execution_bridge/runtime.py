from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .brokers.gmtrade_sim_broker import GMTradeSimBroker
from .io_portfolio import build_price_map, load_target_positions
from .rebalance import plan_rebalance
from .utils import dump_json, ensure_dir, now_str, write_dataframe


def build_broker(config: Dict[str, Any]) -> GMTradeSimBroker:
    """构建掘金仿真券商。

    Args:
        config: 运行配置。

    Returns:
        GMTradeSimBroker: 券商适配器。
    """
    broker_cfg = config["broker"]
    return GMTradeSimBroker(
        token=str(broker_cfg["token"]),
        account_id=str(broker_cfg["account_id"]),
        account_alias=str(broker_cfg.get("account_alias", "")),
        endpoint=str(broker_cfg.get("endpoint", "api.myquant.cn:9000")),
        buy_price_ratio=float(broker_cfg.get("buy_price_ratio", 1.01)),
        sell_price_ratio=float(broker_cfg.get("sell_price_ratio", 0.99)),
        order_wait_seconds=float(broker_cfg.get("order_wait_seconds", 3.0)),
        sell_by_available=bool(broker_cfg.get("sell_by_available", True)),
    )


def run_once(config: Dict[str, Any]) -> Dict[str, Any]:
    """执行一次掘金仿真调仓。

    Args:
        config: 运行配置。

    Returns:
        Dict[str, Any]: 执行报告字典。
    """
    output_dir = ensure_dir(config["output_dir"])
    timestamp = now_str()
    broker_cfg = config["broker"]

    portfolio_path, portfolio_frame, target_positions = load_target_positions(config)
    price_map = build_price_map(portfolio_frame, config.get("price_snapshot_path", ""))
    broker = build_broker(config)
    before_state = broker.load_account_state()

    # 用账户已有持仓价格补齐价格映射，避免老仓无价无法清仓。
    for pos in before_state.positions:
        price_map.setdefault(pos.symbol, pos.last_price)

    # 检查是否存在无价目标。
    missing_symbols = [item.symbol for item in target_positions if float(price_map.get(item.symbol, 0.0)) <= 0]
    if missing_symbols:
        raise ValueError(
            "以下目标证券缺少价格，无法计算股数："
            + ", ".join(missing_symbols)
            + "。请在 latest_portfolio_v1.csv 中加入 price/close 列，或提供 price_snapshot_path。"
        )

    order_intents = plan_rebalance(
        account_state=before_state,
        target_positions=target_positions,
        price_map=price_map,
        lot_size=int(broker_cfg.get("lot_size", 100)),
        min_trade_value=float(broker_cfg.get("min_trade_value", 2000.0)),
        cash_reserve_ratio=float(broker_cfg.get("cash_reserve_ratio", 0.02)),
        sell_by_available=bool(broker_cfg.get("sell_by_available", True)),
    )

    after_state, fills, raw_rows = broker.execute_orders(order_intents, price_map)

    write_dataframe(output_dir / f"orders_{timestamp}.csv", [x.to_dict() for x in order_intents])
    write_dataframe(output_dir / f"fills_{timestamp}.csv", [x.to_dict() for x in fills])
    write_dataframe(output_dir / f"gmtrade_raw_{timestamp}.csv", raw_rows)
    write_dataframe(output_dir / "latest_target_snapshot.csv", [x.to_dict() for x in target_positions])
    dump_json(output_dir / "latest_account_state.json", after_state.to_dict())

    equity_curve_path = output_dir / "equity_curve.csv"
    new_row = {
        "timestamp": timestamp,
        "cash": after_state.cash,
        "nav": after_state.nav(),
        "n_positions": len(after_state.positions),
        "portfolio_file": str(portfolio_path),
    }
    if equity_curve_path.exists():
        eq = pd.read_csv(equity_curve_path)
        eq = pd.concat([eq, pd.DataFrame([new_row])], ignore_index=True)
    else:
        eq = pd.DataFrame([new_row])
    eq.to_csv(equity_curve_path, index=False, encoding="utf-8-sig")

    report = {
        "timestamp": timestamp,
        "portfolio_path": str(portfolio_path),
        "price_snapshot_path": str(config.get("price_snapshot_path", "")),
        "broker_type": "gmtrade_sim",
        "n_target_positions": len(target_positions),
        "n_orders": len(order_intents),
        "n_fills": len(fills),
        "before_nav": before_state.nav(),
        "after_nav": after_state.nav(),
        "before_cash": before_state.cash,
        "after_cash": after_state.cash,
        "positions_after": [x.to_dict() for x in after_state.positions],
    }
    dump_json(output_dir / f"execution_report_{timestamp}.json", report)
    return report
