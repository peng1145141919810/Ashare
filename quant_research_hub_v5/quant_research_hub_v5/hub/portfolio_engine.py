# -*- coding: utf-8 -*-
"""组合构建与回测。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from hub.io_utils import write_csv, write_json
from hub.metrics import annualized_from_period_returns, max_drawdown_from_nav, sharpe_from_period_returns


def _apply_basic_filters(df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
    """应用基础可交易过滤。

    Args:
        df: 最新打分表。
        strategy: 策略配置。

    Returns:
        过滤后的表。
    """
    out = df.copy()
    if 'listed_days' in out.columns:
        out = out.loc[out['listed_days'].fillna(0) >= int(strategy.get('portfolio_min_listed_days', 120))]
    if 'amount_mean_20' in out.columns:
        out = out.loc[out['amount_mean_20'].fillna(0) >= float(strategy.get('portfolio_min_amount_mean_20', 1e7))]
    if 'vol_20' in out.columns:
        out = out.loc[out['vol_20'].fillna(0) <= float(strategy.get('portfolio_max_vol_20', 0.12))]
    for col in ['is_st', 'is_suspended', 'is_limit']:
        if col in out.columns:
            out = out.loc[out[col].fillna(0) == 0]
    if 'is_tradable_basic' in out.columns:
        out = out.loc[out['is_tradable_basic'].fillna(0) == 1]
    return out


def build_latest_portfolio(latest_scores_df: pd.DataFrame, strategy: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """构建最新组合。

    Args:
        latest_scores_df: 最新打分表。
        strategy: 策略配置。
        out_dir: 输出目录。

    Returns:
        组合摘要。
    """
    df = _apply_basic_filters(latest_scores_df, strategy)
    top_k = int(strategy.get('top_k', 20))
    df = df.sort_values('pred_score', ascending=False).head(top_k).copy()

    weak = False
    if 'hs300_ret_20' in df.columns and df['hs300_ret_20'].notna().any():
        weak = bool(float(df['hs300_ret_20'].iloc[0]) < 0)
    exposure = float(strategy.get('portfolio_weak_market_exposure', 0.5) if weak else strategy.get('portfolio_base_exposure', 1.0))
    n = max(len(df), 1)
    df['portfolio_weight'] = exposure / n
    df['target_exposure'] = exposure
    df['cash_buffer'] = 1.0 - exposure

    cols = [c for c in ['date', 'code', 'ts_code', 'board', 'industry', 'pred_score', 'portfolio_weight', 'target_exposure', 'cash_buffer', 'amount_mean_20', 'vol_20', 'listed_days', 'in_hs300'] if c in df.columns]
    latest_portfolio_path = out_dir / 'latest_portfolio_v1.csv'
    write_csv(latest_portfolio_path, df[cols])
    return {
        'latest_portfolio_path': str(latest_portfolio_path),
        'n_pick': int(len(df)),
        'target_exposure': exposure,
        'market_is_weak': weak,
        'latest_portfolio_df': df,
    }


def backtest_from_pred_test(pred_test_df: pd.DataFrame, strategy: Dict[str, Any], out_dir: Path, label_col: str) -> Dict[str, Any]:
    """从测试集预测结果做简单回测。

    Args:
        pred_test_df: 测试集预测表。
        strategy: 策略配置。
        out_dir: 输出目录。
        label_col: 回测收益标签。

    Returns:
        回测摘要。
    """
    df = pred_test_df.copy()
    if 'date' not in df.columns:
        raise ValueError('pred_test_df 缺少 date 列，无法回测')
    df['date'] = pd.to_datetime(df['date'])
    dates = sorted(df['date'].dropna().unique().tolist())
    horizon = int(strategy.get('portfolio_holding_days', 5) or 5)
    top_k = int(strategy.get('top_k', 20) or 20)

    curve_rows: List[Dict[str, Any]] = []
    holding_rows: List[Dict[str, Any]] = []
    nav = 1.0
    peak = 1.0

    for dt in dates[::max(horizon, 1)]:
        g = df.loc[df['date'] == dt].copy()
        g = _apply_basic_filters(g, strategy)
        if g.empty:
            continue
        g = g.sort_values('pred_score', ascending=False).head(top_k).copy()
        weak = False
        if 'hs300_ret_20' in g.columns and g['hs300_ret_20'].notna().any():
            weak = bool(float(g['hs300_ret_20'].iloc[0]) < 0)
        exposure = float(strategy.get('portfolio_weak_market_exposure', 0.5) if weak else strategy.get('portfolio_base_exposure', 1.0))
        weight = exposure / max(len(g), 1)
        period_ret = float(g[label_col].fillna(0.0).mean()) * exposure
        nav *= (1.0 + period_ret)
        peak = max(peak, nav)
        drawdown = nav / peak - 1.0
        curve_rows.append({'date': dt, 'n_pick': int(len(g)), 'period_ret': period_ret, 'target_exposure': exposure, 'market_is_weak': int(weak), 'nav': nav, 'drawdown': drawdown})
        for _, row in g.iterrows():
            holding_rows.append({
                'date': dt,
                'code': row.get('code'),
                'industry': row.get('industry'),
                'board': row.get('board'),
                'pred_score': row.get('pred_score'),
                'weight': weight,
                'cash_weight': 1.0 - exposure,
                label_col: row.get(label_col),
                'amount_mean_20': row.get('amount_mean_20'),
                'vol_20': row.get('vol_20'),
                'weak_market': int(weak),
                'target_exposure': exposure,
            })

    curve = pd.DataFrame(curve_rows)
    holdings = pd.DataFrame(holding_rows)
    write_csv(out_dir / 'portfolio_backtest_curve.csv', curve)
    write_csv(out_dir / 'portfolio_backtest_holdings.csv', holdings)

    annualized = annualized_from_period_returns(curve['period_ret'].tolist() if not curve.empty else [], periods_per_year=max(240 // max(horizon, 1), 1))
    sharpe = sharpe_from_period_returns(curve['period_ret'].tolist() if not curve.empty else [], periods_per_year=max(240 // max(horizon, 1), 1))
    mdd = max_drawdown_from_nav(curve['nav'].tolist() if not curve.empty else [1.0])
    payload = {
        'annualized_ret': float(annualized),
        'sharpe': float(sharpe),
        'max_drawdown': float(mdd),
        'n_rebalance': int(len(curve)),
        'curve_path': str(out_dir / 'portfolio_backtest_curve.csv'),
        'holdings_path': str(out_dir / 'portfolio_backtest_holdings.csv'),
    }
    write_json(out_dir / 'portfolio_summary.json', payload)
    return payload
