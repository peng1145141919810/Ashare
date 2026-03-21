# -*- coding: utf-8 -*-
"""评估指标。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def safe_corr(a: pd.Series, b: pd.Series, method: str = 'pearson') -> float:
    """安全相关系数。

    Args:
        a: 序列一。
        b: 序列二。
        method: 相关方法。

    Returns:
        float
    """
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3:
        return 0.0
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method=method) or 0.0)


def daily_rank_ic_mean(df: pd.DataFrame, date_col: str, pred_col: str, label_col: str) -> float:
    """逐日 rank IC 均值。

    Args:
        df: 数据表。
        date_col: 日期列。
        pred_col: 预测列。
        label_col: 标签列。

    Returns:
        float
    """
    if date_col not in df.columns or pred_col not in df.columns or label_col not in df.columns:
        return 0.0
    vals: List[float] = []
    for _, g in df.groupby(date_col):
        if len(g) < 5:
            continue
        ic = safe_corr(g[pred_col], g[label_col], method='spearman')
        vals.append(ic)
    return float(np.mean(vals)) if vals else 0.0


def summarize_prediction_frame(df: pd.DataFrame, date_col: str, pred_col: str, label_col: str) -> Dict[str, float]:
    """汇总预测指标。

    Args:
        df: 数据表。
        date_col: 日期列。
        pred_col: 预测列。
        label_col: 标签列。

    Returns:
        指标字典。
    """
    return {
        'pearson_corr': safe_corr(df[pred_col], df[label_col], method='pearson'),
        'spearman_corr': safe_corr(df[pred_col], df[label_col], method='spearman'),
        'daily_rank_ic_mean': daily_rank_ic_mean(df, date_col=date_col, pred_col=pred_col, label_col=label_col),
    }


def annualized_from_period_returns(period_returns: Iterable[float], periods_per_year: int) -> float:
    """周期收益转年化。

    Args:
        period_returns: 周期收益序列。
        periods_per_year: 年内周期数。

    Returns:
        float
    """
    vals = np.asarray(list(period_returns), dtype=float)
    if vals.size == 0:
        return 0.0
    nav = np.cumprod(1.0 + vals)
    years = max(vals.size / float(max(periods_per_year, 1)), 1e-9)
    return float(nav[-1] ** (1.0 / years) - 1.0)


def sharpe_from_period_returns(period_returns: Iterable[float], periods_per_year: int) -> float:
    """周期收益转 Sharpe。

    Args:
        period_returns: 周期收益序列。
        periods_per_year: 年内周期数。

    Returns:
        float
    """
    vals = np.asarray(list(period_returns), dtype=float)
    if vals.size < 2:
        return 0.0
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(mu / sd * np.sqrt(periods_per_year))


def max_drawdown_from_nav(nav: Iterable[float]) -> float:
    """净值最大回撤。

    Args:
        nav: 净值序列。

    Returns:
        float
    """
    arr = np.asarray(list(nav), dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = arr / np.where(peak == 0, 1.0, peak) - 1.0
    return float(np.min(dd))
