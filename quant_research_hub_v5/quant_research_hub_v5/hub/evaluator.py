# -*- coding: utf-8 -*-
"""统一评分器。"""

from __future__ import annotations

from typing import Any, Dict


def compute_total_score(train_summary: Dict[str, Any], portfolio_summary: Dict[str, Any], rules: Dict[str, Any]) -> float:
    """计算综合得分。

    Args:
        train_summary: 训练摘要。
        portfolio_summary: 组合摘要。
        rules: 权重配置。

    Returns:
        float
    """
    valid = train_summary.get('valid_metrics', {})
    test = train_summary.get('test_metrics', {})
    score = 0.0
    score += float(valid.get('daily_rank_ic_mean', 0.0)) * float(rules.get('w_valid_ic', 200.0))
    score += float(test.get('daily_rank_ic_mean', 0.0)) * float(rules.get('w_test_ic', 250.0))
    score += float(valid.get('spearman_corr', 0.0)) * float(rules.get('w_valid_spearman', 150.0))
    score += float(test.get('spearman_corr', 0.0)) * float(rules.get('w_test_spearman', 200.0))
    score += float(portfolio_summary.get('annualized_ret', 0.0)) * float(rules.get('w_ret', 20.0))
    score += float(portfolio_summary.get('sharpe', 0.0)) * float(rules.get('w_sharpe', 10.0))
    score -= abs(min(float(portfolio_summary.get('max_drawdown', 0.0)), 0.0)) * float(rules.get('w_drawdown', 80.0))
    return float(score)
