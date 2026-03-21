# -*- coding: utf-8 -*-
"""训练与预测引擎。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hub.dataset import DatasetBundle, split_by_dates
from hub.io_utils import write_csv, write_json
from hub.metrics import summarize_prediction_frame
from hub.model_families import build_model


def _load_feature_transform(path: Optional[Path]):
    """加载候选特征变换函数。

    Args:
        path: 特征包路径。

    Returns:
        transform_features 函数或 None
    """
    if path is None or not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location('feature_pack', path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return getattr(module, 'transform_features', None)
    except Exception:
        return None


def _load_training_override(path: Optional[Path]) -> Dict[str, Any]:
    """加载训练计划覆盖。

    Args:
        path: 覆盖模块路径。

    Returns:
        计划字典。
    """
    if path is None or not path.exists():
        return {}
    try:
        spec = importlib.util.spec_from_file_location('train_override', path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        if hasattr(module, 'override_training_plan'):
            return dict(module.override_training_plan({}) or {})
    except Exception:
        return {}
    return {}


def _select_features(df: pd.DataFrame, feature_cols: List[str], profile: str) -> List[str]:
    """按特征档案筛特征。

    Args:
        df: 数据表。
        feature_cols: 基础特征列。
        profile: 档案名。

    Returns:
        选中的特征列。
    """
    cols = list(feature_cols)
    if profile == 'baseline_plus':
        return cols
    if profile == 'momentum_cross_section':
        keys = ('ret_', 'alpha_ret_', 'hs300_ret_')
        out = [c for c in cols if c.startswith(keys)]
        return out or cols
    if profile == 'vol_liq_quality':
        keys = ('vol_', 'amount_', 'turnover_', 'listed_days', 'total_mv', 'circ_mv')
        out = [c for c in cols if c.startswith(('vol_', 'amount_', 'turnover_')) or c in {'listed_days', 'total_mv', 'circ_mv'}]
        return out or cols
    if profile == 'defensive_residual':
        keep = []
        for c in cols:
            if c.startswith(('alpha_ret_', 'vol_', 'hs300_ret_')) or c in {'listed_days', 'amount_z_20'}:
                keep.append(c)
        return keep or cols
    if profile == 'interaction_sparse':
        return cols
    if profile == 'generated_feature_pack':
        return cols
    return cols


def _apply_interaction_sparse(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """加入稀疏交互项。

    Args:
        df: 数据表。
        cols: 原始特征列。

    Returns:
        (新数据表, 新特征列)
    """
    out = df.copy()
    chosen = [c for c in cols if c in {'ret_5', 'ret_20', 'ret_60', 'vol_20', 'amount_z_20', 'listed_days'}]
    pairs = [('ret_5', 'ret_20'), ('ret_20', 'vol_20'), ('alpha_ret_20_vs_hs300', 'vol_20')]
    for a, b in pairs:
        if a in out.columns and b in out.columns:
            name = f'inter_{a}__{b}'
            out[name] = out[a].astype(float) * out[b].astype(float)
            chosen.append(name)
    keep = list(dict.fromkeys(cols + [c for c in chosen if c in out.columns]))
    return out, keep


def _recent_exponential_weights(dates: pd.Series) -> np.ndarray:
    """生成近端指数加权。

    Args:
        dates: 日期列。

    Returns:
        权重数组。
    """
    order = pd.Series(dates.rank(method='dense').astype(float))
    scaled = (order - order.min()) / max(order.max() - order.min(), 1.0)
    return np.exp(2.0 * scaled.to_numpy())


def train_and_predict(
    bundle: DatasetBundle,
    candidate: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """训练候选实验并输出摘要。

    Args:
        bundle: 数据包。
        candidate: 候选实验配置。
        run_dir: 运行目录。

    Returns:
        训练结果摘要。
    """
    df = bundle.df.copy()
    feature_pack_path = Path(candidate['lab'].get('feature_pack_path', '')) if candidate.get('lab') else None
    train_override_path = Path(candidate['lab'].get('train_override_path', '')) if candidate.get('lab') else None
    generated_model_path = Path(candidate['lab'].get('generated_model_path', '')) if candidate.get('lab') else None

    feature_transform = _load_feature_transform(feature_pack_path)
    if feature_transform is not None:
        try:
            df = feature_transform(df)
        except Exception:
            pass

    feature_cols = [c for c in df.columns if c in bundle.feature_cols or c.startswith('feat_') or c.startswith('inter_')]
    selected = _select_features(df, feature_cols, candidate['feature_profile'])
    if candidate['feature_profile'] == 'interaction_sparse':
        df, selected = _apply_interaction_sparse(df, selected)

    label_col = candidate['label_col']
    date_col = bundle.date_col
    code_col = bundle.code_col
    train_df, valid_df, test_df = split_by_dates(df, date_col=date_col)

    train_plan = _load_training_override(train_override_path)
    X_train = train_df[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df[label_col].astype(float)
    X_valid = valid_df[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_valid = valid_df[label_col].astype(float)
    X_test = test_df[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = test_df[label_col].astype(float)

    if candidate['training_logic'] in {'weighted_recent', 'regularized_recent'} or train_plan.get('sample_weight_mode') == 'recent_exponential':
        sample_weight = _recent_exponential_weights(train_df[date_col])
    else:
        sample_weight = None

    if candidate['training_logic'] == 'feature_select' or train_plan.get('feature_cap'):
        cap = int(train_plan.get('feature_cap', 60) or 60)
        vars_ = X_train.var(axis=0).sort_values(ascending=False)
        selected = vars_.head(min(cap, len(vars_))).index.tolist()
        X_train = X_train[selected]
        X_valid = X_valid[selected]
        X_test = X_test[selected]

    model_family = candidate['model_family']
    model = build_model(model_family, generated_model_path=generated_model_path)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)

    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    pred_train_df = train_df[[date_col, code_col]].copy()
    pred_train_df['pred_score'] = pred_train
    pred_train_df[label_col] = y_train.to_numpy()

    pred_valid_df = valid_df[[date_col, code_col]].copy()
    pred_valid_df['pred_score'] = pred_valid
    pred_valid_df[label_col] = y_valid.to_numpy()

    pred_test_df = test_df.copy()
    pred_test_df['pred_score'] = pred_test

    latest_date = df[date_col].max()
    latest_df = df.loc[df[date_col] == latest_date].copy()
    X_latest = latest_df[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    latest_df['pred_score'] = model.predict(X_latest)

    valid_metrics = summarize_prediction_frame(pred_valid_df, date_col=date_col, pred_col='pred_score', label_col=label_col)
    test_metrics = summarize_prediction_frame(pred_test_df, date_col=date_col, pred_col='pred_score', label_col=label_col)

    train_summary = {
        'strategy_name': candidate['strategy_name'],
        'label_col': label_col,
        'model_family': model_family,
        'feature_profile': candidate['feature_profile'],
        'training_logic': candidate['training_logic'],
        'n_features': int(len(selected)),
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
    }

    write_json(run_dir / 'train_summary.json', train_summary)
    write_csv(run_dir / 'pred_test.csv', pred_test_df)
    write_csv(run_dir / 'latest_scores.csv', latest_df.sort_values('pred_score', ascending=False))

    feature_importance_df = pd.DataFrame(columns=['feature', 'importance'])
    if hasattr(model, 'feature_importances_'):
        feature_importance_df = pd.DataFrame({'feature': selected, 'importance': list(model.feature_importances_)})
    elif hasattr(model, 'coef_'):
        coefs = np.ravel(model.coef_)
        feature_importance_df = pd.DataFrame({'feature': selected, 'importance': np.abs(coefs)})
    else:
        feature_importance_df = pd.DataFrame({'feature': selected, 'importance': [0.0] * len(selected)})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    write_csv(run_dir / 'feature_importance.csv', feature_importance_df)

    return {
        'train_summary_path': str(run_dir / 'train_summary.json'),
        'pred_test_path': str(run_dir / 'pred_test.csv'),
        'latest_scores_path': str(run_dir / 'latest_scores.csv'),
        'feature_importance_path': str(run_dir / 'feature_importance.csv'),
        'train_summary': train_summary,
        'pred_test_df': pred_test_df,
        'latest_scores_df': latest_df,
    }
