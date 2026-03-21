# -*- coding: utf-8 -*-
"""模型家族工厂。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge


class FormulaModel:
    """基于公式的轻量模型。"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(weights or {})
        self._features = []

    def fit(self, X, y, sample_weight=None):
        self._features = list(X.columns)
        if not self.weights:
            import pandas as pd
            y_ser = pd.Series(y)
            scores = {}
            for c in self._features[:20]:
                try:
                    scores[c] = float(pd.Series(X[c]).corr(y_ser) or 0.0)
                except Exception:
                    scores[c] = 0.0
            self.weights = scores
        return self

    def predict(self, X):
        arr = np.zeros(len(X), dtype=float)
        for c, w in self.weights.items():
            if c in X.columns:
                arr += float(w) * np.asarray(X[c], dtype=float)
        return arr


def _load_generated_model(path: Path):
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location('generated_model', path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        if hasattr(module, 'build_model'):
            return module.build_model(random_state=42)
    except Exception:
        return None
    return None


def _lightgbm_cpu_model() -> Any:
    try:
        import lightgbm as lgb  # type: ignore
        return lgb.LGBMRegressor(
            objective='regression',
            n_estimators=320,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=42)


def _lightgbm_gpu_model(model_options: Optional[Dict[str, Any]] = None) -> Any:
    model_options = dict(model_options or {})
    try:
        import lightgbm as lgb  # type: ignore
        params: Dict[str, Any] = {
            'objective': 'regression',
            'n_estimators': int(model_options.get('n_estimators', 420)),
            'learning_rate': float(model_options.get('learning_rate', 0.05)),
            'num_leaves': int(model_options.get('num_leaves', 63)),
            'subsample': float(model_options.get('subsample', 0.9)),
            'colsample_bytree': float(model_options.get('colsample_bytree', 0.9)),
            'random_state': 42,
            'n_jobs': -1,
            'device_type': str(model_options.get('lightgbm_device_type', 'gpu')),
            'max_bin': int(model_options.get('max_bin', 63)),
        }
        if 'gpu_platform_id' in model_options:
            params['gpu_platform_id'] = int(model_options['gpu_platform_id'])
        if 'gpu_device_id' in model_options:
            params['gpu_device_id'] = int(model_options['gpu_device_id'])
        if bool(model_options.get('gpu_use_dp', False)):
            params['gpu_use_dp'] = True
        return lgb.LGBMRegressor(**params)
    except Exception:
        return _lightgbm_cpu_model()


def _xgboost_gpu_model(model_options: Optional[Dict[str, Any]] = None) -> Any:
    model_options = dict(model_options or {})
    try:
        import xgboost as xgb  # type: ignore
        device = str(model_options.get('xgboost_device', 'cuda'))
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=int(model_options.get('n_estimators', 420)),
            learning_rate=float(model_options.get('learning_rate', 0.05)),
            max_depth=int(model_options.get('max_depth', 8)),
            subsample=float(model_options.get('subsample', 0.85)),
            colsample_bytree=float(model_options.get('colsample_bytree', 0.85)),
            reg_lambda=float(model_options.get('reg_lambda', 1.0)),
            tree_method=str(model_options.get('tree_method', 'hist')),
            max_bin=int(model_options.get('max_bin', 256)),
            device=device,
            random_state=42,
            n_jobs=0,
        )
    except Exception:
        return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=42)


def build_model(
    family: str,
    generated_model_path: Optional[Path] = None,
    model_options: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, str, Dict[str, Any]]:
    """构建模型。

    Returns:
        (模型对象, 实际模型家族名, 元信息)
    """
    family = str(family or 'ridge_ranker')
    meta: Dict[str, Any] = {'requested_family': family, 'gpu_requested': family in {'lightgbm_gpu', 'xgboost_gpu'}}

    if family == 'lightgbm_auto':
        model = _lightgbm_cpu_model()
        realized = 'lightgbm_auto' if model.__class__.__name__.startswith('LGBM') else 'hist_gbdt_fallback'
        meta.update({'gpu_used': False, 'backend': realized})
        return model, realized, meta
    if family == 'lightgbm_gpu':
        model = _lightgbm_gpu_model(model_options=model_options)
        realized = 'lightgbm_gpu' if model.__class__.__name__.startswith('LGBM') else 'lightgbm_gpu_fallback_cpu'
        meta.update({'gpu_used': realized == 'lightgbm_gpu', 'backend': realized})
        return model, realized, meta
    if family == 'xgboost_gpu':
        model = _xgboost_gpu_model(model_options=model_options)
        realized = 'xgboost_gpu' if model.__class__.__module__.startswith('xgboost') else 'xgboost_gpu_fallback_cpu'
        meta.update({'gpu_used': realized == 'xgboost_gpu', 'backend': realized})
        return model, realized, meta
    if family == 'ridge_ranker':
        return Ridge(alpha=3.0, random_state=42), 'ridge_ranker', {'requested_family': family, 'gpu_used': False, 'backend': 'ridge'}
    if family == 'elastic_net':
        return ElasticNet(alpha=0.001, l1_ratio=0.25, random_state=42), 'elastic_net', {'requested_family': family, 'gpu_used': False, 'backend': 'elastic_net'}
    if family == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=320, max_depth=7, min_samples_leaf=20, random_state=42, n_jobs=-1), 'extra_trees', {'requested_family': family, 'gpu_used': False, 'backend': 'extra_trees'}
    if family == 'hist_gbdt':
        return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=42), 'hist_gbdt', {'requested_family': family, 'gpu_used': False, 'backend': 'hist_gbdt'}
    if family == 'random_forest':
        return RandomForestRegressor(n_estimators=220, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1), 'random_forest', {'requested_family': family, 'gpu_used': False, 'backend': 'random_forest'}
    if family == 'formula_blend':
        return FormulaModel(), 'formula_blend', {'requested_family': family, 'gpu_used': False, 'backend': 'formula_blend'}
    if family == 'generated_family' and generated_model_path is not None:
        model = _load_generated_model(generated_model_path)
        if model is not None:
            return model, 'generated_family', {'requested_family': family, 'gpu_used': False, 'backend': 'generated_family'}
        return HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=200, random_state=42), 'generated_family_fallback', {'requested_family': family, 'gpu_used': False, 'backend': 'generated_family_fallback'}
    return Ridge(alpha=3.0, random_state=42), 'ridge_ranker', {'requested_family': family, 'gpu_used': False, 'backend': 'ridge_default'}
