# -*- coding: utf-8 -*-
"""模型家族工厂。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge


class FormulaModel:
    """基于公式的轻量模型。"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """初始化。

        Args:
            weights: 特征权重。

        Returns:
            None
        """
        self.weights = dict(weights or {})
        self._features = []

    def fit(self, X, y, sample_weight=None):
        """拟合。

        Args:
            X: 特征表。
            y: 标签。
            sample_weight: 样本权重。

        Returns:
            self
        """
        self._features = list(X.columns)
        if not self.weights:
            # 用简单相关系数估权重。
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
        """预测。

        Args:
            X: 特征表。

        Returns:
            预测数组。
        """
        arr = np.zeros(len(X), dtype=float)
        for c, w in self.weights.items():
            if c in X.columns:
                arr += float(w) * np.asarray(X[c], dtype=float)
        return arr


def _load_generated_model(path: Path):
    """加载候选模型模块。

    Args:
        path: 模块路径。

    Returns:
        模型对象或 None
    """
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


def build_model(family: str, generated_model_path: Optional[Path] = None):
    """构建模型。

    Args:
        family: 模型家族名。
        generated_model_path: 候选模型模块路径。

    Returns:
        sklearn 风格模型。
    """
    family = str(family or 'ridge_ranker')
    if family == 'lightgbm_auto':
        try:
            import lightgbm as lgb  # type: ignore
            return lgb.LGBMRegressor(
                objective='regression',
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
            )
        except Exception:
            return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=42)
    if family == 'ridge_ranker':
        return Ridge(alpha=3.0, random_state=42)
    if family == 'elastic_net':
        return ElasticNet(alpha=0.001, l1_ratio=0.25, random_state=42)
    if family == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=320, max_depth=7, min_samples_leaf=20, random_state=42, n_jobs=-1)
    if family == 'hist_gbdt':
        return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=42)
    if family == 'random_forest':
        return RandomForestRegressor(n_estimators=260, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
    if family == 'formula_blend':
        return FormulaModel()
    if family == 'generated_family' and generated_model_path is not None:
        model = _load_generated_model(generated_model_path)
        if model is not None:
            return model
        return HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=200, random_state=42)
    return Ridge(alpha=3.0, random_state=42)
