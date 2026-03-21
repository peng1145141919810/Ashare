# -*- coding: utf-8 -*-
"""本地校验器（GPU 修复版）。"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hub.dataset import load_training_table


def _probe_gpu() -> Dict[str, Any]:
    """探测本机 GPU 环境。"""
    payload: Dict[str, Any] = {'ok': False, 'detail': '', 'name': '', 'memory': '', 'driver': ''}
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode == 0:
            line = (proc.stdout.strip().splitlines() or [''])[0]
            parts = [x.strip() for x in line.split(',')]
            payload.update({
                'ok': True,
                'detail': line,
                'name': parts[0] if len(parts) > 0 else '',
                'memory': parts[1] if len(parts) > 1 else '',
                'driver': parts[2] if len(parts) > 2 else '',
            })
    except Exception as exc:
        payload['detail'] = str(exc)
    return payload


def _probe_python_package(name: str) -> Dict[str, Any]:
    """探测 Python 包是否可导入。"""
    try:
        module = importlib.import_module(name)
        return {'ok': True, 'version': getattr(module, '__version__', 'unknown')}
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}


def _smoke_test_lightgbm_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """做一个极小的 LightGBM GPU 训练冒烟测试。"""
    try:
        import lightgbm as lgb  # type: ignore

        gp = dict(config.get('gpu_profile', {}))
        X = np.random.RandomState(42).randn(512, 8)
        y = np.random.RandomState(43).randn(512)
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=8,
            learning_rate=0.1,
            num_leaves=15,
            random_state=42,
            n_jobs=1,
            device_type=str(gp.get('lightgbm_device_type', 'gpu')),
            max_bin=int(gp.get('default_max_bin', 63) or 63),
            gpu_platform_id=int(gp.get('gpu_platform_id', 0) or 0),
            gpu_device_id=int(gp.get('gpu_device_id', 0) or 0),
            gpu_use_dp=bool(gp.get('gpu_use_dp', False)),
        )
        model.fit(X, y)
        pred = model.predict(X[:8])
        return {
            'ok': True,
            'detail': f'lightgbm_gpu_smoke_ok pred_mean={float(np.mean(pred)):.6f}',
        }
    except Exception as exc:
        return {'ok': False, 'detail': str(exc)}


def _smoke_test_xgboost_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """做一个极小的 XGBoost GPU 训练冒烟测试。"""
    try:
        import xgboost as xgb  # type: ignore

        gp = dict(config.get('gpu_profile', {}))
        X = np.random.RandomState(44).randn(512, 8)
        y = np.random.RandomState(45).randn(512)
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=12,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method='hist',
            device=str(gp.get('xgboost_device', 'cuda')),
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(X, y)
        pred = model.predict(X[:8])
        return {
            'ok': True,
            'detail': f'xgboost_gpu_smoke_ok pred_mean={float(np.mean(pred)):.6f}',
        }
    except Exception as exc:
        return {'ok': False, 'detail': str(exc)}


def _runtime_summary(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成 GPU 运行时摘要。"""
    lookup = {c['name']: c for c in checks if 'name' in c}
    available_gpu_families = []
    if lookup.get('lightgbm_gpu_fit', {}).get('ok'):
        available_gpu_families.append('lightgbm_gpu')
    if lookup.get('xgboost_gpu_fit', {}).get('ok'):
        available_gpu_families.append('xgboost_gpu')
    return {
        'gpu_visible': bool(lookup.get('nvidia_smi', {}).get('ok')),
        'available_gpu_families': available_gpu_families,
        'lightgbm_gpu_ready': bool(lookup.get('lightgbm_gpu_fit', {}).get('ok')),
        'xgboost_gpu_ready': bool(lookup.get('xgboost_gpu_fit', {}).get('ok')),
    }


def validate_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """校验配置、数据与 GPU 可用性。"""
    result = {'ok': True, 'checks': []}
    data_root = Path(str(config['train_table_dir']))
    if not data_root.exists():
        result['ok'] = False
        result['checks'].append({'name': 'train_table_dir', 'ok': False, 'detail': f'不存在: {data_root}'})
        return result

    bundle = load_training_table(data_root=data_root, label_col=str(config['strategy']['label_col']), max_files=1)
    result['checks'].append({'name': 'data_load', 'ok': True, 'detail': f'rows={len(bundle.df)} features={len(bundle.feature_cols)}'})

    gpu_info = _probe_gpu()
    result['checks'].append({
        'name': 'nvidia_smi',
        'ok': gpu_info['ok'],
        'detail': gpu_info['detail'],
        'name_detected': gpu_info['name'],
        'memory': gpu_info['memory'],
        'driver': gpu_info['driver'],
    })

    result['checks'].append({'name': 'lightgbm_import', **_probe_python_package('lightgbm')})
    result['checks'].append({'name': 'xgboost_import', **_probe_python_package('xgboost')})

    # 只有显卡可见时才做实际 GPU 冒烟测试，避免无卡机器报假红。
    if gpu_info['ok']:
        result['checks'].append({'name': 'lightgbm_gpu_fit', **_smoke_test_lightgbm_gpu(config)})
        result['checks'].append({'name': 'xgboost_gpu_fit', **_smoke_test_xgboost_gpu(config)})
    else:
        result['checks'].append({'name': 'lightgbm_gpu_fit', 'ok': False, 'detail': 'nvidia-smi 不可见，跳过 GPU 冒烟测试'})
        result['checks'].append({'name': 'xgboost_gpu_fit', 'ok': False, 'detail': 'nvidia-smi 不可见，跳过 GPU 冒烟测试'})

    result['runtime'] = _runtime_summary(result['checks'])
    # “ok” 只表示环境能运行，不要求 GPU 一定可用。
    result['ok'] = any(c.get('name') == 'data_load' and c.get('ok') for c in result['checks'])
    return result
