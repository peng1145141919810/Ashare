# -*- coding: utf-8 -*-
"""本地校验器。"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import Any, Dict

from hub.dataset import load_training_table


def _probe_gpu() -> Dict[str, Any]:
    """探测本机 GPU 环境。

    Args:
        None

    Returns:
        GPU 探测结果。
    """
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


def validate_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """校验配置与数据。

    Args:
        config: 配置字典。

    Returns:
        校验结果。
    """
    result = {'ok': True, 'checks': []}
    data_root = Path(str(config['train_table_dir']))
    if not data_root.exists():
        result['ok'] = False
        result['checks'].append({'name': 'train_table_dir', 'ok': False, 'detail': f'不存在: {data_root}'})
        return result
    bundle = load_training_table(data_root=data_root, label_col=str(config['strategy']['label_col']), max_files=1)
    result['checks'].append({'name': 'data_load', 'ok': True, 'detail': f'rows={len(bundle.df)} features={len(bundle.feature_cols)}'})

    gpu_info = _probe_gpu()
    result['checks'].append({'name': 'nvidia_smi', 'ok': gpu_info['ok'], 'detail': gpu_info['detail'], 'name_detected': gpu_info['name'], 'memory': gpu_info['memory'], 'driver': gpu_info['driver']})
    result['checks'].append({'name': 'lightgbm_import', **_probe_python_package('lightgbm')})
    result['checks'].append({'name': 'xgboost_import', **_probe_python_package('xgboost')})
    return result
