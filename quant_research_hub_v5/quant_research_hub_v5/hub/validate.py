# -*- coding: utf-8 -*-
"""本地校验器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from hub.dataset import load_training_table


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
    return result
