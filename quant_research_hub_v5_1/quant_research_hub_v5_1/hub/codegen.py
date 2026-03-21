# -*- coding: utf-8 -*-
"""自动代码实验室。

这里不直接改主库脚本，而是在每个候选实验自己的工作区里生成候选代码，
保证失败时不会污染主工程。
"""

from __future__ import annotations

import importlib.util
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from hub.io_utils import write_json
from hub.llm_client import LLMClient


FEATURE_TEMPLATE = r'''
# -*- coding: utf-8 -*-
"""候选特征包。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """对输入数据做候选特征变换。

    Args:
        df: 原始数据表。

    Returns:
        变换后的数据表。
    """
    out = df.copy()
    if 'ret_5' in out.columns and 'ret_20' in out.columns:
        out['feat_mom_spread_5_20'] = out['ret_5'] - out['ret_20']
    if 'ret_20' in out.columns and 'ret_60' in out.columns:
        out['feat_mom_spread_20_60'] = out['ret_20'] - out['ret_60']
    if 'vol_20' in out.columns and 'amount_mean_20' in out.columns:
        out['feat_vol_liq_ratio'] = out['vol_20'] / (out['amount_mean_20'].abs() + 1.0)
    if 'alpha_ret_20_vs_hs300' in out.columns and 'vol_20' in out.columns:
        out['feat_alpha_vol_mix'] = out['alpha_ret_20_vs_hs300'] / (out['vol_20'].abs() + 1e-6)
    return out
'''

TRAIN_TEMPLATE = r'''
# -*- coding: utf-8 -*-
"""候选训练逻辑覆盖。"""

from __future__ import annotations

from typing import Dict, Any


def override_training_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """返回训练计划的增量修改。

    Args:
        plan: 原始训练计划。

    Returns:
        新的训练计划。
    """
    out = dict(plan)
    out.setdefault('sample_weight_mode', 'recent_exponential')
    out.setdefault('clip_label_quantile', 0.01)
    out.setdefault('feature_cap', 80)
    return out
'''

MODEL_TEMPLATE = r'''
# -*- coding: utf-8 -*-
"""候选模型家族。"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor


def build_model(random_state: int = 42):
    """构造候选模型。

    Args:
        random_state: 随机种子。

    Returns:
        sklearn 风格模型。
    """
    return HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=260,
        random_state=random_state,
    )
'''


class CodegenLab:
    """候选代码实验室。"""

    def __init__(self, llm_client: LLMClient):
        """初始化。

        Args:
            llm_client: LLM 客户端。

        Returns:
            None
        """
        self.llm_client = llm_client

    def _validate_module(self, path: Path) -> Dict[str, Any]:
        """尝试编译与导入模块。

        Args:
            path: 模块路径。

        Returns:
            校验结果。
        """
        try:
            source = path.read_text(encoding='utf-8')
            compile(source, str(path), 'exec')
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            return {'ok': True, 'error': ''}
        except Exception:
            return {'ok': False, 'error': traceback.format_exc()}

    def _maybe_llm_code(self, kind: str, context: Dict[str, Any], fallback: str) -> str:
        """用 LLM 生成代码，失败则回退模板。

        Args:
            kind: 代码类型。
            context: 上下文。
            fallback: 兜底模板。

        Returns:
            代码文本。
        """
        if not self.llm_client.is_enabled():
            return fallback
        system_prompt = (
            '你是量化研究中枢的代码生成器。'
            '只输出可运行 Python 代码，不要输出解释。'
            '代码优先依赖 pandas、numpy、sklearn；如需更重模型，可选 lightgbm 或 xgboost，但必须提供失败时的回退。'
        )
        user_prompt = f'请生成 {kind} 候选代码。上下文：{context}'
        txt = self.llm_client.chat_text(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
        if not txt:
            return fallback
        return txt

    def build_workspace(self, workspace_dir: Path, context: Dict[str, Any]) -> Dict[str, Any]:
        """生成候选代码工作区。

        Args:
            workspace_dir: 工作区目录。
            context: 候选实验上下文。

        Returns:
            工作区描述。
        """
        workspace_dir.mkdir(parents=True, exist_ok=True)
        feature_code = self._maybe_llm_code('feature_pack.py', context, FEATURE_TEMPLATE)
        train_code = self._maybe_llm_code('train_override.py', context, TRAIN_TEMPLATE)
        model_code = self._maybe_llm_code('generated_model.py', context, MODEL_TEMPLATE)

        feature_path = workspace_dir / 'feature_pack.py'
        train_path = workspace_dir / 'train_override.py'
        model_path = workspace_dir / 'generated_model.py'
        feature_path.write_text(feature_code, encoding='utf-8')
        train_path.write_text(train_code, encoding='utf-8')
        model_path.write_text(model_code, encoding='utf-8')

        validations = {
            'feature_pack': self._validate_module(feature_path),
            'train_override': self._validate_module(train_path),
            'generated_model': self._validate_module(model_path),
        }
        write_json(workspace_dir / 'workspace_validation.json', validations)
        return {
            'workspace_dir': str(workspace_dir),
            'feature_pack_path': str(feature_path),
            'train_override_path': str(train_path),
            'generated_model_path': str(model_path),
            'validations': validations,
        }
