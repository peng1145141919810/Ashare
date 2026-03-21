# -*- coding: utf-8 -*-
"""训练表读取与字段推断。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DATE_CANDIDATES = ['date', 'trade_date', 'datetime']
CODE_CANDIDATES = ['ts_code', 'code', 'symbol']
INDUSTRY_CANDIDATES = ['industry', 'sw_level1', 'sector']

RESERVED_EXACT = {
    'year', 'month', 'day', 'pred_score', 'portfolio_weight', 'cash_weight', 'weight',
    'is_st', 'is_limit', 'is_suspended', 'is_tradable_basic', 'in_hs300', 'board_code',
    'industry_code'
}


@dataclass
class DatasetBundle:
    """数据集描述。"""

    df: pd.DataFrame
    date_col: str
    code_col: str
    industry_col: Optional[str]
    feature_cols: List[str]
    label_cols: List[str]


class DatasetError(Exception):
    """数据错误。"""


def _detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """自动识别列名。

    Args:
        df: 数据表。
        candidates: 候选列名。

    Returns:
        匹配到的列名。
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _iter_data_files(data_root: Path) -> List[Path]:
    """枚举数据文件。

    Args:
        data_root: 数据目录。

    Returns:
        文件列表。
    """
    if data_root.is_file():
        return [data_root]
    patterns = ['*.parquet', '*.csv']
    files: List[Path] = []
    for p in patterns:
        files.extend(sorted(data_root.rglob(p)))
    return files


def _read_file(path: Path) -> pd.DataFrame:
    """读取单个文件。

    Args:
        path: 文件路径。

    Returns:
        DataFrame
    """
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_training_table(
    data_root: Path,
    label_col: str,
    max_files: Optional[int] = None,
    sample_rows: Optional[int] = None,
) -> DatasetBundle:
    """读取训练表。

    Args:
        data_root: 数据目录或文件。
        label_col: 主标签列。
        max_files: 最多读多少个文件。
        sample_rows: 若不为空，仅保留前 N 行用于快速调试。

    Returns:
        DatasetBundle
    """
    files = _iter_data_files(data_root)
    if not files:
        raise DatasetError(f'未找到训练表文件: {data_root}')
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    parts: List[pd.DataFrame] = []
    for fp in files:
        parts.append(_read_file(fp))
    df = pd.concat(parts, ignore_index=True)
    if sample_rows is not None and sample_rows > 0:
        df = df.head(sample_rows).copy()

    date_col = _detect_col(df, DATE_CANDIDATES)
    code_col = _detect_col(df, CODE_CANDIDATES)
    industry_col = _detect_col(df, INDUSTRY_CANDIDATES)
    if date_col is None:
        raise DatasetError('无法识别日期列，请保证存在 date 或 trade_date')
    if code_col is None:
        raise DatasetError('无法识别股票代码列，请保证存在 ts_code 或 code')
    if label_col not in df.columns:
        raise DatasetError(f'标签列不存在: {label_col}')

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, code_col]).reset_index(drop=True)

    label_cols = [c for c in df.columns if c.startswith('future_ret_')]
    numeric_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
    feature_cols: List[str] = []
    for c in numeric_cols:
        if c in RESERVED_EXACT:
            continue
        if c in label_cols:
            continue
        if c == label_col:
            continue
        feature_cols.append(c)

    if not feature_cols:
        raise DatasetError('没有识别到可训练的数值特征列')

    return DatasetBundle(
        df=df,
        date_col=date_col,
        code_col=code_col,
        industry_col=industry_col,
        feature_cols=feature_cols,
        label_cols=label_cols,
    )


def split_by_dates(df: pd.DataFrame, date_col: str, train_ratio: float = 0.6, valid_ratio: float = 0.2):
    """按日期切分训练/验证/测试。

    Args:
        df: 数据表。
        date_col: 日期列。
        train_ratio: 训练集比例。
        valid_ratio: 验证集比例。

    Returns:
        (train_df, valid_df, test_df)
    """
    dates = sorted(pd.Series(df[date_col].dropna().unique()).tolist())
    if len(dates) < 10:
        raise DatasetError('唯一日期太少，无法切分训练/验证/测试')
    n = len(dates)
    train_end = max(1, int(n * train_ratio))
    valid_end = max(train_end + 1, int(n * (train_ratio + valid_ratio)))
    train_dates = set(dates[:train_end])
    valid_dates = set(dates[train_end:valid_end])
    test_dates = set(dates[valid_end:])
    train_df = df.loc[df[date_col].isin(train_dates)].copy()
    valid_df = df.loc[df[date_col].isin(valid_dates)].copy()
    test_df = df.loc[df[date_col].isin(test_dates)].copy()
    return train_df, valid_df, test_df
