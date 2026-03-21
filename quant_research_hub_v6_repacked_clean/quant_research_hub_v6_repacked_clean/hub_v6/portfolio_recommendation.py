# -*- coding: utf-8 -*-
"""V6 持仓建议层：读取 V5.1 最优实验输出，生成模拟盘前建议。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .config_utils import ensure_dir


def _sort_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ['total_score', 'sharpe', 'valid_ic', 'created_at'] if c in df.columns]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols, ascending=[False] * len(sort_cols))


def _resolve_run_dir(hub_root: Path, row: Dict[str, Any]) -> Path | None:
    candidates: List[Path] = []
    run_id = str(row.get('run_id', '') or '').strip()
    if run_id:
        candidates.append(hub_root / 'runs' / run_id)
    for key in ['latest_portfolio_path', 'portfolio_summary_path', 'train_summary_path', 'pred_test_path']:
        raw = str(row.get(key, '') or '').strip()
        if not raw:
            continue
        path = Path(raw)
        candidates.append(path.parent if path.suffix else path)
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate)
        if text in seen:
            continue
        seen.add(text)
        if candidate.exists() and (candidate / 'latest_portfolio_v1.csv').exists():
            return candidate
    return None


def _latest_cycle_results(hub_root: Path) -> pd.DataFrame:
    state_path = hub_root / 'controller_state.json'
    if not state_path.exists():
        return pd.DataFrame()
    try:
        state = json.loads(state_path.read_text(encoding='utf-8'))
    except Exception:
        return pd.DataFrame()
    cycle_id = str(state.get('last_cycle_id', '') or '').strip()
    if not cycle_id:
        return pd.DataFrame()
    cycle_summary_path = hub_root / 'cycles' / cycle_id / 'cycle_summary.json'
    if not cycle_summary_path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(cycle_summary_path.read_text(encoding='utf-8'))
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(list(payload.get('results', []) or []))


def _pick_best_run(hub_root: Path) -> Tuple[Dict[str, Any], Path]:
    registry_path = hub_root / 'registry' / 'experiment_registry.csv'
    frames: List[Tuple[str, pd.DataFrame]] = []
    cycle_df = _latest_cycle_results(hub_root=hub_root)
    if not cycle_df.empty:
        frames.append(('latest_cycle', cycle_df))
    if registry_path.exists():
        frames.append(('registry', pd.read_csv(registry_path)))
    if not frames:
        raise FileNotFoundError(f'未找到注册表: {registry_path}')

    skipped: List[str] = []
    for source_name, raw_df in frames:
        df = raw_df.copy()
        if df.empty:
            continue
        if 'status' in df.columns:
            df = df.loc[df['status'] == 'ok'].copy()
        if df.empty:
            continue
        df = _sort_score_frame(df)
        for _, series in df.iterrows():
            row = series.to_dict()
            run_dir = _resolve_run_dir(hub_root=hub_root, row=row)
            if run_dir is not None:
                row['selection_source'] = source_name
                return row, run_dir
            run_id = str(row.get('run_id', '') or '').strip()
            if run_id:
                skipped.append(run_id)
    if skipped:
        raise FileNotFoundError(f'未找到可用 run 目录，已跳过无效 run_id={skipped[:8]}')
    raise RuntimeError('当前没有可用于持仓建议的实验结果。')


def _read_positions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / 'latest_portfolio_v1.csv'
    if not path.exists():
        raise FileNotFoundError(f'未找到最新组合文件: {path}')
    return pd.read_csv(path)


def _diff_positions(prev_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    key_col = 'ts_code' if 'ts_code' in new_df.columns else ('code' if 'code' in new_df.columns else new_df.columns[0])
    prev = prev_df[[key_col, 'portfolio_weight']].copy() if (not prev_df.empty and 'portfolio_weight' in prev_df.columns) else pd.DataFrame(columns=[key_col, 'portfolio_weight'])
    prev = prev.rename(columns={'portfolio_weight': 'prev_weight'})
    now = new_df[[key_col, 'portfolio_weight']].copy()
    now = now.rename(columns={'portfolio_weight': 'target_weight'})
    merged = now.merge(prev, how='outer', on=key_col).fillna(0.0)
    merged['delta_weight'] = merged['target_weight'] - merged['prev_weight']
    merged['action'] = merged['delta_weight'].apply(lambda x: 'buy' if x > 1e-6 else ('sell' if x < -1e-6 else 'hold'))
    return merged.sort_values(['action', 'delta_weight'], ascending=[True, False])


def _symbol_col(df: pd.DataFrame) -> str:
    if 'ts_code' in df.columns:
        return 'ts_code'
    if 'code' in df.columns:
        return 'code'
    return str(df.columns[0])


def _load_performance_feedback(config: Dict[str, Any], bridge_root: Path | None) -> Dict[str, Any]:
    if bridge_root is None:
        raw_root = str(config.get('paths', {}).get('bridge_root', '') or '').strip()
        bridge_root = Path(raw_root) if raw_root else None
    if bridge_root is None:
        return {}
    path = bridge_root / 'performance_feedback.json'
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _portfolio_limits(rec_cfg: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, float]:
    override = dict(feedback.get('portfolio_overrides', {}) or {})
    return {
        'max_names': int(override.get('max_names', rec_cfg.get('max_names', 20)) or 20),
        'single_name_cap': float(override.get('single_name_cap', rec_cfg.get('single_name_cap', 0.10)) or 0.10),
        'total_exposure_cap': float(override.get('total_exposure_cap', rec_cfg.get('total_exposure_cap', 1.0)) or 1.0),
    }


def _load_snapshot_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=['ts_code', 'code', 'price', 'price_date', 'price_source'])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=['ts_code', 'code', 'price', 'price_date', 'price_source'])
    if 'ts_code' not in df.columns and 'code' in df.columns:
        df = df.copy()
        df['ts_code'] = df['code']
    if 'code' not in df.columns and 'ts_code' in df.columns:
        df = df.copy()
        df['code'] = df['ts_code'].map(_ts_to_code)
    df['ts_code'] = df['ts_code'].astype(str).str.strip().str.upper()
    df['code'] = df['code'].map(_ts_to_code)
    if 'price' not in df.columns and 'close' in df.columns:
        df = df.copy()
        df['price'] = df['close']
    if 'date' in df.columns:
        df = df.copy()
        df['price_date'] = df['date']
    else:
        df['price_date'] = ''
    df['price_source'] = 'tushare_snapshot'
    return df[['ts_code', 'code', 'price', 'price_date', 'price_source']].copy()


def _iter_candidate_codes(df: pd.DataFrame) -> Iterable[str]:
    fields = [field for field in ['ts_code', 'code'] if field in df.columns]
    for field in fields:
        for value in df[field].dropna().astype(str).tolist():
            text = value.strip().upper()
            if not text:
                continue
            yield text if '.' in text else text.zfill(6)


def _ts_to_code(ts_code: str) -> str:
    text = str(ts_code or '').strip().upper()
    if not text:
        return ''
    return text.split('.', 1)[0] if '.' in text else text.zfill(6)


def _fallback_enriched_prices(enriched_dir: Path, symbols: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        code = _ts_to_code(symbol)
        if not code:
            continue
        file_path = enriched_dir / f'{code}.csv'
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path, usecols=['date', 'close'])
        except Exception:
            continue
        if df.empty:
            continue
        df = df.dropna(subset=['date', 'close']).sort_values('date')
        if df.empty:
            continue
        last = df.iloc[-1]
        rows.append(
            {
                'ts_code': symbol if '.' in str(symbol) else code,
                'code': code,
                'price': float(last['close']),
                'price_date': str(last['date']),
                'price_source': 'enriched_daily_fallback',
            }
        )
    return pd.DataFrame(rows)


def _attach_price_context(pos_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    market_cfg = dict(config.get('market_pipeline', {}) or {})
    snapshot_path = Path(str(market_cfg.get('price_snapshot_path', '') or '').strip())
    enriched_dir = Path(str(market_cfg.get('enriched_dir', '') or '').strip())
    price_df = _load_snapshot_prices(snapshot_path) if str(snapshot_path) else pd.DataFrame(columns=['ts_code', 'code', 'price', 'price_date', 'price_source'])

    symbols = list(dict.fromkeys(_iter_candidate_codes(pos_df)))
    if str(enriched_dir):
        missing = set(symbols)
        if not price_df.empty:
            missing = {item for item in symbols if item not in set(price_df['ts_code'].astype(str))}
        if missing:
            fallback_df = _fallback_enriched_prices(enriched_dir=enriched_dir, symbols=missing)
            if not fallback_df.empty:
                price_df = pd.concat([price_df, fallback_df], ignore_index=True)

    if price_df.empty:
        out = pos_df.copy()
        out['price'] = pd.NA
        out['price_date'] = ''
        out['price_source'] = ''
        return out

    price_df = price_df.drop_duplicates(subset=['ts_code', 'code'], keep='last').copy()
    out = pos_df.copy()
    key_col = _symbol_col(out)
    if key_col == 'code':
        out[key_col] = out[key_col].map(_ts_to_code)
    else:
        out[key_col] = out[key_col].astype(str).str.strip().str.upper()
    if key_col not in price_df.columns:
        if key_col == 'code' and 'ts_code' in out.columns:
            out['ts_code'] = out['ts_code'].astype(str).str.strip().str.upper()
            out = out.merge(price_df[['ts_code', 'price', 'price_date', 'price_source']], on='ts_code', how='left')
        else:
            out['price'] = pd.NA
            out['price_date'] = ''
            out['price_source'] = ''
            return out
    else:
        out = out.merge(price_df[[key_col, 'price', 'price_date', 'price_source']], on=key_col, how='left')
    if 'close' in out.columns:
        out['price'] = pd.to_numeric(out['price'], errors='coerce').fillna(pd.to_numeric(out['close'], errors='coerce'))
        out['price_source'] = out['price_source'].fillna('').mask(out['price_source'].fillna('').eq('') & out['close'].notna(), 'portfolio_close')
    return out


def build_portfolio_recommendation(config: Dict[str, Any], bridge_root: Path | None = None) -> Dict[str, Any]:
    runtime_cfg = dict(config.get('v5_gpu_runtime', {}) or {})
    rec_cfg = dict(config.get('portfolio_recommendation', {}) or {})
    feedback = _load_performance_feedback(config=config, bridge_root=bridge_root)
    limits = _portfolio_limits(rec_cfg=rec_cfg, feedback=feedback)
    hub_root = Path(str(runtime_cfg.get('hub_output_root', '') or '').strip())
    out_root = ensure_dir(Path(str(config['paths'].get('portfolio_output_root', '') or '').strip()))
    row, run_dir = _pick_best_run(hub_root=hub_root)
    pos_df = _read_positions(run_dir=run_dir)
    max_names = int(limits['max_names'])
    single_name_cap = float(limits['single_name_cap'])
    total_exposure_cap = float(limits['total_exposure_cap'])
    pos_df = pos_df.head(max_names).copy()
    if 'portfolio_weight' in pos_df.columns:
        pos_df['portfolio_weight'] = pos_df['portfolio_weight'].astype(float).clip(upper=single_name_cap)
        total_weight = float(pos_df['portfolio_weight'].sum())
        if total_weight > total_exposure_cap and total_weight > 0:
            pos_df['portfolio_weight'] = pos_df['portfolio_weight'] * (total_exposure_cap / total_weight)
    pos_df = _attach_price_context(pos_df=pos_df, config=config)
    prev_path = out_root / 'target_positions_prev.csv'
    prev_df = pd.read_csv(prev_path) if prev_path.exists() else pd.DataFrame()
    rebalance_df = _diff_positions(prev_df=prev_df, new_df=pos_df)
    if not rebalance_df.empty:
        key_col = _symbol_col(pos_df)
        extra_cols = [col for col in [key_col, 'price', 'price_date', 'price_source'] if col in pos_df.columns]
        rebalance_df = rebalance_df.merge(pos_df[extra_cols], on=key_col, how='left')

    bridge_context = {}
    if bridge_root is not None:
        ctx_path = bridge_root / 'enriched_context.json'
        if ctx_path.exists():
            try:
                bridge_context = json.loads(ctx_path.read_text(encoding='utf-8'))
            except Exception:
                bridge_context = {}

    price_covered = int(pd.to_numeric(pos_df.get('price'), errors='coerce').fillna(0).gt(0).sum()) if 'price' in pos_df.columns else 0
    missing_price_symbols = []
    if 'price' in pos_df.columns:
        key_col = _symbol_col(pos_df)
        missing_price_symbols = pos_df.loc[pd.to_numeric(pos_df['price'], errors='coerce').fillna(0) <= 0, key_col].astype(str).tolist()

    summary = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy_name': str(row.get('strategy_name', '')),
        'strategy_key': str(row.get('strategy_key', '')),
        'run_id': str(row.get('run_id', '')),
        'run_dir': str(run_dir),
        'selection_source': str(row.get('selection_source', 'registry')),
        'total_score': float(row.get('total_score', 0.0) or 0.0),
        'sharpe': float(row.get('sharpe', 0.0) or 0.0),
        'max_drawdown': float(row.get('max_drawdown', 0.0) or 0.0),
        'gpu_used': bool(row.get('gpu_used', False)),
        'n_names': int(len(pos_df)),
        'price_coverage': {
            'covered': price_covered,
            'total': int(len(pos_df)),
            'coverage_ratio': float(price_covered / max(len(pos_df), 1)),
            'missing_symbols': missing_price_symbols,
        },
        'simulation_ready': bool(not rec_cfg.get('simulation_ready_need_gate', False) or float(row.get('total_score', 0.0) or 0.0) >= 45.0),
        'portfolio_limits': limits,
        'performance_feedback': feedback,
        'research_context': bridge_context,
    }

    target_path = out_root / 'target_positions.csv'
    rebalance_path = out_root / 'rebalance_orders.csv'
    summary_path = out_root / 'portfolio_recommendation.json'
    pos_df.to_csv(target_path, index=False, encoding='utf-8-sig')
    rebalance_df.to_csv(rebalance_path, index=False, encoding='utf-8-sig')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pos_df.to_csv(prev_path, index=False, encoding='utf-8-sig')
    return {
        'summary_path': str(summary_path),
        'target_positions_path': str(target_path),
        'rebalance_orders_path': str(rebalance_path),
        'n_names': int(len(pos_df)),
        'run_id': str(row.get('run_id', '')),
    }
