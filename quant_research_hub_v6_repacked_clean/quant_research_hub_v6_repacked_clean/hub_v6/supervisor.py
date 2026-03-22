# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, subprocess, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from .config_utils import ensure_dir, load_config
from .execution_bridge_runner import (
    build_execution_runtime_config as materialize_execution_runtime_config,
    execution_policy,
    run_execution_bridge as dispatch_execution_bridge,
)
from .local_augmentations import build_v5_cycle_review, emit_runtime_stage_note
from .logging_utils import log_line
from .market_pipeline import run_market_pipeline
from .orchestrator_v6 import run_v6_cycle
from .portfolio_recommendation import build_portfolio_recommendation
from .portfolio_release import publish_portfolio_release


def _now_text() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _check_gpu_requirement(config: Dict[str, Any]) -> None:
    if not bool(config.get('supervisor', {}).get('require_gpu', False)):
        return
    try:
        import xgboost as xgb
        _ = xgb.__version__
    except Exception as exc:
        raise RuntimeError('未检测到 xgboost，GPU 训练链路无法启动。') from exc

def _stamp_path(config: Dict[str, Any]) -> Path:
    root = Path(str(config['paths']['bridge_root']))
    ensure_dir(root)
    return root / 'last_token_plan.json'

def _should_run_token_plan(config: Dict[str, Any]) -> bool:
    stamp = _stamp_path(config)
    if not stamp.exists():
        return True
    try:
        payload = json.loads(stamp.read_text(encoding='utf-8'))
        ts = datetime.fromisoformat(str(payload.get('timestamp')))
    except Exception:
        return True
    raw_hours = config.get('supervisor', {}).get('token_plan_min_interval_hours', 24)
    hours = 24.0 if raw_hours in (None, "") else float(raw_hours)
    return datetime.now() - ts >= timedelta(hours=hours)

def _write_stamp(config: Dict[str, Any]) -> None:
    _stamp_path(config).write_text(json.dumps({'timestamp': datetime.now().isoformat()}, ensure_ascii=False, indent=2), encoding='utf-8')


def _strategy_feedback_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    bridge_root = ensure_dir(Path(str(config['paths']['bridge_root'])))
    supervisor_root = ensure_dir(Path(str(config['paths']['research_root'])) / 'supervisor')
    return {
        'bridge': bridge_root / 'performance_feedback.json',
        'supervisor': supervisor_root / 'performance_feedback.json',
    }


def _default_strategy_feedback(config: Dict[str, Any], equity_curve_path: Path) -> Dict[str, Any]:
    return {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'available': False,
        'source_equity_curve': str(equity_curve_path),
        'regime': 'neutral',
        'metrics': {},
        'route_weights': {},
        'route_space_overrides': {},
        'preferred_model_families': [],
        'ban_model_families': [],
        'strategy_overrides': {},
        'portfolio_overrides': {},
    }


def _build_strategy_feedback(config: Dict[str, Any]) -> Dict[str, Any]:
    equity_curve_path = Path(str(config['paths'].get('live_execution_root', ''))) / 'equity_curve.csv'
    feedback = _default_strategy_feedback(config=config, equity_curve_path=equity_curve_path)
    dynamic_cfg = dict(config.get('dynamic_strategy', {}) or {})
    if not bool(dynamic_cfg.get('enabled', False)) or not equity_curve_path.exists():
        return feedback
    try:
        curve = pd.read_csv(equity_curve_path)
    except Exception:
        return feedback
    if curve.empty or 'nav' not in curve.columns:
        return feedback
    curve = curve.copy()
    ts_col = 'timestamp' if 'timestamp' in curve.columns else curve.columns[0]
    curve[ts_col] = pd.to_datetime(curve[ts_col], errors='coerce')
    curve['nav'] = pd.to_numeric(curve['nav'], errors='coerce')
    curve = curve.dropna(subset=[ts_col, 'nav']).sort_values(ts_col).reset_index(drop=True)
    lookback_days = int(dynamic_cfg.get('lookback_days', 5) or 5)
    curve = curve.tail(max(lookback_days + 1, 2)).reset_index(drop=True)
    if curve.empty:
        return feedback

    latest_nav = float(curve['nav'].iloc[-1])
    prev_nav = float(curve['nav'].iloc[-2]) if len(curve) >= 2 else latest_nav
    day_ret = latest_nav / prev_nav - 1.0 if prev_nav > 0 else 0.0
    ret_3d = latest_nav / float(curve['nav'].iloc[-4]) - 1.0 if len(curve) >= 4 and float(curve['nav'].iloc[-4]) > 0 else day_ret
    ret_5d = latest_nav / float(curve['nav'].iloc[-6]) - 1.0 if len(curve) >= 6 and float(curve['nav'].iloc[-6]) > 0 else ret_3d
    rolling_peak = pd.to_numeric(curve['nav'], errors='coerce').cummax()
    current_drawdown = latest_nav / float(rolling_peak.iloc[-1]) - 1.0 if float(rolling_peak.iloc[-1]) > 0 else 0.0

    def_day = float(dynamic_cfg.get('defensive_daily_return_threshold', -0.02) or -0.02)
    def_3d = float(dynamic_cfg.get('defensive_three_day_return_threshold', -0.03) or -0.03)
    agg_day = float(dynamic_cfg.get('aggressive_daily_return_threshold', 0.015) or 0.015)
    agg_3d = float(dynamic_cfg.get('aggressive_three_day_return_threshold', 0.02) or 0.02)

    regime = 'neutral'
    if day_ret <= def_day or ret_3d <= def_3d or current_drawdown <= -0.05:
        regime = 'defensive'
    elif day_ret >= agg_day and ret_3d >= agg_3d and current_drawdown > -0.03:
        regime = 'aggressive'

    if regime == 'defensive':
        route_weights = {'risk': 0.28, 'portfolio': 0.2, 'data': 0.18, 'model': 0.12, 'feature': 0.1, 'training': 0.07, 'hybrid': 0.05}
        route_space_overrides = {
            'top_ks': [10, 15, 20],
            'base_exposures': [0.75, 0.65, 0.55],
            'weak_exposures': [0.25, 0.2, 0.15],
            'model_families': ['ridge_ranker', 'lightgbm_auto', 'xgboost_gpu'],
        }
        preferred_models = ['ridge_ranker', 'lightgbm_auto', 'xgboost_gpu']
        ban_models = ['generated_family']
        strategy_overrides = {
            'top_k': 15,
            'portfolio_base_exposure': 0.75,
            'portfolio_weak_market_exposure': 0.25,
            'portfolio_single_name_cap': 0.08,
        }
        portfolio_overrides = {'max_names': 12, 'single_name_cap': 0.08, 'total_exposure_cap': 0.85}
    elif regime == 'aggressive':
        route_weights = {'model': 0.24, 'feature': 0.2, 'training': 0.18, 'hybrid': 0.15, 'portfolio': 0.1, 'risk': 0.07, 'data': 0.06}
        route_space_overrides = {
            'top_ks': [15, 20, 25],
            'base_exposures': [1.0, 0.95, 0.9],
            'weak_exposures': [0.55, 0.45, 0.35],
            'model_families': ['xgboost_gpu', 'lightgbm_auto', 'ridge_ranker'],
        }
        preferred_models = ['xgboost_gpu', 'lightgbm_auto', 'ridge_ranker']
        ban_models = []
        strategy_overrides = {
            'top_k': 20,
            'portfolio_base_exposure': 1.0,
            'portfolio_weak_market_exposure': 0.55,
            'portfolio_single_name_cap': 0.1,
        }
        portfolio_overrides = {'max_names': 20, 'single_name_cap': 0.1, 'total_exposure_cap': 1.0}
    else:
        route_weights = {'feature': 0.18, 'model': 0.18, 'training': 0.16, 'portfolio': 0.16, 'risk': 0.14, 'data': 0.1, 'hybrid': 0.08}
        route_space_overrides = {
            'top_ks': [15, 20, 30],
            'base_exposures': [1.0, 0.9, 0.8],
            'weak_exposures': [0.5, 0.4, 0.3],
        }
        preferred_models = ['xgboost_gpu', 'ridge_ranker', 'lightgbm_auto']
        ban_models = []
        strategy_overrides = {
            'top_k': 20,
            'portfolio_base_exposure': 1.0,
            'portfolio_weak_market_exposure': 0.5,
            'portfolio_single_name_cap': 0.1,
        }
        portfolio_overrides = {'max_names': 20, 'single_name_cap': 0.1, 'total_exposure_cap': 1.0}

    feedback.update(
        {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'available': True,
            'regime': regime,
            'metrics': {
                'latest_nav': latest_nav,
                'daily_return': day_ret,
                'three_day_return': ret_3d,
                'five_day_return': ret_5d,
                'current_drawdown': current_drawdown,
                'samples': int(len(curve)),
            },
            'route_weights': route_weights,
            'route_space_overrides': route_space_overrides,
            'preferred_model_families': preferred_models,
            'ban_model_families': ban_models,
            'strategy_overrides': strategy_overrides,
            'portfolio_overrides': portfolio_overrides,
        }
    )
    return feedback


def _write_strategy_feedback(config: Dict[str, Any], feedback: Dict[str, Any]) -> None:
    for path in _strategy_feedback_paths(config).values():
        path.write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding='utf-8')


def _apply_strategy_feedback(template: Dict[str, Any], feedback: Dict[str, Any]) -> None:
    strategy_override = dict(feedback.get('strategy_overrides', {}) or {})
    route_space_override = dict(feedback.get('route_space_overrides', {}) or {})
    if strategy_override:
        template.setdefault('strategy', {}).update(strategy_override)
    if route_space_override:
        route_space = template.setdefault('route_space', {})
        for key, value in route_space_override.items():
            if value not in (None, []):
                route_space[key] = value
    preferred_models = list(feedback.get('preferred_model_families', []) or [])
    ban_models = set(feedback.get('ban_model_families', []) or [])
    if preferred_models:
        route_space = template.setdefault('route_space', {})
        current = list(route_space.get('model_families', []) or [])
        route_space['model_families'] = [m for m in list(dict.fromkeys(preferred_models + current)) if m not in ban_models]

def _build_v5_gpu_config(config: Dict[str, Any], project_root: Path, feedback: Dict[str, Any] | None = None) -> Path:
    v5_root = project_root / 'v5_gpu_runtime'
    template = json.loads((v5_root / 'configs' / 'hub_config.v5_1.local.json').read_text(encoding='utf-8'))
    runtime_cfg = dict(config.get('v5_gpu_runtime', {}))
    template['project_root'] = str(runtime_cfg['project_root'])
    template['train_table_dir'] = str(runtime_cfg['train_table_dir'])
    template['hub_output_root'] = str(runtime_cfg['hub_output_root'])
    template['execution']['python_executable'] = str(runtime_cfg['python_executable'])
    template['research_brain']['max_cycles'] = int(config.get('supervisor', {}).get('v5_gpu_max_cycles_per_tick', 8) or 8)
    template['research_brain']['sleep_seconds'] = 0
    template['llm_brain']['enabled'] = True
    template['llm_brain']['api_key_env'] = str(config['providers']['deepseek_worker']['api_key_env'])
    template['llm_brain']['base_url'] = str(config['providers']['deepseek_worker']['base_url'])
    template['llm_brain']['model'] = str(config['providers']['deepseek_worker']['model'])
    template['llm_brain']['timeout_seconds'] = int(config['providers']['deepseek_worker'].get('timeout_seconds', 90) or 90)
    template['llm_brain']['temperature'] = 0.15
    template['bridge_inputs'] = {'enabled': True, 'bridge_root': str(runtime_cfg['bridge_input_root'])}
    if feedback:
        _apply_strategy_feedback(template, feedback)
    out_path = v5_root / 'configs' / 'hub_config.v5_1.integrated_gpu.json'
    out_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding='utf-8')
    local_settings = v5_root / 'hub' / 'local_settings.py'
    local_settings.write_text(
        "# -*- coding: utf-8 -*-\nCONFIG_PATH = r\"configs/hub_config.v5_1.integrated_gpu.json\"\nMODE = \"adaptive_research_brain\"\nDRY_RUN = False\nMAX_CYCLES = %d\nSLEEP_SECONDS = 0\n" % int(config.get('supervisor', {}).get('v5_gpu_max_cycles_per_tick', 8) or 8),
        encoding='utf-8'
    )
    return out_path

def _run_v5_gpu(config: Dict[str, Any], project_root: Path, feedback: Dict[str, Any] | None = None) -> None:
    v5_root = project_root / 'v5_gpu_runtime'
    _build_v5_gpu_config(config, project_root, feedback=feedback)
    pyexe = str(config.get('v5_gpu_runtime', {}).get('python_executable'))
    script = v5_root / 'run_research_hub_v5_1_local.py'
    env = os.environ.copy()
    output_root = str(config.get('v5_gpu_runtime', {}).get('hub_output_root', '') or '')
    log_line(
        config,
        f"Supervisor: V5.1 研究进程已启动，输出根={output_root}，可观察 controller_state.json / registry/experiment_registry.csv / cycles/*/cycle_summary.json",
    )
    subprocess.run([pyexe, str(script)], cwd=str(v5_root), check=True, env=env)

def _write_supervisor_state(config: Dict[str, Any], payload: Dict[str, Any]) -> None:
    root = ensure_dir(Path(str(config['paths']['research_root'])) / 'supervisor')
    payload['updated_at'] = _now_text()
    (root / 'supervisor_state.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _compact_stage_summary(value: Any, max_chars: int = 240) -> str:
    if value in (None, ''):
        return ''
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    return text[:max_chars]


def _stage_bookkeeping(state: Dict[str, Any], stage_name: str, stage_label: str, stage_order: int, stage_total: int, status: str, summary: str = '') -> None:
    stages = dict(state.get('stages', {}) or {})
    stage_payload = dict(stages.get(stage_name, {}) or {})
    if status == 'running' and 'started_at' not in stage_payload:
        stage_payload['started_at'] = _now_text()
    if status in {'completed', 'failed', 'skipped'}:
        stage_payload['completed_at'] = _now_text()
    stage_payload.update({
        'label': stage_label,
        'order': stage_order,
        'total': stage_total,
        'status': status,
    })
    if summary:
        stage_payload['summary'] = summary
    stages[stage_name] = stage_payload
    state['stages'] = stages
    state['current_stage'] = {
        'name': stage_name,
        'label': stage_label,
        'order': stage_order,
        'total': stage_total,
        'status': status,
        'updated_at': _now_text(),
        'summary': summary,
    }
    history = list(state.get('stage_history', []) or [])
    history.append({
        'timestamp': _now_text(),
        'stage': stage_name,
        'label': stage_label,
        'order': stage_order,
        'total': stage_total,
        'status': status,
        'summary': summary,
    })
    state['stage_history'] = history[-50:]


def _stage_start(config: Dict[str, Any], state: Dict[str, Any], stage_name: str, stage_label: str, stage_order: int, stage_total: int) -> None:
    _stage_bookkeeping(state, stage_name, stage_label, stage_order, stage_total, 'running')
    note = emit_runtime_stage_note(config=config, stage_name=stage_name, stage_label=stage_label, status='running')
    if note:
        runtime_notes = list(state.get('runtime_notes', []) or [])
        runtime_notes.append(note)
        state['runtime_notes'] = runtime_notes[-20:]
    _write_supervisor_state(config, state)
    log_line(config, f"Supervisor: [{stage_order}/{stage_total}] {stage_label} 开始。")


def _stage_finish(config: Dict[str, Any], state: Dict[str, Any], stage_name: str, stage_label: str, stage_order: int, stage_total: int, summary: str = '') -> None:
    _stage_bookkeeping(state, stage_name, stage_label, stage_order, stage_total, 'completed', summary=summary)
    _write_supervisor_state(config, state)
    msg = f"Supervisor: [{stage_order}/{stage_total}] {stage_label} 完成。"
    if summary:
        msg += f" {summary}"
    log_line(config, msg)


def _stage_skip(config: Dict[str, Any], state: Dict[str, Any], stage_name: str, stage_label: str, stage_order: int, stage_total: int, summary: str = '') -> None:
    _stage_bookkeeping(state, stage_name, stage_label, stage_order, stage_total, 'skipped', summary=summary)
    _write_supervisor_state(config, state)
    msg = f"Supervisor: [{stage_order}/{stage_total}] {stage_label} 跳过。"
    if summary:
        msg += f" {summary}"
    log_line(config, msg)


def _stage_fail(config: Dict[str, Any], state: Dict[str, Any], stage_name: str, stage_label: str, stage_order: int, stage_total: int, summary: str) -> None:
    _stage_bookkeeping(state, stage_name, stage_label, stage_order, stage_total, 'failed', summary=summary)
    _write_supervisor_state(config, state)
    log_line(config, f"Supervisor: [{stage_order}/{stage_total}] {stage_label} 失败。{summary}")


def _maybe_publish_release(config: Dict[str, Any], state: Dict[str, Any], source_mode: str) -> None:
    release_cfg = dict(config.get('trade_release', {}) or {})
    if not bool(release_cfg.get('enabled', True)):
        state['portfolio_release_skipped'] = 'disabled'
        return
    try:
        release = publish_portfolio_release(
            config=config,
            source_mode=source_mode,
            profile=str(config.get('runtime_selection', {}).get('profile', '') or ''),
        )
        state['portfolio_release'] = {
            'release_id': str(release.get('release_id', '') or ''),
            'trade_date': str(release.get('trade_date', '') or ''),
            'manifest_path': str(release.get('artifacts', {}).get('manifest_path', '') or ''),
        }
        log_line(
            config,
            (
                "Supervisor: 已发布 portfolio release "
                f"release_id={state['portfolio_release'].get('release_id', '')} "
                f"trade_date={state['portfolio_release'].get('trade_date', '')}"
            ),
        )
    except Exception as exc:
        state['portfolio_release_error'] = str(exc)
        log_line(config, f"Supervisor: portfolio release 发布失败：{exc}")


def _supervisor_direct_execution_decision(config: Dict[str, Any]) -> Dict[str, Any]:
    policy = execution_policy(config)
    if str(policy.get('account_mode', 'simulation')) != 'precision':
        return {'allowed': True, 'reason': 'simulation_mode'}
    if not bool(policy.get('precision_trade_enabled', False)):
        return {'allowed': False, 'reason': 'precision_trade_disabled'}
    if not bool(policy.get('allow_integrated_precision_execution', False)):
        return {'allowed': False, 'reason': 'precision_mode_deferred_to_execution_only'}
    return {'allowed': True, 'reason': 'precision_mode_allowed'}

def _build_execution_runtime_config(config: Dict[str, Any]) -> Path:
    return materialize_execution_runtime_config(
        config=config,
        explicit_portfolio_path=str(Path(str(config['paths']['portfolio_output_root'])) / 'target_positions.csv'),
    )

def _run_execution_bridge(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    return dispatch_execution_bridge(
        config=config,
        project_root=project_root,
        explicit_portfolio_path=str(Path(str(config['paths']['portfolio_output_root'])) / 'target_positions.csv'),
    )


def run_resume_downstream(config_path: Path, include_execution: bool = False) -> None:
    """从最近一次已完成的 V5 结果继续，生成持仓建议，并可选重跑执行桥。"""
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent
    stage_total = 1 + (1 if include_execution and bool(config.get('execution_bridge', {}).get('enabled', False)) else 0)
    state: Dict[str, Any] = {'started_at': _now_text(), 'resume_mode': 'downstream_only'}
    if bool(config.get('portfolio_recommendation', {}).get('enabled', False)):
        _stage_start(config, state, 'portfolio_recommendation', '断点续跑持仓建议生成', 1, stage_total)
        try:
            rec = build_portfolio_recommendation(config=config, bridge_root=Path(str(config['paths']['bridge_root'])))
            state['portfolio_recommendation'] = rec
            _maybe_publish_release(config=config, state=state, source_mode='resume_downstream')
            _stage_finish(
                config,
                state,
                'portfolio_recommendation',
                '断点续跑持仓建议生成',
                1,
                stage_total,
                summary=(
                    f"run_id={rec.get('run_id')} n_names={rec.get('n_names')} "
                    f"regime={rec.get('market_regime', '')} tech_allow={rec.get('tech_allow_count', 0)}"
                ),
            )
        except Exception as exc:
            state['portfolio_recommendation_error'] = str(exc)
            _stage_fail(config, state, 'portfolio_recommendation', '断点续跑持仓建议生成', 1, stage_total, summary=str(exc))
            raise
    else:
        state['portfolio_recommendation_skipped'] = 'disabled'
        _stage_skip(config, state, 'portfolio_recommendation', '断点续跑持仓建议生成', 1, stage_total, summary='disabled')

    if include_execution and bool(config.get('execution_bridge', {}).get('enabled', False)):
        _stage_start(config, state, 'execution_bridge', '断点续跑执行桥', stage_total, stage_total)
        summary_path = Path(str(config['paths']['portfolio_output_root'])) / 'portfolio_recommendation.json'
        should_trade = True
        direct_exec = _supervisor_direct_execution_decision(config)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            should_trade = bool(summary.get('simulation_ready', True))
        if not bool(direct_exec.get('allowed', False)):
            state['execution_bridge_skipped'] = str(direct_exec.get('reason', 'direct_execution_blocked'))
            _stage_skip(
                config,
                state,
                'execution_bridge',
                '断点续跑执行桥',
                stage_total,
                stage_total,
                summary=str(direct_exec.get('reason', 'direct_execution_blocked')),
            )
        elif should_trade:
            try:
                state['execution_bridge'] = _run_execution_bridge(config=config, project_root=project_root)
                _stage_finish(config, state, 'execution_bridge', '断点续跑执行桥', stage_total, stage_total, summary='execution_bridge_completed')
            except Exception as exc:
                state['execution_bridge_error'] = str(exc)
                _stage_fail(config, state, 'execution_bridge', '断点续跑执行桥', stage_total, stage_total, summary=str(exc))
                raise
        else:
            state['execution_bridge_skipped'] = 'simulation_ready_false'
            _stage_skip(config, state, 'execution_bridge', '断点续跑执行桥', stage_total, stage_total, summary='simulation_ready_false')
    elif include_execution:
        state['execution_bridge_skipped'] = 'disabled'
        _stage_skip(config, state, 'execution_bridge', '断点续跑执行桥', stage_total, stage_total, summary='disabled')
    else:
        state['execution_bridge_skipped'] = 'resume_without_execution'
        if stage_total > 1:
            _stage_skip(config, state, 'execution_bridge', '断点续跑执行桥', stage_total, stage_total, summary='resume_without_execution')
    _write_supervisor_state(config, state)

def run_integrated_supervisor(
    config_path: Path,
    run_mode_label: str = 'integrated_supervisor',
    release_source_mode: str = '',
) -> None:
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent
    _check_gpu_requirement(config)
    effective_release_source_mode = str(release_source_mode or run_mode_label or 'integrated_supervisor')
    sup = dict(config.get('supervisor', {}))
    max_ticks = int(sup.get('max_ticks', 1) or 1)
    run_forever = bool(sup.get('run_forever', False))
    sleep_seconds = int(sup.get('sleep_seconds', 300) or 300)
    stage_defs = []
    if bool(config.get('market_pipeline', {}).get('enabled', False)):
        stage_defs.append(('market_pipeline', '市场数据流水线'))
    stage_defs.append(('strategy_feedback', '策略反馈刷新'))
    stage_defs.append(('v6_planning', 'V6 研究计划'))
    stage_defs.append(('v5_gpu', 'V5.1 GPU 研究'))
    if bool(config.get('portfolio_recommendation', {}).get('enabled', False)):
        stage_defs.append(('portfolio_recommendation', '持仓建议生成'))
    if bool(config.get('execution_bridge', {}).get('enabled', False)):
        stage_defs.append(('execution_bridge', '执行桥'))
    stage_total = len(stage_defs)
    stage_order_map = {name: idx for idx, (name, _) in enumerate(stage_defs, start=1)}
    stage_label_map = {name: label for name, label in stage_defs}
    tick = 0
    while True:
        tick += 1
        state: Dict[str, Any] = {'tick': tick, 'started_at': _now_text(), 'run_mode': str(run_mode_label or 'integrated_supervisor')}
        portfolio_ready = not bool(config.get('portfolio_recommendation', {}).get('enabled', False))
        if bool(config.get('market_pipeline', {}).get('enabled', False)):
            _stage_start(config, state, 'market_pipeline', stage_label_map['market_pipeline'], stage_order_map['market_pipeline'], stage_total)
            try:
                state['market_pipeline'] = run_market_pipeline(config=config)
                _stage_finish(
                    config,
                    state,
                    'market_pipeline',
                    stage_label_map['market_pipeline'],
                    stage_order_map['market_pipeline'],
                    stage_total,
                    summary=_compact_stage_summary(state['market_pipeline']),
                )
            except Exception as exc:
                state['market_pipeline_error'] = str(exc)
                _stage_fail(
                    config,
                    state,
                    'market_pipeline',
                    stage_label_map['market_pipeline'],
                    stage_order_map['market_pipeline'],
                    stage_total,
                    summary=str(exc),
                )
                log_line(config, f"Supervisor: 市场数据流水线失败：{exc}")
        _stage_start(config, state, 'strategy_feedback', stage_label_map['strategy_feedback'], stage_order_map['strategy_feedback'], stage_total)
        try:
            feedback = _build_strategy_feedback(config)
            _write_strategy_feedback(config, feedback)
            state['strategy_feedback'] = feedback
            _stage_finish(
                config,
                state,
                'strategy_feedback',
                stage_label_map['strategy_feedback'],
                stage_order_map['strategy_feedback'],
                stage_total,
                summary=f"regime={feedback.get('regime', 'neutral')}",
            )
        except Exception as exc:
            feedback = _default_strategy_feedback(config=config, equity_curve_path=Path(str(config['paths'].get('live_execution_root', ''))) / 'equity_curve.csv')
            state['strategy_feedback_error'] = str(exc)
            _stage_fail(
                config,
                state,
                'strategy_feedback',
                stage_label_map['strategy_feedback'],
                stage_order_map['strategy_feedback'],
                stage_total,
                summary=str(exc),
            )
            log_line(config, f"Supervisor: 策略反馈生成失败：{exc}")
        v6_stage_order = stage_order_map['v6_planning']
        v6_stage_label = stage_label_map['v6_planning']
        if _should_run_token_plan(config):
            _stage_start(config, state, 'v6_planning', v6_stage_label, v6_stage_order, stage_total)
            try:
                run_v6_cycle(config_path=config_path, mode='full_cycle')
                _write_stamp(config)
                state['v6_ran'] = True
                _stage_finish(config, state, 'v6_planning', v6_stage_label, v6_stage_order, stage_total, summary='full_cycle_completed')
            except Exception as exc:
                state['v6_ran'] = False
                state['v6_error'] = str(exc)
                _stage_fail(config, state, 'v6_planning', v6_stage_label, v6_stage_order, stage_total, summary=str(exc))
                raise
        else:
            state['v6_ran'] = False
            _stage_skip(config, state, 'v6_planning', v6_stage_label, v6_stage_order, stage_total, summary='沿用 24 小时内最近一次研究计划')
        _stage_start(config, state, 'v5_gpu', stage_label_map['v5_gpu'], stage_order_map['v5_gpu'], stage_total)
        try:
            _run_v5_gpu(config, project_root, feedback=feedback)
            state['v5_gpu_completed'] = True
            review_summary = 'review=not_generated'
            try:
                review_result = build_v5_cycle_review(config=config)
                if bool(review_result.get('ok', False)):
                    state['v5_cycle_review'] = dict(review_result.get('review', {}) or {})
                    cycle_id = str(state['v5_cycle_review'].get('cycle_id', '') or '')
                    review_summary = f"review=ok cycle_id={cycle_id}" if cycle_id else 'review=ok'
                else:
                    review_error = str(review_result.get('error', 'review_unavailable') or 'review_unavailable')
                    state['v5_cycle_review_error'] = review_error
                    review_summary = f"review={review_error}"
                    log_line(config, f"Supervisor: V5 本地复盘未生成 {review_error}")
            except Exception as review_exc:
                state['v5_cycle_review_error'] = str(review_exc)
                review_summary = f"review_error={review_exc}"
                log_line(config, f"Supervisor: V5 本地复盘失败：{review_exc}")
            _stage_finish(
                config,
                state,
                'v5_gpu',
                stage_label_map['v5_gpu'],
                stage_order_map['v5_gpu'],
                stage_total,
                summary=(
                    f"hub_output_root={config.get('v5_gpu_runtime', {}).get('hub_output_root', '')} "
                    f"{review_summary}"
                ),
            )
        except Exception as exc:
            state['v5_gpu_error'] = str(exc)
            _stage_fail(
                config,
                state,
                'v5_gpu',
                stage_label_map['v5_gpu'],
                stage_order_map['v5_gpu'],
                stage_total,
                summary=str(exc),
            )
            raise
        if bool(config.get('portfolio_recommendation', {}).get('enabled', False)):
            _stage_start(config, state, 'portfolio_recommendation', stage_label_map['portfolio_recommendation'], stage_order_map['portfolio_recommendation'], stage_total)
            try:
                rec = build_portfolio_recommendation(config=config, bridge_root=Path(str(config['paths']['bridge_root'])))
                state['portfolio_recommendation'] = rec
                portfolio_ready = True
                _maybe_publish_release(config=config, state=state, source_mode=effective_release_source_mode)
                _stage_finish(
                    config,
                    state,
                    'portfolio_recommendation',
                    stage_label_map['portfolio_recommendation'],
                    stage_order_map['portfolio_recommendation'],
                    stage_total,
                    summary=(
                        f"run_id={rec.get('run_id')} n_names={rec.get('n_names')} "
                        f"regime={rec.get('market_regime', '')} tech_allow={rec.get('tech_allow_count', 0)}"
                    ),
                )
            except Exception as exc:
                state['portfolio_recommendation_error'] = str(exc)
                portfolio_ready = False
                _stage_fail(
                    config,
                    state,
                    'portfolio_recommendation',
                    stage_label_map['portfolio_recommendation'],
                    stage_order_map['portfolio_recommendation'],
                    stage_total,
                    summary=str(exc),
                )
                log_line(config, f"Supervisor: 持仓建议生成失败：{exc}")
        if bool(config.get('execution_bridge', {}).get('enabled', False)):
            _stage_start(config, state, 'execution_bridge', stage_label_map['execution_bridge'], stage_order_map['execution_bridge'], stage_total)
            try:
                if not portfolio_ready:
                    state['execution_bridge_skipped'] = 'portfolio_recommendation_failed'
                    _stage_skip(
                        config,
                        state,
                        'execution_bridge',
                        stage_label_map['execution_bridge'],
                        stage_order_map['execution_bridge'],
                        stage_total,
                        summary='portfolio_recommendation_failed',
                    )
                    log_line(config, 'Supervisor: 持仓建议未成功生成，本轮跳过执行桥以避免沿用旧文件。')
                    _write_supervisor_state(config, state)
                    if (not run_forever) and tick >= max_ticks:
                        break
                    time.sleep(sleep_seconds)
                    continue
                summary_path = Path(str(config['paths']['portfolio_output_root'])) / 'portfolio_recommendation.json'
                should_trade = True
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding='utf-8'))
                    should_trade = bool(summary.get('simulation_ready', True))
                direct_exec = _supervisor_direct_execution_decision(config)
                if not bool(direct_exec.get('allowed', False)):
                    state['execution_bridge_skipped'] = str(direct_exec.get('reason', 'direct_execution_blocked'))
                    _stage_skip(
                        config,
                        state,
                        'execution_bridge',
                        stage_label_map['execution_bridge'],
                        stage_order_map['execution_bridge'],
                        stage_total,
                        summary=str(direct_exec.get('reason', 'direct_execution_blocked')),
                    )
                elif should_trade:
                    state['execution_bridge'] = _run_execution_bridge(config=config, project_root=project_root)
                    feedback = _build_strategy_feedback(config)
                    _write_strategy_feedback(config, feedback)
                    state['strategy_feedback_post_trade'] = feedback
                    control_summary = dict(state['execution_bridge'].get('portfolio_control', {}) or {})
                    exec_feedback = dict(control_summary.get('execution_feedback_summary', {}) or {})
                    _stage_finish(
                        config,
                        state,
                        'execution_bridge',
                        stage_label_map['execution_bridge'],
                        stage_order_map['execution_bridge'],
                        stage_total,
                        summary=(
                            f"orders={state['execution_bridge'].get('n_orders', 0)} "
                            f"fills={state['execution_bridge'].get('n_fills', 0)} "
                            f"turnover={float(control_summary.get('final_turnover_ratio', 0.0) or 0.0):.4f} "
                            f"feedback_success={int(exec_feedback.get('n_success', 0) or 0)}"
                        ),
                    )
                else:
                    state['execution_bridge_skipped'] = 'simulation_ready_false'
                    _stage_skip(
                        config,
                        state,
                        'execution_bridge',
                        stage_label_map['execution_bridge'],
                        stage_order_map['execution_bridge'],
                        stage_total,
                        summary='simulation_ready_false',
                    )
            except Exception as exc:
                state['execution_bridge_error'] = str(exc)
                _stage_fail(
                    config,
                    state,
                    'execution_bridge',
                    stage_label_map['execution_bridge'],
                    stage_order_map['execution_bridge'],
                    stage_total,
                    summary=str(exc),
                )
                log_line(config, f"Supervisor: 执行桥失败：{exc}")
        _write_supervisor_state(config, state)
        if (not run_forever) and tick >= max_ticks:
            break
        time.sleep(sleep_seconds)


def run_research_only(config_path: Path) -> None:
    """运行研究链并发布组合 release，不直接触发执行桥。"""
    config = load_config(config_path)
    shadow = json.loads(json.dumps(config, ensure_ascii=False))
    exec_cfg = dict(shadow.get('execution_bridge', {}) or {})
    exec_cfg['enabled'] = False
    shadow['execution_bridge'] = exec_cfg
    temp_path = Path(config_path).with_name(f"{Path(config_path).stem}.research_only.runtime.json")
    temp_path.write_text(json.dumps(shadow, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        run_integrated_supervisor(
            temp_path,
            run_mode_label='research_only',
            release_source_mode='research_only',
        )
    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass


def run_release_only(
    config_path: Path,
    source_mode: str = "release_only",
    summary_path: str = "",
    target_positions_path: str = "",
    note: str = "",
) -> Dict[str, Any]:
    """仅把当前最新组合建议发布为可执行 release。"""
    config = load_config(config_path)
    release = publish_portfolio_release(
        config=config,
        source_mode=str(source_mode or "release_only"),
        profile=str(config.get('runtime_selection', {}).get('profile', '') or ''),
        summary_path=str(summary_path or ""),
        target_positions_path=str(target_positions_path or ""),
        note=str(note or ""),
    )
    log_line(
        config,
        (
            "Supervisor: release_only 完成 "
            f"release_id={release.get('release_id', '')} "
            f"trade_date={release.get('trade_date', '')}"
        ),
    )
    return release
