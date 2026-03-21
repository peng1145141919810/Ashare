# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, subprocess, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from .config_utils import ensure_dir, load_config
from .logging_utils import log_line
from .market_pipeline import run_market_pipeline
from .orchestrator_v6 import run_v6_cycle
from .portfolio_recommendation import build_portfolio_recommendation

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
    subprocess.run([pyexe, str(script)], cwd=str(v5_root), check=True, env=env)

def _write_supervisor_state(config: Dict[str, Any], payload: Dict[str, Any]) -> None:
    root = ensure_dir(Path(str(config['paths']['research_root'])) / 'supervisor')
    (root / 'supervisor_state.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def _build_execution_runtime_config(config: Dict[str, Any]) -> Path:
    exec_cfg = dict(config.get('execution_bridge', {}) or {})
    template_path = Path(str(exec_cfg['config_template_path']))
    payload = json.loads(template_path.read_text(encoding='utf-8'))
    payload['portfolio_root'] = str(config['paths'].get('portfolio_output_root', payload.get('portfolio_root', '')))
    payload['explicit_portfolio_path'] = str(Path(str(config['paths']['portfolio_output_root'])) / 'target_positions.csv')
    payload['price_snapshot_path'] = str(config.get('market_pipeline', {}).get('price_snapshot_path', payload.get('price_snapshot_path', '')))
    payload['output_dir'] = str(config['paths'].get('live_execution_root', payload.get('output_dir', '')))
    out_path = Path(str(exec_cfg['autogen_config_path']))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return out_path

def _run_execution_bridge(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    exec_cfg = dict(config.get('execution_bridge', {}) or {})
    runtime_config_path = _build_execution_runtime_config(config)
    pyexe = str(exec_cfg['python_executable'])
    script = Path(str(exec_cfg['script_path']))
    env = os.environ.copy()
    proc = subprocess.run(
        [pyexe, str(script), '--config', str(runtime_config_path)],
        cwd=str(project_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = (proc.stdout or '').strip()
    try:
        return json.loads(stdout) if stdout else {'ok': True, 'stdout': ''}
    except Exception:
        return {'ok': True, 'stdout': stdout}


def run_resume_downstream(config_path: Path, include_execution: bool = False) -> None:
    """从最近一次已完成的 V5 结果继续，生成持仓建议，并可选重跑执行桥。"""
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent
    state: Dict[str, Any] = {'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'resume_mode': 'downstream_only'}
    if bool(config.get('portfolio_recommendation', {}).get('enabled', False)):
        rec = build_portfolio_recommendation(config=config, bridge_root=Path(str(config['paths']['bridge_root'])))
        log_line(config, f"Supervisor: 断点续跑已生成持仓建议，run_id={rec.get('run_id')} n_names={rec.get('n_names')}")
        state['portfolio_recommendation'] = rec
    else:
        state['portfolio_recommendation_skipped'] = 'disabled'

    if include_execution and bool(config.get('execution_bridge', {}).get('enabled', False)):
        summary_path = Path(str(config['paths']['portfolio_output_root'])) / 'portfolio_recommendation.json'
        should_trade = True
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            should_trade = bool(summary.get('simulation_ready', True))
        if should_trade:
            state['execution_bridge'] = _run_execution_bridge(config=config, project_root=project_root)
            log_line(config, 'Supervisor: 断点续跑执行桥已完成。')
        else:
            log_line(config, 'Supervisor: 断点续跑 simulation_ready=False，跳过执行桥。')
            state['execution_bridge_skipped'] = 'simulation_ready_false'
    elif include_execution:
        state['execution_bridge_skipped'] = 'disabled'
    else:
        state['execution_bridge_skipped'] = 'resume_without_execution'
    _write_supervisor_state(config, state)

def run_integrated_supervisor(config_path: Path) -> None:
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent
    _check_gpu_requirement(config)
    sup = dict(config.get('supervisor', {}))
    max_ticks = int(sup.get('max_ticks', 1) or 1)
    run_forever = bool(sup.get('run_forever', False))
    sleep_seconds = int(sup.get('sleep_seconds', 300) or 300)
    tick = 0
    while True:
        tick += 1
        state: Dict[str, Any] = {'tick': tick, 'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        portfolio_ready = not bool(config.get('portfolio_recommendation', {}).get('enabled', False))
        if bool(config.get('market_pipeline', {}).get('enabled', False)):
            try:
                state['market_pipeline'] = run_market_pipeline(config=config)
            except Exception as exc:
                log_line(config, f"Supervisor: 市场数据流水线失败：{exc}")
                state['market_pipeline_error'] = str(exc)
        try:
            feedback = _build_strategy_feedback(config)
            _write_strategy_feedback(config, feedback)
            state['strategy_feedback'] = feedback
            log_line(config, f"Supervisor: 策略反馈已刷新，regime={feedback.get('regime', 'neutral')}")
        except Exception as exc:
            feedback = _default_strategy_feedback(config=config, equity_curve_path=Path(str(config['paths'].get('live_execution_root', ''))) / 'equity_curve.csv')
            state['strategy_feedback_error'] = str(exc)
            log_line(config, f"Supervisor: 策略反馈生成失败：{exc}")
        if _should_run_token_plan(config):
            log_line(config, 'Supervisor: 开始运行 V6 研究计划层。')
            run_v6_cycle(config_path=config_path, mode='full_cycle')
            _write_stamp(config)
            state['v6_ran'] = True
        else:
            log_line(config, 'Supervisor: 本 tick 跳过 V6，沿用 24 小时内最近一次研究计划。')
            state['v6_ran'] = False
        log_line(config, 'Supervisor: 开始运行 V5.1 GPU 循环研究。')
        _run_v5_gpu(config, project_root, feedback=feedback)
        state['v5_gpu_completed'] = True
        if bool(config.get('portfolio_recommendation', {}).get('enabled', False)):
            try:
                rec = build_portfolio_recommendation(config=config, bridge_root=Path(str(config['paths']['bridge_root'])))
                log_line(config, f"Supervisor: 持仓建议已生成，run_id={rec.get('run_id')} n_names={rec.get('n_names')}")
                state['portfolio_recommendation'] = rec
                portfolio_ready = True
            except Exception as exc:
                log_line(config, f"Supervisor: 持仓建议生成失败：{exc}")
                state['portfolio_recommendation_error'] = str(exc)
                portfolio_ready = False
        if bool(config.get('execution_bridge', {}).get('enabled', False)):
            try:
                if not portfolio_ready:
                    log_line(config, 'Supervisor: 持仓建议未成功生成，本轮跳过执行桥以避免沿用旧文件。')
                    state['execution_bridge_skipped'] = 'portfolio_recommendation_failed'
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
                if should_trade:
                    state['execution_bridge'] = _run_execution_bridge(config=config, project_root=project_root)
                    log_line(config, 'Supervisor: 执行桥已完成。')
                    feedback = _build_strategy_feedback(config)
                    _write_strategy_feedback(config, feedback)
                    state['strategy_feedback_post_trade'] = feedback
                else:
                    log_line(config, 'Supervisor: 本轮 simulation_ready=False，跳过执行桥。')
                    state['execution_bridge_skipped'] = 'simulation_ready_false'
            except Exception as exc:
                log_line(config, f"Supervisor: 执行桥失败：{exc}")
                state['execution_bridge_error'] = str(exc)
        _write_supervisor_state(config, state)
        if (not run_forever) and tick >= max_ticks:
            break
        time.sleep(sleep_seconds)
