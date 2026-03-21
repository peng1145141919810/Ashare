# -*- coding: utf-8 -*-
"""V5 主控入口。"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hub.candidate_factory import build_cycle_plan
from hub.config_utils import ensure_required_keys, load_config
from hub.data_scout import scout_data_sources
from hub.io_utils import ensure_dir, write_json
from hub.llm_client import LLMClient
from hub.logging_utils import setup_logger
from hub.research_diagnosis import diagnose_research_state
from hub.registry import load_registry
from hub.single_run_v5 import execute_single_experiment_v5
from hub.strategy_family import evolve_strategy_family
from hub.validate import validate_environment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='量化研究中枢 V5')
    p.add_argument('--config', required=True)
    p.add_argument('--mode', default='adaptive_research_brain', choices=['validate_only', 'plan', 'batch', 'adaptive_research_brain'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--max-cycles', type=int, default=None)
    p.add_argument('--sleep-seconds', type=int, default=None)
    return p.parse_args()


def _logger(config: Dict[str, Any], name: str):
    return setup_logger(Path(str(config['hub_output_root'])) / 'logs', name)


def _refresh_family_and_gate(config: Dict[str, Any]) -> Dict[str, Any]:
    hub_root = Path(str(config['hub_output_root']))
    registry_path = hub_root / 'registry' / 'experiment_registry.csv'
    family = evolve_strategy_family(registry_path=registry_path, family_output_dir=hub_root / 'strategy_family', evolution_rules=config.get('evolution_rules', {}))
    gate = {'deployment_ready': False, 'reason': '暂无 champion'}
    state_path = hub_root / 'strategy_family' / 'strategy_family_state.csv'
    if state_path.exists():
        import pandas as pd
        df = pd.read_csv(state_path)
        champ = df.loc[df['current_role'] == 'champion'].copy() if not df.empty else df
        if not champ.empty:
            champ = champ.sort_values(['stability_score', 'latest_total_score'], ascending=[False, False])
            row = champ.iloc[0]
            rule = config.get('deployment_gate', {})
            ready = float(row.get('latest_total_score', 0.0)) >= float(rule.get('min_total_score', 45.0)) and float(row.get('latest_sharpe', 0.0)) >= float(rule.get('min_sharpe', 1.0)) and abs(min(float(row.get('latest_max_drawdown', 0.0)), 0.0)) <= float(rule.get('max_drawdown_abs', 0.22))
            gate = {
                'deployment_ready': bool(ready),
                'candidate_strategy_key': str(row.get('strategy_key', '')),
                'candidate_strategy_name': str(row.get('strategy_name', '')),
                'latest_total_score': float(row.get('latest_total_score', 0.0)),
                'latest_sharpe': float(row.get('latest_sharpe', 0.0)),
                'latest_max_drawdown': float(row.get('latest_max_drawdown', 0.0)),
                'reason': '满足部署闸门' if ready else 'champion 尚未满足部署闸门',
            }
    write_json(hub_root / 'strategy_family' / 'deployment_gate.json', gate)
    return {'family': family, 'gate': gate}


def run_validate_only(config: Dict[str, Any]) -> Dict[str, Any]:
    payload = validate_environment(config)
    write_json(Path(str(config['hub_output_root'])) / 'validation.json', payload)
    return payload


def _prepare_cycle(config: Dict[str, Any], cycle_index: int) -> Dict[str, Any]:
    hub_root = Path(str(config['hub_output_root']))
    registry_path = hub_root / 'registry' / 'experiment_registry.csv'
    registry_df = load_registry(registry_path)
    cycle_id = f"cycle_{cycle_index:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cycle_dir = ensure_dir(hub_root / 'cycles' / cycle_id)
    diagnosis = diagnose_research_state(registry_path=registry_path, output_path=cycle_dir / 'diagnosis.json')
    llm_client = LLMClient(config.get('llm_brain', {}))
    scout_result = scout_data_sources(config, cycle_dir=cycle_dir)
    plan = build_cycle_plan(base_config=config, registry_df=registry_df, cycle_index=cycle_index, diagnosis=diagnosis, llm_client=llm_client, cycle_dir=cycle_dir)
    return {'hub_root': hub_root, 'registry_path': registry_path, 'registry_df': registry_df, 'cycle_dir': cycle_dir, 'cycle_id': cycle_id, 'diagnosis': diagnosis, 'scout_result': scout_result, 'llm_client': llm_client, 'plan': plan}


def run_plan_only(config: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _prepare_cycle(config, cycle_index=1)
    return {
        'cycle_id': ctx['plan']['cycle_id'],
        'cycle_dir': str(ctx['plan']['cycle_dir']),
        'n_candidates': len(ctx['plan']['candidate_configs']),
        'budget': ctx['plan']['budget'],
        'diagnosis': ctx['diagnosis'],
    }


def _write_live_progress(hub_root: Path, payload: Dict[str, Any]) -> None:
    write_json(hub_root / 'live_progress.json', payload)


def run_batch(config: Dict[str, Any], dry_run: bool, cycle_index: int) -> Dict[str, Any]:
    log = _logger(config, f'brain_cycle_{cycle_index:03d}')
    ctx = _prepare_cycle(config, cycle_index=cycle_index)
    plan = ctx['plan']
    hub_root = Path(str(config['hub_output_root']))
    results = []
    total = len(plan['candidate_configs'])
    cycle_start = time.time()
    heartbeat_seconds = int(config.get('execution', {}).get('heartbeat_seconds', 30) or 30)
    log.info('本轮计划已生成。cycle=%s n_candidates=%s budget=%s', plan['cycle_id'], total, plan['budget'])
    _write_live_progress(hub_root, {
        'cycle_id': plan['cycle_id'],
        'cycle_index': cycle_index,
        'status': 'running',
        'n_total': total,
        'n_completed': 0,
        'percent': 0.0,
        'current_candidate_index': 0,
        'current_candidate_name': '',
        'budget': plan['budget'],
        'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_update_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    for idx, cfg in enumerate(plan['candidate_configs'], start=1):
        c = cfg['candidate']
        cand_start = time.time()
        log.info('开始执行候选实验 %s/%s: %s | route=%s model=%s feature=%s logic=%s', idx, total, c['strategy_name'], c['research_route'], c['model_family'], c['feature_profile'], c['training_logic'])
        _write_live_progress(hub_root, {
            'cycle_id': plan['cycle_id'],
            'cycle_index': cycle_index,
            'status': 'running',
            'n_total': total,
            'n_completed': idx - 1,
            'percent': round((idx - 1) / max(total, 1) * 100.0, 2),
            'current_candidate_index': idx,
            'current_candidate_name': c['strategy_name'],
            'current_route': c['research_route'],
            'current_model_family': c['model_family'],
            'current_feature_profile': c['feature_profile'],
            'current_training_logic': c['training_logic'],
            'budget': plan['budget'],
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_update_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
        ret = execute_single_experiment_v5(cfg, dry_run=dry_run, logger=log, heartbeat_seconds=heartbeat_seconds)
        results.append(ret['record'])
        elapsed = float(ret.get('elapsed_seconds', time.time() - cand_start))
        completed = len(results)
        avg_elapsed = sum(float(r.get('elapsed_seconds', 0.0) or 0.0) for r in results) / max(completed, 1)
        eta_seconds = avg_elapsed * max(total - completed, 0)
        status = str(ret['record'].get('status', 'unknown'))
        score = float(ret['record'].get('total_score', 0.0) or 0.0)
        log.info('完成候选实验 %s/%s | status=%s score=%.2f elapsed=%.1fs avg=%.1fs eta=%.1fs progress=%.1f%%', completed, total, status, score, elapsed, avg_elapsed, eta_seconds, completed / max(total, 1) * 100.0)
        _write_live_progress(hub_root, {
            'cycle_id': plan['cycle_id'],
            'cycle_index': cycle_index,
            'status': 'running',
            'n_total': total,
            'n_completed': completed,
            'percent': round(completed / max(total, 1) * 100.0, 2),
            'current_candidate_index': min(idx + 1, total),
            'current_candidate_name': '' if completed >= total else plan['candidate_configs'][idx]['candidate']['strategy_name'],
            'last_finished_candidate_name': c['strategy_name'],
            'last_finished_status': status,
            'last_finished_score': score,
            'last_finished_elapsed_seconds': elapsed,
            'avg_elapsed_seconds': avg_elapsed,
            'eta_seconds': eta_seconds,
            'budget': plan['budget'],
            'last_update_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
    family_gate = _refresh_family_and_gate(config)
    summary = {
        'cycle_id': plan['cycle_id'],
        'cycle_index': cycle_index,
        'n_candidates': total,
        'budget': plan['budget'],
        'diagnosis': ctx['diagnosis'],
        'results': results,
        'family': family_gate['family'],
        'gate': family_gate['gate'],
        'stop_reason': '',
        'cycle_elapsed_seconds': time.time() - cycle_start,
    }
    write_json(ctx['cycle_dir'] / 'cycle_summary.json', summary)
    _write_live_progress(hub_root, {
        'cycle_id': plan['cycle_id'],
        'cycle_index': cycle_index,
        'status': 'completed',
        'n_total': total,
        'n_completed': total,
        'percent': 100.0,
        'budget': plan['budget'],
        'cycle_elapsed_seconds': time.time() - cycle_start,
        'last_update_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    return summary


def run_adaptive_research_brain(config: Dict[str, Any], dry_run: bool, max_cycles: Optional[int], sleep_seconds: Optional[int]) -> Dict[str, Any]:
    log = _logger(config, 'adaptive_research_brain')
    hub_root = Path(str(config['hub_output_root']))
    ensure_dir(hub_root)

    brain_cfg = dict(config.get('research_brain', {}))
    if max_cycles is None:
        max_cycles = brain_cfg.get('max_cycles')
    if sleep_seconds is None:
        sleep_seconds = int(brain_cfg.get('sleep_seconds', 0) or 0)

    cycle = 1
    history: List[Dict[str, Any]] = []
    stop_reason = ''
    while True:
        summary = run_batch(config=config, dry_run=dry_run, cycle_index=cycle)
        top_score = max([float(r.get('total_score', 0.0) or 0.0) for r in summary.get('results', [])] + [0.0])
        history.append({
            'cycle_index': cycle,
            'cycle_id': summary['cycle_id'],
            'n_candidates': summary['n_candidates'],
            'top_score': top_score,
            'budget': summary['budget'],
            'issues': summary['diagnosis'].get('issues', []),
        })
        write_json(hub_root / 'controller_state.json', {
            'current_cycle_index': cycle,
            'last_cycle_id': summary['cycle_id'],
            'last_budget': summary['budget'],
            'last_issues': summary['diagnosis'].get('issues', []),
            'history': history,
            'stop_reason': '',
        })
        if max_cycles is not None and cycle >= int(max_cycles):
            stop_reason = 'max_cycles_reached'
            break
        cycle += 1
        if sleep_seconds > 0:
            log.info('本轮结束，休眠 %ss 后进入下一轮。', sleep_seconds)
            time.sleep(sleep_seconds)
    final = {'n_cycles': len(history), 'history': history, 'stop_reason': stop_reason}
    write_json(hub_root / 'adaptive_loop_final.json', final)
    write_json(hub_root / 'controller_state.json', {
        'current_cycle_index': cycle,
        'last_cycle_id': history[-1]['cycle_id'] if history else '',
        'last_budget': history[-1]['budget'] if history else {},
        'last_issues': history[-1]['issues'] if history else [],
        'history': history,
        'stop_reason': stop_reason,
    })
    log.info('研究脑闭环结束。cycles=%s stop_reason=%s', len(history), stop_reason or 'manual_or_none')
    return final


def run_main(config: Dict[str, Any], mode: str, dry_run: bool, max_cycles: Optional[int], sleep_seconds: Optional[int]) -> Dict[str, Any]:
    ensure_required_keys(config)
    if mode == 'validate_only':
        return run_validate_only(config)
    if mode == 'plan':
        return run_plan_only(config)
    if mode == 'batch':
        return run_batch(config, dry_run=dry_run, cycle_index=1)
    return run_adaptive_research_brain(config, dry_run=dry_run, max_cycles=max_cycles, sleep_seconds=sleep_seconds)


def run_local() -> Dict[str, Any]:
    from hub import local_settings
    cfg = load_config(local_settings.CONFIG_PATH)
    return run_main(cfg, mode=local_settings.MODE, dry_run=bool(local_settings.DRY_RUN), max_cycles=local_settings.MAX_CYCLES, sleep_seconds=local_settings.SLEEP_SECONDS)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    result = run_main(cfg, mode=args.mode, dry_run=bool(args.dry_run), max_cycles=args.max_cycles, sleep_seconds=args.sleep_seconds)
    print(result)


if __name__ == '__main__':
    main()
