from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config_utils import ensure_dir, load_config
from .execution_bridge_runner import execution_policy, run_execution_bridge
from .market_state import load_latest_market_state
from .portfolio_release import load_latest_release, load_release_by_id, record_release_execution
from .safety_guard import (
    apply_execution_safety_overrides,
    assess_system_safety,
    load_system_safety_state,
    record_incident,
    save_system_safety_state,
)
from .trading_clock import clock_now, current_execution_window, is_trading_day, market_stage


def _trade_clock_root(config: Dict[str, Any]) -> Path:
    return ensure_dir(Path(str(config.get("paths", {}).get("trade_clock_root", "") or "")).resolve())


def _parse_iso(text: str) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _load_release(config: Dict[str, Any], release_id: str = "") -> Dict[str, Any]:
    if str(release_id).strip():
        return load_release_by_id(config=config, release_id=str(release_id).strip())
    return load_latest_release(config=config)


def _market_state_runtime(release_doc: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(release_doc.get("market_state", {}) or {})
    if payload:
        return payload
    return dict(load_latest_market_state(config=config, allow_build=False) or {})


def _apply_market_state_execution_overrides(config: Dict[str, Any], market_state: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    portfolio_control = dict(updated.get("portfolio_control", {}) or {})
    current_turnover = float(portfolio_control.get("max_daily_turnover_ratio", 0.25) or 0.25)
    turnover_multiplier = float(market_state.get("turnover_multiplier", 1.0) or 1.0)
    portfolio_control["max_daily_turnover_ratio"] = round(max(current_turnover * turnover_multiplier, 0.0), 6)
    if str(market_state.get("new_position_policy", "") or "") in {"no_new_positions", "reduce_only"}:
        portfolio_control["reduce_only"] = True
    updated["portfolio_control"] = portfolio_control
    updated["market_state_runtime"] = {
        "market_regime": str(market_state.get("market_regime", "") or ""),
        "style_bias": str(market_state.get("style_bias", "") or ""),
        "mechanism_bias": str(market_state.get("mechanism_bias", "") or ""),
        "risk_budget_multiplier": float(market_state.get("risk_budget_multiplier", 1.0) or 1.0),
        "turnover_multiplier": turnover_multiplier,
        "entry_strictness": float(market_state.get("entry_strictness", 0.5) or 0.5),
        "new_position_policy": str(market_state.get("new_position_policy", "allow") or "allow"),
    }
    return updated


def assess_execution_gate(
    config: Dict[str, Any],
    release_id: str = "",
    ignore_window: bool = False,
    now: datetime | None = None,
) -> Dict[str, Any]:
    current_dt = now or clock_now(str(config.get("trade_clock", {}).get("timezone", "Asia/Shanghai") or "Asia/Shanghai"))
    policy = execution_policy(config)
    account_mode = str(policy.get("account_mode", "simulation"))
    precision_trade_enabled = bool(policy.get("precision_trade_enabled", False))
    gate: Dict[str, Any] = {
        "now": current_dt.isoformat(timespec="seconds"),
        "market_stage": market_stage(current_dt),
        "ignore_window": bool(ignore_window),
        "account_mode": account_mode,
        "precision_trade_enabled": precision_trade_enabled,
    }
    try:
        release_doc = _load_release(config=config, release_id=release_id)
    except Exception as exc:
        gate.update({
            "ok": False,
            "should_execute": False,
            "reason": f"release_unavailable: {exc}",
            "release": None,
        })
        return gate

    trading_day_info = is_trading_day(config=config, target_date=current_dt.date())
    window = current_execution_window(config=config, now=current_dt)
    valid_after = _parse_iso(str(release_doc.get("valid_after", "") or ""))
    expires_at = _parse_iso(str(release_doc.get("expires_at", "") or ""))
    release_trade_date = str(release_doc.get("trade_date", "") or "")
    simulation_ready = bool(release_doc.get("simulation_ready", True))
    if account_mode == "simulation":
        time_window_ok = True
        valid_after_ok = True
        not_expired = True
        trade_date_ok = True
        calendar_ok = True
        should_execute = bool(simulation_ready)
        reason = "simulation_ready" if should_execute else "simulation_ready_false"
    else:
        time_window_ok = bool(ignore_window or window is not None)
        valid_after_ok = bool(ignore_window or valid_after is None or current_dt >= valid_after)
        not_expired = bool(expires_at is None or current_dt <= expires_at)
        trade_date_ok = bool(release_trade_date == current_dt.date().isoformat())
        calendar_ok = bool(trading_day_info.get("ok", False) and trading_day_info.get("is_trading_day", False))
        should_execute = all([
            precision_trade_enabled,
            calendar_ok,
            trade_date_ok,
            simulation_ready,
            time_window_ok,
            valid_after_ok,
            not_expired,
        ])
        if not precision_trade_enabled:
            reason = "precision_trade_disabled"
        else:
            reason = "eligible" if should_execute else "gate_blocked"
    gate.update(
        {
            "ok": True,
            "should_execute": bool(should_execute),
            "calendar_ok": calendar_ok,
            "release_trade_date": release_trade_date,
            "trade_date_ok": trade_date_ok,
            "simulation_ready": simulation_ready,
            "time_window_ok": time_window_ok,
            "valid_after_ok": valid_after_ok,
            "not_expired": not_expired,
            "active_execution_window": {
                "label": window.label,
                "start": window.start.strftime("%H:%M:%S"),
                "end": window.end.strftime("%H:%M:%S"),
            } if window else None,
            "release": {
                "release_id": str(release_doc.get("release_id", "") or ""),
                "trade_date": release_trade_date,
                "manifest_path": str(release_doc.get("artifacts", {}).get("manifest_path", "") or ""),
                "target_positions_path": str(release_doc.get("artifacts", {}).get("target_positions_path", "") or ""),
                "profile": str(release_doc.get("profile", "") or ""),
                "source_mode": str(release_doc.get("source_mode", "") or ""),
            },
            "reason": reason,
        }
    )
    return gate


def run_execution_only(
    config_path: Path,
    release_id: str = "",
    ignore_window: bool = False,
    gate_only: bool = False,
    trigger_label: str = "manual",
    trigger_source: str = "manual",
) -> Dict[str, Any]:
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent
    policy = execution_policy(config)
    namespace = str(policy.get("namespace", "main") or "main").strip() or "main"
    shadow_run = bool(policy.get("shadow_run", False))
    gate = assess_execution_gate(config=config, release_id=release_id, ignore_window=ignore_window)
    safety = assess_system_safety(
        config=config,
        gate=gate,
        project_root=project_root,
        service_name="execution_only",
        current_mode="execution_only",
        force_account_refresh=bool(not gate_only),
    )
    if gate_only:
        return {
            "ok": bool(gate.get("ok", False)),
            "status": "gate_only",
            "gate": gate,
            "safety": safety,
            "execution_namespace": namespace,
            "shadow_run": shadow_run,
            "market_state": _market_state_runtime(release_doc=_load_release(config=config, release_id=release_id), config=config) if bool(gate.get("ok", False)) else {},
        }
    if not bool(gate.get("should_execute", False)) and not shadow_run:
        release_doc = _load_release(config=config, release_id=release_id) if bool(gate.get("ok", False)) else {}
        return {
            "ok": bool(gate.get("ok", False)),
            "status": "skipped",
            "gate": gate,
            "safety": safety,
            "execution_namespace": namespace,
            "shadow_run": shadow_run,
            "market_state": _market_state_runtime(release_doc=release_doc, config=config) if release_doc else {},
        }
    if not bool(safety.get("allow_execution", False)) and not shadow_run:
        release_doc = _load_release(config=config, release_id=release_id) if bool(gate.get("ok", False)) else {}
        return {
            "ok": False,
            "status": "safety_blocked",
            "gate": gate,
            "safety": safety,
            "execution_namespace": namespace,
            "shadow_run": shadow_run,
            "market_state": _market_state_runtime(release_doc=release_doc, config=config) if release_doc else {},
        }

    release_doc = _load_release(config=config, release_id=release_id)
    market_state = _market_state_runtime(release_doc=release_doc, config=config)
    release_context = {
        "release_id": str(release_doc.get("release_id", "") or ""),
        "trade_date": str(release_doc.get("trade_date", "") or ""),
        "profile": str(release_doc.get("profile", "") or ""),
        "source_mode": str(release_doc.get("source_mode", "") or ""),
        "manifest_path": str(release_doc.get("artifacts", {}).get("manifest_path", "") or ""),
        "trigger_label": str(trigger_label or "manual"),
        "trigger_source": str(trigger_source or "manual"),
        "system_mode": str(safety.get("system_mode", "") or ""),
        "market_safety_regime": str(safety.get("market_safety_regime", "") or ""),
        "market_regime": str(market_state.get("market_regime", "") or ""),
        "style_bias": str(market_state.get("style_bias", "") or ""),
        "mechanism_bias": str(market_state.get("mechanism_bias", "") or ""),
        "new_position_policy": str(market_state.get("new_position_policy", "") or ""),
        "effective_reduce_only": bool(safety.get("effective_reduce_only", False)),
        "effective_turnover_multiplier": float(safety.get("effective_turnover_multiplier", 1.0) or 1.0),
        "execution_namespace": namespace,
        "shadow_run": shadow_run,
        "shadow_reason": "shadow_run_bypass" if shadow_run and (not bool(gate.get("should_execute", False)) or not bool(safety.get("allow_execution", False))) else "",
    }
    execution_config = apply_execution_safety_overrides(config=config, safety_report=safety)
    execution_config = _apply_market_state_execution_overrides(config=execution_config, market_state=market_state)
    try:
        report = run_execution_bridge(
            config=execution_config,
            project_root=project_root,
            explicit_portfolio_path=str(release_doc.get("artifacts", {}).get("target_positions_path", "") or ""),
            release_context=release_context,
        )
    except Exception as exc:
        state = load_system_safety_state(config)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        state["system_mode"] = "HALT"
        state["halt_reason"] = "execution_bridge_error"
        state["last_incident_level"] = "error"
        state["last_incident_type"] = "execution_bridge_error"
        save_system_safety_state(config=config, state=state)
        record_incident(
            config=config,
            incident_type="execution_bridge_error",
            severity="error",
            component="execution_manager",
            reason=str(exc),
            action_taken="execution_stopped",
            requires_human_action=True,
            before_system_mode=str(safety.get("system_mode", "") or ""),
            after_system_mode="HALT",
            before_market_regime=str(safety.get("market_safety_regime", "") or ""),
            after_market_regime=str(safety.get("market_safety_regime", "") or ""),
            context_snapshot_ref=str(_trade_clock_root(config) / "system_safety_state.json"),
        )
        return {
            "ok": False,
            "status": "execution_error",
            "gate": gate,
            "safety": safety,
            "execution_namespace": namespace,
            "shadow_run": shadow_run,
            "error": str(exc),
        }
    dispatch_root = ensure_dir(_trade_clock_root(config) / "dispatches" / namespace / datetime.now().strftime("%Y%m%d_%H%M%S"))
    dispatch_path = dispatch_root / "execution_dispatch.json"
    dispatch_doc = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trigger_label": str(trigger_label or "manual"),
        "trigger_source": str(trigger_source or "manual"),
        "gate": gate,
        "safety": safety,
        "release": release_context,
        "execution_report": report,
    }
    dispatch_path.write_text(json.dumps(dispatch_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = _trade_clock_root(config) / "latest_execution_dispatch.json"
    if namespace != "main":
        latest_path = _trade_clock_root(config) / f"latest_execution_dispatch.{namespace}.json"
    latest_path.write_text(json.dumps(dispatch_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    history_paths = record_release_execution(
        release_doc=release_doc,
        execution_record={
            "timestamp": dispatch_doc["timestamp"],
            "trigger_label": str(trigger_label or "manual"),
            "trigger_source": str(trigger_source or "manual"),
            "execution_report_path": str(report.get("execution_report_path", "") or ""),
            "dispatch_path": str(dispatch_path),
            "n_orders": int(report.get("n_orders", 0) or 0),
            "n_fills": int(report.get("n_fills", 0) or 0),
        },
    )
    return {
        "ok": True,
        "status": "executed",
        "gate": gate,
        "safety": safety,
        "execution_namespace": namespace,
        "shadow_run": shadow_run,
        "market_state": market_state,
        "release": release_context,
        "dispatch_path": str(dispatch_path),
        "latest_dispatch_path": str(latest_path),
        "release_execution_paths": history_paths,
        "execution_report": report,
    }
