from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

import pandas as pd

from .config_utils import ensure_dir
from .execution_bridge_runner import execution_policy
from .trading_clock import clock_now, current_execution_window, is_trading_day, load_execution_windows, next_trading_day

RELEASE_SCHEMA_VERSION = 1


def _paths(config: Dict[str, Any]) -> Dict[str, Path]:
    paths = dict(config.get("paths", {}) or {})
    root = ensure_dir(Path(str(paths.get("trade_release_root", "") or "")).resolve())
    latest_root = ensure_dir(root / "latest")
    releases_root = ensure_dir(root / "releases")
    history_path = root / "release_history.jsonl"
    latest_pointer = root / "latest_release.json"
    return {
        "root": root,
        "latest_root": latest_root,
        "releases_root": releases_root,
        "history_path": history_path,
        "latest_pointer": latest_pointer,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_paths(config: Dict[str, Any], summary_path: str = "", target_positions_path: str = "") -> Tuple[Path, Path, Path | None]:
    portfolio_root = Path(str(config.get("paths", {}).get("portfolio_output_root", "") or "")).resolve()
    summary = Path(summary_path).resolve() if str(summary_path).strip() else portfolio_root / "portfolio_recommendation.json"
    target = Path(target_positions_path).resolve() if str(target_positions_path).strip() else portfolio_root / "target_positions.csv"
    rebalance = portfolio_root / "rebalance_orders.csv"
    return summary, target, rebalance if rebalance.exists() else None


def _resolve_trade_date(config: Dict[str, Any], now: datetime) -> Dict[str, Any]:
    trade_day_today = is_trading_day(config=config, target_date=now.date())
    earliest_window = load_execution_windows(config)[0] if load_execution_windows(config) else None
    allow_same_day = bool(trade_day_today.get("ok", False) and trade_day_today.get("is_trading_day", False))
    if earliest_window is not None and now.timetz().replace(tzinfo=None) >= earliest_window.start:
        allow_same_day = False
    if allow_same_day:
        trade_date = now.date().isoformat()
        return {
            "ok": True,
            "trade_date": trade_date,
            "selection_reason": "same_day_before_execution_window",
        }
    next_day = next_trading_day(config=config, base_date=now.date(), include_today=False)
    return {
        "ok": bool(next_day.get("ok", False)),
        "trade_date": str(next_day.get("next_trading_day", "") or ""),
        "selection_reason": "next_trading_day",
        "calendar_path": str(next_day.get("calendar_path", "") or ""),
    }


def _release_constraints(config: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    portfolio_cfg = dict(config.get("portfolio_recommendation", {}) or {})
    control_cfg = dict(config.get("portfolio_control", {}) or {})
    broker_cfg = dict(config.get("broker", {}) or {})
    if not broker_cfg:
        bridge_cfg = dict(config.get("execution_bridge_runtime", {}) or {})
        broker_cfg = dict(bridge_cfg.get("broker", {}) or {})
    portfolio_limits = dict(summary.get("portfolio_limits", {}) or {})
    return {
        "single_name_cap": float(portfolio_limits.get("single_name_cap", portfolio_cfg.get("single_name_cap", 0.10)) or 0.10),
        "total_exposure_cap": float(portfolio_limits.get("total_exposure_cap", portfolio_cfg.get("total_exposure_cap", 1.0)) or 1.0),
        "max_names": int(portfolio_limits.get("max_names", portfolio_cfg.get("max_names", 20)) or 20),
        "max_daily_turnover_ratio": float(control_cfg.get("max_daily_turnover_ratio", 0.25) or 0.25),
        "drift_threshold": float(control_cfg.get("drift_threshold", 0.005) or 0.005),
        "cash_reserve_ratio": float(broker_cfg.get("cash_reserve_ratio", 0.02) or 0.02),
    }


def publish_portfolio_release(
    config: Dict[str, Any],
    source_mode: str,
    profile: str,
    summary_path: str = "",
    target_positions_path: str = "",
    note: str = "",
) -> Dict[str, Any]:
    now = clock_now(str(config.get("trade_clock", {}).get("timezone", "Asia/Shanghai") or "Asia/Shanghai"))
    src_summary, src_target, src_rebalance = _source_paths(config=config, summary_path=summary_path, target_positions_path=target_positions_path)
    if not src_summary.exists():
        raise FileNotFoundError(f"未找到持仓建议摘要文件: {src_summary}")
    if not src_target.exists():
        raise FileNotFoundError(f"未找到目标持仓文件: {src_target}")

    target_df = pd.read_csv(src_target)
    if target_df.empty:
        raise ValueError(f"目标持仓为空，拒绝发布 release: {src_target}")
    summary = _load_json(src_summary)

    trade_date_info = _resolve_trade_date(config=config, now=now)
    if not bool(trade_date_info.get("ok", False)):
        raise RuntimeError("交易日历不可用，无法安全发布 portfolio release。")
    trade_date = str(trade_date_info.get("trade_date", "") or "")
    release_cfg = dict(config.get("trade_release", {}) or {})
    window = current_execution_window(config=config, now=now)
    execution_windows = load_execution_windows(config)
    valid_after_text = str(release_cfg.get("valid_after_time", "") or (execution_windows[0].start.strftime("%H:%M:%S") if execution_windows else "09:30:30"))
    expires_at_text = str(release_cfg.get("expires_at_time", "15:00:00") or "15:00:00")

    release_id = f"release_{now.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    path_map = _paths(config)
    release_dir = ensure_dir(path_map["releases_root"] / release_id)
    target_copy = release_dir / "target_positions.csv"
    summary_copy = release_dir / "portfolio_recommendation.json"
    rebalance_copy = release_dir / "rebalance_orders.csv"
    manifest_path = release_dir / "release_manifest.json"

    _copy_if_exists(src_target, target_copy)
    _copy_if_exists(src_summary, summary_copy)
    if src_rebalance is not None:
        _copy_if_exists(src_rebalance, rebalance_copy)

    release_doc = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": release_id,
        "status": "published",
        "generated_at": now.isoformat(timespec="seconds"),
        "source_mode": str(source_mode or "research_only"),
        "profile": str(profile or ""),
        "trade_date": trade_date,
        "valid_after": f"{trade_date}T{valid_after_text}+08:00",
        "expires_at": f"{trade_date}T{expires_at_text}+08:00",
        "selection_reason": str(trade_date_info.get("selection_reason", "") or ""),
        "note": str(note or ""),
        "target_count": int(len(target_df.index)),
        "strategy_name": str(summary.get("strategy_name", "") or ""),
        "strategy_key": str(summary.get("strategy_key", "") or ""),
        "run_id": str(summary.get("run_id", "") or ""),
        "simulation_ready": bool(summary.get("simulation_ready", True)),
        "execution_policy": execution_policy(config),
        "constraints": _release_constraints(config=config, summary=summary),
        "artifacts": {
            "manifest_path": str(manifest_path),
            "release_dir": str(release_dir),
            "target_positions_path": str(target_copy),
            "portfolio_summary_path": str(summary_copy),
            "rebalance_orders_path": str(rebalance_copy) if rebalance_copy.exists() else "",
            "source_target_positions_path": str(src_target),
            "source_portfolio_summary_path": str(src_summary),
        },
        "checksums": {
            "target_positions_sha256": _sha256_of_file(target_copy),
            "portfolio_summary_sha256": _sha256_of_file(summary_copy),
        },
        "research_snapshot": {
            "generated_at": str(summary.get("generated_at", "") or ""),
            "total_score": float(summary.get("total_score", 0.0) or 0.0),
            "sharpe": float(summary.get("sharpe", 0.0) or 0.0),
            "n_names": int(summary.get("n_names", len(target_df.index)) or len(target_df.index)),
            "active_window_at_publish": window.label if window else "",
        },
    }
    manifest_path.write_text(json.dumps(release_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_root = path_map["latest_root"]
    _copy_if_exists(manifest_path, latest_root / "release_manifest.json")
    _copy_if_exists(target_copy, latest_root / "target_positions.csv")
    _copy_if_exists(summary_copy, latest_root / "portfolio_recommendation.json")
    if rebalance_copy.exists():
        _copy_if_exists(rebalance_copy, latest_root / "rebalance_orders.csv")

    pointer_doc = {
        "release_id": release_id,
        "trade_date": trade_date,
        "status": "published",
        "generated_at": release_doc["generated_at"],
        "manifest_path": str(manifest_path),
        "target_positions_path": str(target_copy),
        "profile": str(profile or ""),
        "source_mode": str(source_mode or "research_only"),
    }
    path_map["latest_pointer"].write_text(json.dumps(pointer_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    with path_map["history_path"].open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(pointer_doc, ensure_ascii=False) + "\n")
    return release_doc


def load_latest_release(config: Dict[str, Any]) -> Dict[str, Any]:
    pointer_path = _paths(config)["latest_pointer"]
    if not pointer_path.exists():
        raise FileNotFoundError(f"未找到 latest release 指针: {pointer_path}")
    pointer = _load_json(pointer_path)
    manifest_path = Path(str(pointer.get("manifest_path", "") or "")).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"latest release manifest 不存在: {manifest_path}")
    return _load_json(manifest_path)


def load_release_by_id(config: Dict[str, Any], release_id: str) -> Dict[str, Any]:
    manifest_path = _paths(config)["releases_root"] / str(release_id).strip() / "release_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到指定 release: {manifest_path}")
    return _load_json(manifest_path)


def record_release_execution(release_doc: Dict[str, Any], execution_record: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = Path(str(release_doc.get("artifacts", {}).get("manifest_path", "") or "")).resolve()
    release_dir = manifest_path.parent
    history_path = release_dir / "execution_history.jsonl"
    latest_path = release_dir / "latest_execution.json"
    latest_path.write_text(json.dumps(execution_record, ensure_ascii=False, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(execution_record, ensure_ascii=False) + "\n")
    return {
        "history_path": str(history_path),
        "latest_execution_path": str(latest_path),
    }
