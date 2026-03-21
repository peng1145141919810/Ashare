from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from tools.preflight_check import run_preflight


def _package_root() -> Path:
    return Path(__file__).resolve().parent / "quant_research_hub_v6_repacked_clean" / "quant_research_hub_v6_repacked_clean"


PACKAGE_ROOT = _package_root()
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hub_v6 import local_settings as LS
from hub_v6.config_builder import build_runtime_config
from hub_v6.clock_supervisor import run_trade_clock


def _profile_overrides(profile: str) -> Dict[str, Dict[str, Any]]:
    if profile == "quick_test":
        return {
            "supervisor": {
                "v5_gpu_max_cycles_per_tick": int(LS.QUICK_TEST_V5_GPU_MAX_CYCLES_PER_TICK),
                "token_plan_min_interval_hours": float(LS.QUICK_TEST_TOKEN_PLAN_MIN_INTERVAL_HOURS),
            },
            "event_ingest": {
                "max_pdf_fetch_per_run": int(LS.QUICK_TEST_MAX_PDF_FETCH_PER_RUN),
            },
            "event_extract": {
                "max_events_per_run": int(LS.QUICK_TEST_MAX_EVENTS_PER_RUN),
                "batch_size": int(LS.QUICK_TEST_DEEPSEEK_BATCH_SIZE),
            },
            "research_context_pack": {
                "max_priority_events": int(LS.QUICK_TEST_MAX_PRIORITY_EVENTS),
            },
        }
    return {
        "supervisor": {
            "v5_gpu_max_cycles_per_tick": int(LS.OVERNIGHT_V5_GPU_MAX_CYCLES_PER_TICK),
            "token_plan_min_interval_hours": float(LS.OVERNIGHT_TOKEN_PLAN_MIN_INTERVAL_HOURS),
        },
        "event_ingest": {
            "max_pdf_fetch_per_run": int(LS.OVERNIGHT_MAX_PDF_FETCH_PER_RUN),
        },
        "event_extract": {
            "max_events_per_run": int(LS.OVERNIGHT_MAX_EVENTS_PER_RUN),
            "batch_size": int(LS.OVERNIGHT_DEEPSEEK_BATCH_SIZE),
        },
        "research_context_pack": {
            "max_priority_events": int(LS.OVERNIGHT_MAX_PRIORITY_EVENTS),
        },
    }


def _deep_update(config: Dict[str, Any], overrides: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for section, values in overrides.items():
        bucket = dict(config.get(section, {}) or {})
        bucket.update(values)
        config[section] = bucket
    return config


def _apply_execution_runtime_overrides(config: Dict[str, Any], execution_mode: str = "", precision_trade: str = "default") -> Dict[str, Any]:
    policy = dict(config.get("execution_policy", {}) or {})
    if str(execution_mode).strip():
        policy["account_mode"] = str(execution_mode).strip().lower()
    if str(precision_trade).strip().lower() == "on":
        policy["precision_trade_enabled"] = True
    elif str(precision_trade).strip().lower() == "off":
        policy["precision_trade_enabled"] = False
    config["execution_policy"] = policy
    return config


def _write_runtime_config(profile: str, execution_mode: str = "", precision_trade: str = "default") -> Path:
    config = build_runtime_config()
    config = _deep_update(config, _profile_overrides(profile))
    config = _apply_execution_runtime_overrides(config=config, execution_mode=execution_mode, precision_trade=precision_trade)
    config["runtime_selection"] = {
        "profile": str(profile),
        "default_mode": str(getattr(LS, "RUN_MODE", "integrated_supervisor") or "integrated_supervisor"),
        "execution_mode": str(config.get("execution_policy", {}).get("account_mode", "") or ""),
        "precision_trade_enabled": bool(config.get("execution_policy", {}).get("precision_trade_enabled", False)),
    }
    config_path = PACKAGE_ROOT / "configs" / f"hub_config.v6.runtime.{profile}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight always-on trade clock supervisor")
    parser.add_argument("--profile", default="quick_test", help="Runtime profile used to resolve config")
    parser.add_argument("--config", default="", help="Optional explicit runtime config path")
    parser.add_argument("--poll-seconds", type=int, default=0, help="Optional override for clock poll seconds")
    parser.add_argument("--once", action="store_true", help="Run one heartbeat and exit")
    parser.add_argument("--execution-mode", default="", choices=["", "simulation", "precision"], help="Execution account mode override")
    parser.add_argument("--precision-trade", default="default", choices=["default", "on", "off"], help="Precision-trade switch override")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip lightweight preflight")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    config_path = (
        Path(args.config).resolve()
        if str(args.config).strip()
        else _write_runtime_config(
            str(args.profile).strip(),
            execution_mode=str(args.execution_mode).strip(),
            precision_trade=str(args.precision_trade).strip(),
        )
    )
    if not args.skip_preflight:
        report = run_preflight(repo_root=repo_root, profile=str(args.profile).strip(), mode="execution_only", explicit_config=str(config_path))
        if not bool(report.get("ok", False)):
            raise SystemExit("Trade clock preflight failed.")
    run_trade_clock(
        config_path=config_path,
        profile=str(args.profile).strip(),
        poll_seconds=(int(args.poll_seconds) if int(args.poll_seconds or 0) > 0 else None),
        once=bool(args.once),
    )


if __name__ == "__main__":
    main()
