from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

VALID_ACCOUNT_MODES = {"simulation", "precision"}


def execution_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(config.get("execution_policy", {}) or {})
    mode = str(raw.get("account_mode", "simulation") or "simulation").strip().lower()
    if mode not in VALID_ACCOUNT_MODES:
        mode = "simulation"
    return {
        "account_mode": mode,
        "precision_trade_enabled": bool(raw.get("precision_trade_enabled", False)),
        "allow_integrated_precision_execution": bool(raw.get("allow_integrated_precision_execution", False)),
    }


def _apply_account_profile(payload: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    broker_cfg = dict(payload.get("broker", {}) or {})
    account_profiles = dict(broker_cfg.get("account_profiles", {}) or {})
    selected = dict(account_profiles.get(str(policy.get("account_mode", "simulation")), {}) or {})
    if selected:
        broker_cfg["account_id"] = str(selected.get("account_id", broker_cfg.get("account_id", "")) or broker_cfg.get("account_id", ""))
        broker_cfg["account_alias"] = str(selected.get("account_alias", broker_cfg.get("account_alias", "")) or broker_cfg.get("account_alias", ""))
    broker_cfg["selected_account_mode"] = str(policy.get("account_mode", "simulation"))
    payload["broker"] = broker_cfg
    payload["execution_policy"] = policy
    return payload


def build_execution_runtime_config(
    config: Dict[str, Any],
    explicit_portfolio_path: str = "",
    release_context: Dict[str, Any] | None = None,
) -> Path:
    exec_cfg = dict(config.get("execution_bridge", {}) or {})
    template_path = Path(str(exec_cfg["config_template_path"]))
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    policy = execution_policy(config)
    portfolio_root = str(config["paths"].get("portfolio_output_root", payload.get("portfolio_root", "")))
    payload["portfolio_root"] = portfolio_root
    payload["explicit_portfolio_path"] = str(Path(explicit_portfolio_path).resolve()) if str(explicit_portfolio_path).strip() else str(Path(portfolio_root) / "target_positions.csv")
    payload["price_snapshot_path"] = str(config.get("market_pipeline", {}).get("price_snapshot_path", payload.get("price_snapshot_path", "")))
    payload["output_dir"] = str(config["paths"].get("live_execution_root", payload.get("output_dir", "")))
    control_cfg = dict(config.get("portfolio_control", {}) or {})
    control_cfg.setdefault("codex_dev_log_path", str(Path(__file__).resolve().parents[3] / "CODEX_DEV_LOG.md"))
    payload["portfolio_control"] = control_cfg
    if release_context:
        payload["release"] = release_context
    payload = _apply_account_profile(payload=payload, policy=policy)
    out_path = Path(str(exec_cfg["autogen_config_path"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def run_execution_bridge(
    config: Dict[str, Any],
    project_root: Path,
    explicit_portfolio_path: str = "",
    release_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    exec_cfg = dict(config.get("execution_bridge", {}) or {})
    runtime_config_path = build_execution_runtime_config(
        config=config,
        explicit_portfolio_path=explicit_portfolio_path,
        release_context=release_context,
    )
    pyexe = str(exec_cfg["python_executable"])
    script = Path(str(exec_cfg["script_path"]))
    env = os.environ.copy()
    proc = subprocess.run(
        [pyexe, str(script), "--config", str(runtime_config_path)],
        cwd=str(project_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = (proc.stdout or "").strip()
    try:
        report = json.loads(stdout) if stdout else {"ok": True, "stdout": ""}
    except Exception:
        report = {"ok": True, "stdout": stdout}
    report.setdefault("runtime_config_path", str(runtime_config_path))
    if release_context:
        report.setdefault("release", release_context)
    return report
