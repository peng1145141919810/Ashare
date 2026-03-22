from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from .config_utils import ensure_dir, load_config
from .execution_manager import assess_execution_gate
from .portfolio_release import load_latest_release, load_release_by_id
from .safety_guard import assess_system_safety
from .trading_clock import clock_now, is_trading_day, next_trading_day, trading_clock_snapshot

PHASE_SEQUENCE = ("research", "release", "preopen_gate", "simulation", "shadow", "summary")
FINAL_PHASE_STATUSES = {"success", "failed", "skipped", "timeout"}
RESULT_START = "===== ASHARE RESULT JSON START ====="
RESULT_END = "===== ASHARE RESULT JSON END ====="


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    lock_name: str
    scheduled_time: str
    timeout_minutes: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _trade_clock_root(config: Dict[str, Any]) -> Path:
    return ensure_dir(Path(str(config.get("paths", {}).get("trade_clock_root", "") or "")).resolve())


def _automation_runs_root(config: Dict[str, Any]) -> Path:
    return ensure_dir(Path(str(config.get("paths", {}).get("automation_runs_root", "") or "")).resolve())


def _phase_state_root(config: Dict[str, Any]) -> Path:
    return ensure_dir(_trade_clock_root(config) / "phase_state")


def _locks_root(config: Dict[str, Any]) -> Path:
    return ensure_dir(_trade_clock_root(config) / "locks")


def _runtime_root(config: Dict[str, Any]) -> Path:
    return ensure_dir(_trade_clock_root(config) / "runtime")


def _clock_state_path(config: Dict[str, Any]) -> Path:
    return _trade_clock_root(config) / "clock_state.json"


def _scheduler_runtime_state_path(config: Dict[str, Any]) -> Path:
    return _runtime_root(config) / "scheduler_runtime.json"


def _stop_request_path(config: Dict[str, Any]) -> Path:
    return _runtime_root(config) / "stop_request.json"


def _phase_runtime_dir(config: Dict[str, Any], trade_date: str) -> Path:
    return ensure_dir(_runtime_root(config) / trade_date.replace("-", ""))


def _runtime_log_paths(config: Dict[str, Any], trade_date: str, phase_name: str) -> Dict[str, Path]:
    root = _phase_runtime_dir(config, trade_date)
    return {
        "root": root,
        "stdout": root / f"{phase_name}.stdout.log",
        "stderr": root / f"{phase_name}.stderr.log",
    }


def _phase_state_path(config: Dict[str, Any], trade_date: str) -> Path:
    return _phase_state_root(config) / f"{trade_date.replace('-', '')}.json"


def _lock_path(config: Dict[str, Any], lock_name: str) -> Path:
    return _locks_root(config) / f"{lock_name}.lock.json"


def _load_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    fallback = dict(default or {})
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def _scheduler_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(dict(config.get("trade_clock", {}) or {}).get("scheduler", {}) or {})


def _phase_specs(config: Dict[str, Any]) -> Dict[str, PhaseSpec]:
    scheduler = _scheduler_cfg(config)
    return {
        "research": PhaseSpec("research", "research", str(scheduler.get("research_time", "15:05:00") or "15:05:00"), int(scheduler.get("research_timeout_minutes", 420) or 420)),
        "release": PhaseSpec("release", "release", str(scheduler.get("release_time", "15:10:00") or "15:10:00"), int(scheduler.get("release_timeout_minutes", 30) or 30)),
        "preopen_gate": PhaseSpec("preopen_gate", "execution", str(scheduler.get("preopen_gate_time", "09:20:00") or "09:20:00"), int(scheduler.get("preopen_gate_timeout_minutes", 15) or 15)),
        "simulation": PhaseSpec("simulation", "simulation", str(scheduler.get("simulation_time", "09:30:35") or "09:30:35"), int(scheduler.get("simulation_timeout_minutes", 45) or 45)),
        "shadow": PhaseSpec("shadow", "shadow", str(scheduler.get("shadow_time", "09:35:00") or "09:35:00"), int(scheduler.get("shadow_timeout_minutes", 30) or 30)),
        "summary": PhaseSpec("summary", "summary", str(scheduler.get("summary_time", "15:20:00") or "15:20:00"), int(scheduler.get("summary_timeout_minutes", 20) or 20)),
    }


def _empty_phase_state() -> Dict[str, Any]:
    return {
        "status": "queued",
        "scheduled_for": "",
        "started_at": "",
        "finished_at": "",
        "return_code": None,
        "release_id": "",
        "warning_count": 0,
        "error_message": "",
        "stdout_log": "",
        "stderr_log": "",
        "stdout_tail": [],
        "stderr_tail": [],
        "result_status": "",
        "result_payload": {},
    }


def _ensure_cycle_state(config: Dict[str, Any], trade_date: str, profile: str) -> Dict[str, Any]:
    path = _phase_state_path(config, trade_date)
    state = _load_json(path, default={})
    phases = dict(state.get("phases", {}) or {})
    for phase_name in PHASE_SEQUENCE:
        bucket = dict(phases.get(phase_name, {}) or {})
        default_bucket = _empty_phase_state()
        default_bucket.update(bucket)
        phases[phase_name] = default_bucket
    state.update(
        {
            "date": str(trade_date),
            "scheduler_profile": str(profile or ""),
            "updated_at": clock_now().isoformat(timespec="seconds"),
            "release_id": str(state.get("release_id", "") or ""),
            "fallback": dict(state.get("fallback", {}) or {}),
            "phases": phases,
        }
    )
    if "created_at" not in state:
        state["created_at"] = state["updated_at"]
    _write_json(path, state)
    return state


def _save_cycle_state(config: Dict[str, Any], state: Dict[str, Any]) -> Path:
    state["updated_at"] = clock_now().isoformat(timespec="seconds")
    return _write_json(_phase_state_path(config, str(state.get("date", "") or "")), state)


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if process:
                ctypes.windll.kernel32.CloseHandle(process)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_lock(config: Dict[str, Any], lock_name: str, trade_date: str, phase_name: str) -> Dict[str, Any] | None:
    path = _lock_path(config, lock_name)
    current = _load_json(path, default={})
    owner_pid = int(current.get("owner_pid", 0) or 0)
    child_pid = int(current.get("child_pid", 0) or 0)
    if current and (_process_alive(child_pid) or _process_alive(owner_pid)):
        return None
    if path.exists():
        path.unlink(missing_ok=True)
    payload = {
        "phase_name": str(phase_name),
        "trade_date": str(trade_date),
        "owner_pid": os.getpid(),
        "child_pid": 0,
        "acquired_at": clock_now().isoformat(timespec="seconds"),
    }
    _write_json(path, payload)
    return payload


def _update_lock_child_pid(config: Dict[str, Any], lock_name: str, child_pid: int) -> None:
    path = _lock_path(config, lock_name)
    current = _load_json(path, default={})
    if not current:
        return
    current["child_pid"] = int(child_pid or 0)
    current["updated_at"] = clock_now().isoformat(timespec="seconds")
    _write_json(path, current)


def _release_lock(config: Dict[str, Any], lock_name: str) -> None:
    path = _lock_path(config, lock_name)
    if not path.exists():
        return
    current = _load_json(path, default={})
    owner_pid = int(current.get("owner_pid", 0) or 0)
    if owner_pid and owner_pid != os.getpid():
        return
    path.unlink(missing_ok=True)


def _clear_owned_locks(config: Dict[str, Any]) -> None:
    for lock_file in _locks_root(config).glob("*.lock.json"):
        current = _load_json(lock_file, default={})
        if int(current.get("owner_pid", 0) or 0) != os.getpid():
            continue
        child_pid = int(current.get("child_pid", 0) or 0)
        if _process_alive(child_pid):
            continue
        lock_file.unlink(missing_ok=True)


def _parse_time(value: str) -> datetime.time:
    return datetime.strptime(str(value or "00:00:00"), "%H:%M:%S").time()


def _scheduled_wallclock(now: datetime, hms: str) -> datetime:
    return datetime.combine(now.date(), _parse_time(hms), now.tzinfo)


def _next_trade_date(config: Dict[str, Any], base_date: date) -> str:
    next_info = next_trading_day(config=config, base_date=base_date, include_today=False)
    return str(next_info.get("next_trading_day", "") or "")


def _current_trade_date(config: Dict[str, Any], now: datetime) -> str:
    trading_day = is_trading_day(config=config, target_date=now.date())
    if bool(trading_day.get("ok", False)) and bool(trading_day.get("is_trading_day", False)):
        return now.date().isoformat()
    return ""


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    return lines[-max(int(limit or 0), 0):]


def _count_warning_lines(lines: Iterable[str]) -> int:
    count = 0
    for line in lines:
        lowered = str(line or "").lower()
        if "warning" in lowered or "warn" in lowered or "风险" in lowered:
            count += 1
    return count


def _extract_result_json(stdout_text: str) -> Dict[str, Any]:
    if RESULT_START not in stdout_text or RESULT_END not in stdout_text:
        return {}
    raw = stdout_text.split(RESULT_START, 1)[1].split(RESULT_END, 1)[0].strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _subprocess_phase(
    config: Dict[str, Any],
    trade_date: str,
    phase_name: str,
    command: list[str],
    timeout_minutes: int,
) -> Dict[str, Any]:
    logs = _runtime_log_paths(config, trade_date, phase_name)
    lock = _acquire_lock(config, _phase_specs(config)[phase_name].lock_name, trade_date=trade_date, phase_name=phase_name)
    if lock is None:
        return {
            "ok": False,
            "phase_status": "skipped",
            "return_code": None,
            "error_message": "lock_held",
            "stdout_log": str(logs["stdout"]),
            "stderr_log": str(logs["stderr"]),
            "stdout_tail": [],
            "stderr_tail": [],
            "result_payload": {},
            "warning_count": 0,
        }
    timed_out = False
    process: subprocess.Popen[str] | None = None
    try:
        with logs["stdout"].open("w", encoding="utf-8") as stdout_handle, logs["stderr"].open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=str(_repo_root()),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            _update_lock_child_pid(config, _phase_specs(config)[phase_name].lock_name, process.pid)
            deadline = time.time() + max(int(timeout_minutes or 0), 1) * 60
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                if time.time() >= deadline:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    break
                time.sleep(5)
        stdout_tail = _tail_lines(logs["stdout"], int(_scheduler_cfg(config).get("log_tail_lines", 30) or 30))
        stderr_tail = _tail_lines(logs["stderr"], int(_scheduler_cfg(config).get("log_tail_lines", 30) or 30))
        stdout_text = logs["stdout"].read_text(encoding="utf-8", errors="ignore") if logs["stdout"].exists() else ""
        result_payload = _extract_result_json(stdout_text)
        return_code = process.returncode if process is not None else None
        error_message = ""
        if timed_out:
            error_message = f"timeout_after_{int(timeout_minutes or 0)}m"
        elif return_code not in (0, None):
            error_message = "\n".join(stderr_tail[-3:]).strip() or f"child_exit_{return_code}"
        elif isinstance(result_payload, dict) and str(result_payload.get("error", "") or "").strip():
            error_message = str(result_payload.get("error", "") or "").strip()
        return {
            "ok": (not timed_out) and return_code == 0,
            "timed_out": timed_out,
            "return_code": return_code,
            "error_message": error_message,
            "stdout_log": str(logs["stdout"]),
            "stderr_log": str(logs["stderr"]),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "result_payload": result_payload,
            "warning_count": _count_warning_lines(list(stdout_tail) + list(stderr_tail)),
        }
    finally:
        _release_lock(config, _phase_specs(config)[phase_name].lock_name)

def _latest_release_safe(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return load_latest_release(config)
    except Exception:
        return {}


def _release_by_id_safe(config: Dict[str, Any], release_id: str) -> Dict[str, Any]:
    if not str(release_id or "").strip():
        return {}
    try:
        return load_release_by_id(config, release_id=release_id)
    except Exception:
        return {}


def _portfolio_artifact_freshness_hours(summary_path: Path, target_path: Path) -> float:
    latest_ts = max(summary_path.stat().st_mtime, target_path.stat().st_mtime)
    age_seconds = max(time.time() - latest_ts, 0.0)
    return age_seconds / 3600.0


def _find_fallback_source(config: Dict[str, Any]) -> Dict[str, Any]:
    scheduler = _scheduler_cfg(config)
    max_age_hours = float(scheduler.get("fallback_max_portfolio_age_hours", 96) or 96)
    portfolio_root = Path(str(config.get("paths", {}).get("portfolio_output_root", "") or "")).resolve()
    summary_path = portfolio_root / "portfolio_recommendation.json"
    target_path = portfolio_root / "target_positions.csv"
    if summary_path.exists() and target_path.exists():
        age_hours = _portfolio_artifact_freshness_hours(summary_path, target_path)
        if age_hours <= max_age_hours:
            return {
                "ok": True,
                "kind": "portfolio_artifacts",
                "summary_path": str(summary_path),
                "target_positions_path": str(target_path),
                "age_hours": round(age_hours, 3),
                "fallback_source_release_id": "",
                "fallback_reason": "research_phase_failed_use_recent_portfolio_artifacts",
            }
    latest_release = _latest_release_safe(config)
    artifacts = dict(latest_release.get("artifacts", {}) or {})
    summary_path = Path(str(artifacts.get("portfolio_summary_path", "") or "")).resolve() if str(artifacts.get("portfolio_summary_path", "") or "").strip() else Path()
    target_path = Path(str(artifacts.get("target_positions_path", "") or "")).resolve() if str(artifacts.get("target_positions_path", "") or "").strip() else Path()
    if summary_path.exists() and target_path.exists():
        age_hours = _portfolio_artifact_freshness_hours(summary_path, target_path)
        if age_hours <= max_age_hours:
            return {
                "ok": True,
                "kind": "latest_release_artifacts",
                "summary_path": str(summary_path),
                "target_positions_path": str(target_path),
                "age_hours": round(age_hours, 3),
                "fallback_source_release_id": str(latest_release.get("release_id", "") or ""),
                "fallback_reason": "research_phase_failed_use_latest_release_artifacts",
            }
    return {
        "ok": False,
        "kind": "",
        "summary_path": "",
        "target_positions_path": "",
        "age_hours": None,
        "fallback_source_release_id": str(latest_release.get("release_id", "") or ""),
        "fallback_reason": "no_acceptable_fallback_source",
    }


def _phase_command(
    config: Dict[str, Any],
    phase_name: str,
    profile: str,
    trade_date: str,
    cycle_state: Dict[str, Any],
) -> list[str]:
    scheduler = _scheduler_cfg(config)
    command = [sys.executable, str(_repo_root() / "launch_canonical.py"), "--profile", str(profile or ""), "--skip-preflight"]
    if phase_name == "research":
        return command + ["--mode", "research_only"]
    if phase_name == "release":
        fallback = dict(cycle_state.get("fallback", {}) or {})
        release_source_mode = "release_only"
        release_note = ""
        extra: list[str] = []
        if bool(fallback.get("active", False)):
            release_source_mode = "fallback_release"
            release_note = (
                f"fallback_reason={fallback.get('fallback_reason', '')}; "
                f"fallback_source_release_id={fallback.get('fallback_source_release_id', '')}; "
                f"target_trade_date={trade_date}"
            )
            if str(fallback.get("summary_path", "") or "").strip():
                extra.extend(["--source-summary-path", str(fallback.get("summary_path", "")).strip()])
            if str(fallback.get("target_positions_path", "") or "").strip():
                extra.extend(["--source-target-positions-path", str(fallback.get("target_positions_path", "")).strip()])
        return command + ["--mode", "release_only", "--release-source-mode", release_source_mode, "--release-note", release_note] + extra
    if phase_name == "preopen_gate":
        release_id = str(cycle_state.get("release_id", "") or "")
        extra = ["--mode", "execution_only", "--gate-only", "--ignore-window"]
        if release_id:
            extra.extend(["--release-id", release_id])
        extra.extend(
            [
                "--execution-mode",
                str(scheduler.get("simulation_execution_mode", "precision") or "precision"),
                "--precision-trade",
                "on" if bool(scheduler.get("simulation_precision_trade_enabled", True)) else "off",
                "--execution-namespace",
                str(scheduler.get("simulation_namespace", "simulation") or "simulation"),
            ]
        )
        return command + extra
    if phase_name == "simulation":
        release_id = str(cycle_state.get("release_id", "") or "")
        extra = [
            "--mode",
            "execution_only",
            "--execution-mode",
            str(scheduler.get("simulation_execution_mode", "precision") or "precision"),
            "--precision-trade",
            "on" if bool(scheduler.get("simulation_precision_trade_enabled", True)) else "off",
            "--execution-namespace",
            str(scheduler.get("simulation_namespace", "simulation") or "simulation"),
        ]
        if release_id:
            extra.extend(["--release-id", release_id])
        return command + extra
    if phase_name == "shadow":
        release_id = str(cycle_state.get("release_id", "") or "")
        extra = [
            "--mode",
            "execution_only",
            "--execution-mode",
            str(scheduler.get("shadow_execution_mode", "precision") or "precision"),
            "--precision-trade",
            "on" if bool(scheduler.get("shadow_precision_trade_enabled", True)) else "off",
            "--execution-namespace",
            str(scheduler.get("shadow_namespace", "shadow") or "shadow"),
            "--shadow-run",
        ]
        if release_id:
            extra.extend(["--release-id", release_id])
        return command + extra
    raise ValueError(f"Unsupported phase command: {phase_name}")


def _phase_outcome_from_execution_payload(phase_name: str, payload: Dict[str, Any]) -> str:
    status = str(payload.get("status", "") or "")
    gate = dict(payload.get("gate", {}) or {})
    safety = dict(payload.get("safety", {}) or {})
    if phase_name == "preopen_gate":
        if not bool(gate.get("ok", False)):
            return "failed"
        if not bool(gate.get("calendar_ok", False)):
            return "skipped"
        if not bool(safety.get("allow_execution", False)):
            return "skipped"
        if not bool(gate.get("should_execute", False)):
            return "skipped"
        return "success"
    if status == "executed":
        return "success"
    if status in {"skipped", "safety_blocked"}:
        return "skipped"
    if status == "execution_error":
        return "failed"
    return "failed"


def _normalise_phase_result(
    config: Dict[str, Any],
    phase_name: str,
    trade_date: str,
    raw_result: Dict[str, Any],
) -> Dict[str, Any]:
    result_payload = dict(raw_result.get("result_payload", {}) or {})
    if bool(raw_result.get("timed_out", False)):
        phase_status = "timeout"
    elif phase_name in {"research", "release"}:
        phase_status = "success" if bool(raw_result.get("ok", False)) else "failed"
    elif phase_name in {"preopen_gate", "simulation", "shadow"}:
        phase_status = _phase_outcome_from_execution_payload(phase_name, result_payload)
        if not raw_result.get("ok", False) and phase_status == "success":
            phase_status = "failed"
    else:
        phase_status = "failed"
    release_id = ""
    if phase_name == "release":
        release_id = str(result_payload.get("release_id", "") or "")
    elif phase_name == "research":
        latest_release = _latest_release_safe(config)
        if str(latest_release.get("trade_date", "") or "") == str(trade_date):
            release_id = str(latest_release.get("release_id", "") or "")
    else:
        release_id = str(result_payload.get("release", {}).get("release_id", "") or result_payload.get("gate", {}).get("release", {}).get("release_id", "") or "")
    error_message = str(raw_result.get("error_message", "") or "").strip()
    if not error_message and phase_status == "skipped":
        error_message = str(
            result_payload.get("reason", "")
            or result_payload.get("gate", {}).get("reason", "")
            or result_payload.get("safety", {}).get("halt_reason", "")
            or "phase_skipped"
        ).strip()
    return {
        "status": phase_status,
        "return_code": raw_result.get("return_code"),
        "release_id": release_id,
        "warning_count": int(raw_result.get("warning_count", 0) or 0),
        "error_message": error_message,
        "stdout_log": str(raw_result.get("stdout_log", "") or ""),
        "stderr_log": str(raw_result.get("stderr_log", "") or ""),
        "stdout_tail": list(raw_result.get("stdout_tail", []) or []),
        "stderr_tail": list(raw_result.get("stderr_tail", []) or []),
        "result_status": str(result_payload.get("status", result_payload.get("status_code", "")) or ""),
        "result_payload": result_payload,
    }


def _phase_state_final(entry: Dict[str, Any]) -> bool:
    return str(entry.get("status", "") or "") in FINAL_PHASE_STATUSES


def _sync_runtime_state(
    config: Dict[str, Any],
    profile: str,
    payload: Dict[str, Any],
) -> None:
    current = _load_json(_scheduler_runtime_state_path(config), default={})
    current.update(payload)
    current.setdefault("service_profile", str(profile or ""))
    current["updated_at"] = clock_now().isoformat(timespec="seconds")
    _write_json(_scheduler_runtime_state_path(config), current)

def _latest_release_for_trade_date(config: Dict[str, Any], trade_date: str) -> Dict[str, Any]:
    latest_release = _latest_release_safe(config)
    if str(latest_release.get("trade_date", "") or "") == str(trade_date):
        return latest_release
    return {}


def _daily_release_summary(release_doc: Dict[str, Any]) -> Dict[str, Any]:
    if not release_doc:
        return {"available": False}
    fallback_active = str(release_doc.get("source_mode", "") or "") == "fallback_release" or "fallback_reason=" in str(release_doc.get("note", "") or "")
    return {
        "available": True,
        "release_id": str(release_doc.get("release_id", "") or ""),
        "trade_date": str(release_doc.get("trade_date", "") or ""),
        "profile": str(release_doc.get("profile", "") or ""),
        "source_mode": str(release_doc.get("source_mode", "") or ""),
        "generated_at": str(release_doc.get("generated_at", "") or ""),
        "target_count": int(release_doc.get("target_count", 0) or 0),
        "simulation_ready": bool(release_doc.get("simulation_ready", False)),
        "fallback_active": fallback_active,
        "note": str(release_doc.get("note", "") or ""),
    }


def _portfolio_v2a_summary(release_doc: Dict[str, Any]) -> Dict[str, Any]:
    if not release_doc:
        return {"available": False}
    posture = dict(release_doc.get("portfolio_posture", {}) or {})
    v2a = dict(release_doc.get("portfolio_v2a", {}) or {})
    return {
        "available": True,
        "state_counts": dict(v2a.get("state_counts", {}) or {}),
        "action_counts": dict(v2a.get("action_counts", {}) or {}),
        "new_entry_count": int(v2a.get("new_entry_count", 0) or 0),
        "replacement_count": int(v2a.get("replacement_count", 0) or 0),
        "total_exposure_cap": float(posture.get("total_exposure_cap", 0.0) or 0.0),
        "new_entry_budget": float(posture.get("new_entry_budget", 0.0) or 0.0),
        "rebalance_mode": str(posture.get("rebalance_mode", "") or ""),
        "current_position_count": int(posture.get("current_position_count", 0) or 0),
        "weak_existing_count": int(posture.get("weak_existing_count", 0) or 0),
    }


def _market_state_summary(release_doc: Dict[str, Any]) -> Dict[str, Any]:
    market_state = dict(release_doc.get("market_state", {}) or {})
    if not market_state:
        return {"available": False}
    market_state["available"] = True
    return market_state


def _oms_root_for_namespace(config: Dict[str, Any], namespace: str) -> Path:
    base_root = Path(str(config.get("oms", {}).get("output_root", config.get("paths", {}).get("oms_output_root", "")) or "")).resolve()
    if str(namespace or "").strip() and str(namespace or "").strip() != "main":
        return base_root / str(namespace).strip()
    return base_root


def _oms_summary_for_namespace(config: Dict[str, Any], namespace: str) -> Dict[str, Any]:
    oms_root = _oms_root_for_namespace(config, namespace)
    summary_path = oms_root / "snapshots" / "oms_summary.json"
    summary = _load_json(summary_path, default={})
    if not summary:
        return {"available": False, "namespace": namespace, "oms_summary_path": str(summary_path)}
    summary["available"] = True
    summary["namespace"] = namespace
    summary["oms_summary_path"] = str(summary_path)
    return summary


def _build_daily_pack(
    config: Dict[str, Any],
    trade_date: str,
    profile: str,
    cycle_state: Dict[str, Any],
) -> Dict[str, Any]:
    pack_dir = ensure_dir(_automation_runs_root(config) / trade_date.replace("-", ""))
    logs_dir = ensure_dir(pack_dir / "logs")
    release_doc = _release_by_id_safe(config, str(cycle_state.get("release_id", "") or "")) or _latest_release_for_trade_date(config, trade_date)
    release_summary = _daily_release_summary(release_doc)
    market_summary = _market_state_summary(release_doc)
    v2a_summary = _portfolio_v2a_summary(release_doc)
    scheduler = _scheduler_cfg(config)
    simulation_namespace = str(scheduler.get("simulation_namespace", "simulation") or "simulation")
    shadow_namespace = str(scheduler.get("shadow_namespace", "shadow") or "shadow")
    simulation_oms = _oms_summary_for_namespace(config, simulation_namespace)
    shadow_oms = _oms_summary_for_namespace(config, shadow_namespace)
    safety_state = _load_json(_trade_clock_root(config) / "system_safety_state.json", default={})
    phase_status_path = _phase_state_path(config, trade_date)
    warnings: list[Dict[str, Any]] = []
    critical_flags: list[Dict[str, Any]] = []
    for phase_name in PHASE_SEQUENCE:
        phase_entry = dict(cycle_state.get("phases", {}).get(phase_name, {}) or {})
        status = str(phase_entry.get("status", "") or "")
        if status in {"failed", "timeout"}:
            critical_flags.append({"phase": phase_name, "code": status, "message": str(phase_entry.get("error_message", "") or status)})
        elif status == "skipped":
            warnings.append({"phase": phase_name, "code": "skipped", "message": str(phase_entry.get("error_message", "") or "phase_skipped")})
        for stream_name in ("stdout_log", "stderr_log"):
            raw = str(phase_entry.get(stream_name, "") or "").strip()
            if not raw:
                continue
            src = Path(raw)
            if src.exists():
                suffix = "stdout" if stream_name == "stdout_log" else "stderr"
                shutil.copy2(src, logs_dir / f"{phase_name}.{suffix}.log")
    if not release_summary.get("available", False):
        critical_flags.append({"phase": "release", "code": "missing_release", "message": "No formal release was available for this trade date."})
    if str(safety_state.get("system_mode", "") or "") not in {"", "NORMAL"}:
        warnings.append(
            {
                "phase": "safety",
                "code": str(safety_state.get("system_mode", "") or "UNKNOWN"),
                "message": str(safety_state.get("halt_reason", "") or safety_state.get("market_safety_regime", "") or "safety_not_normal"),
            }
        )
    if int(simulation_oms.get("gap", {}).get("n_gap_symbols", 0) or 0) > 0:
        warnings.append({"phase": "simulation", "code": "oms_gap", "message": f"simulation gap symbols={simulation_oms.get('gap', {}).get('n_gap_symbols', 0)}"})
    if int(shadow_oms.get("gap", {}).get("n_gap_symbols", 0) or 0) > 0:
        warnings.append({"phase": "shadow", "code": "oms_gap", "message": f"shadow gap symbols={shadow_oms.get('gap', {}).get('n_gap_symbols', 0)}"})

    _write_json(pack_dir / "phase_status.json", cycle_state)
    _write_json(pack_dir / "daily_release_summary.json", release_summary)
    _write_json(pack_dir / "market_state_summary.json", market_summary)
    _write_json(pack_dir / "portfolio_v2a_summary.json", v2a_summary)
    _write_json(pack_dir / "oms_summary_simulation.json", simulation_oms)
    _write_json(pack_dir / "oms_summary_shadow.json", shadow_oms)
    _write_json(pack_dir / "warnings.json", {"count": len(warnings), "items": warnings})
    _write_json(pack_dir / "critical_flags.json", {"count": len(critical_flags), "items": critical_flags})
    if phase_status_path.exists():
        shutil.copy2(phase_status_path, pack_dir / "phase_status.source.json")

    phase_overview = {
        phase_name: str(dict(cycle_state.get("phases", {}).get(phase_name, {}) or {}).get("status", "") or "")
        for phase_name in PHASE_SEQUENCE
    }
    target_count = int(release_doc.get("target_count", 0) or release_summary.get("target_count", 0) or 0)
    report_lines = [
        f"日期: {trade_date}",
        f"研究档位: {profile}",
        f"release_id: {release_summary.get('release_id', '') or '无'}",
        f"market_regime: {market_summary.get('market_regime', '') or 'unknown'}",
        f"style_bias: {market_summary.get('style_bias', '') or 'unknown'}",
        f"V2A posture: {v2a_summary.get('rebalance_mode', '') or 'unknown'}",
        f"目标持仓数: {target_count}",
        f"simulation: {phase_overview.get('simulation', '') or 'queued'}",
        f"shadow: {phase_overview.get('shadow', '') or 'queued'}",
        f"OMS gap(simulation): {simulation_oms.get('gap', {}).get('n_gap_symbols', 0) if simulation_oms.get('available', False) else 'n/a'}",
        f"OMS gap(shadow): {shadow_oms.get('gap', {}).get('n_gap_symbols', 0) if shadow_oms.get('available', False) else 'n/a'}",
        "Top warnings:",
    ]
    if warnings:
        for item in warnings[:6]:
            report_lines.append(f"- [{item.get('phase', '')}] {item.get('message', '')}")
    else:
        report_lines.append("- none")
    _write_text(pack_dir / "daily_report.txt", "\n".join(report_lines) + "\n")
    _write_text(pack_dir / "daily_report.md", "# Daily Automation Report\n\n" + "\n".join(report_lines) + "\n")
    manifest = {
        "generated_at": clock_now().isoformat(timespec="seconds"),
        "trade_date": trade_date,
        "scheduler_profile": str(profile or ""),
        "release_id": str(release_summary.get("release_id", "") or ""),
        "pack_dir": str(pack_dir),
        "phase_status_path": str(pack_dir / "phase_status.json"),
        "simulation_namespace": simulation_namespace,
        "shadow_namespace": shadow_namespace,
        "phase_overview": phase_overview,
        "warning_count": len(warnings),
        "critical_count": len(critical_flags),
        "report_path": str(pack_dir / "daily_report.txt"),
    }
    _write_json(pack_dir / "run_manifest.json", manifest)
    return manifest


def _run_summary_phase(
    config: Dict[str, Any],
    trade_date: str,
    profile: str,
    cycle_state: Dict[str, Any],
) -> Dict[str, Any]:
    logs = _runtime_log_paths(config, trade_date, "summary")
    lock = _acquire_lock(config, _phase_specs(config)["summary"].lock_name, trade_date=trade_date, phase_name="summary")
    if lock is None:
        return {
            "status": "skipped",
            "return_code": None,
            "release_id": str(cycle_state.get("release_id", "") or ""),
            "warning_count": 0,
            "error_message": "lock_held",
            "stdout_log": str(logs["stdout"]),
            "stderr_log": str(logs["stderr"]),
            "stdout_tail": [],
            "stderr_tail": [],
            "result_status": "",
            "result_payload": {},
        }
    try:
        manifest = _build_daily_pack(config=config, trade_date=trade_date, profile=profile, cycle_state=cycle_state)
        text = (
            f"summary_success trade_date={trade_date} "
            f"release_id={manifest.get('release_id', '')} "
            f"pack_dir={manifest.get('pack_dir', '')}"
        )
        _write_text(logs["stdout"], text + "\n")
        _write_text(logs["stderr"], "")
        return {
            "status": "success",
            "return_code": 0,
            "release_id": str(manifest.get("release_id", "") or ""),
            "warning_count": 0,
            "error_message": "",
            "stdout_log": str(logs["stdout"]),
            "stderr_log": str(logs["stderr"]),
            "stdout_tail": [text],
            "stderr_tail": [],
            "result_status": "summary_success",
            "result_payload": manifest,
        }
    except Exception as exc:
        _write_text(logs["stdout"], "")
        _write_text(logs["stderr"], f"{exc}\n")
        return {
            "status": "failed",
            "return_code": 1,
            "release_id": str(cycle_state.get("release_id", "") or ""),
            "warning_count": 0,
            "error_message": str(exc),
            "stdout_log": str(logs["stdout"]),
            "stderr_log": str(logs["stderr"]),
            "stdout_tail": [],
            "stderr_tail": [str(exc)],
            "result_status": "summary_failed",
            "result_payload": {},
        }
    finally:
        _release_lock(config, _phase_specs(config)["summary"].lock_name)

def _mark_phase_running(
    config: Dict[str, Any],
    trade_date: str,
    profile: str,
    phase_name: str,
    scheduled_for: str,
) -> Dict[str, Any]:
    state = _ensure_cycle_state(config, trade_date, profile)
    phase_entry = dict(state.get("phases", {}).get(phase_name, {}) or {})
    phase_entry.update(
        {
            "status": "running",
            "scheduled_for": scheduled_for,
            "started_at": clock_now().isoformat(timespec="seconds"),
            "finished_at": "",
            "error_message": "",
        }
    )
    state["current_phase"] = phase_name
    state["phases"][phase_name] = phase_entry
    _save_cycle_state(config, state)
    return state


def _mark_phase_complete(
    config: Dict[str, Any],
    trade_date: str,
    profile: str,
    phase_name: str,
    phase_result: Dict[str, Any],
) -> Dict[str, Any]:
    state = _ensure_cycle_state(config, trade_date, profile)
    phase_entry = dict(state.get("phases", {}).get(phase_name, {}) or {})
    phase_entry.update(phase_result)
    phase_entry["finished_at"] = clock_now().isoformat(timespec="seconds")
    state["phases"][phase_name] = phase_entry
    if phase_name == "release" and str(phase_result.get("release_id", "") or "").strip():
        state["release_id"] = str(phase_result.get("release_id", "") or "").strip()
    if phase_name == "summary" and isinstance(phase_result.get("result_payload"), dict):
        state["summary_pack_dir"] = str(phase_result["result_payload"].get("pack_dir", "") or "")
    state["current_phase"] = ""
    _save_cycle_state(config, state)
    return state


def _candidate_phases(config: Dict[str, Any], profile: str, now: datetime) -> list[Dict[str, Any]]:
    specs = _phase_specs(config)
    current_trade_date = _current_trade_date(config, now)
    next_trade_date = _next_trade_date(config, now.date())
    candidates: list[Dict[str, Any]] = []
    if next_trade_date:
        candidates.extend(
            [
                {"trade_date": next_trade_date, "phase_name": "research", "scheduled_at": _scheduled_wallclock(now, specs["research"].scheduled_time)},
                {"trade_date": next_trade_date, "phase_name": "release", "scheduled_at": _scheduled_wallclock(now, specs["release"].scheduled_time)},
            ]
        )
    if current_trade_date:
        candidates.extend(
            [
                {"trade_date": current_trade_date, "phase_name": "preopen_gate", "scheduled_at": _scheduled_wallclock(now, specs["preopen_gate"].scheduled_time)},
                {"trade_date": current_trade_date, "phase_name": "simulation", "scheduled_at": _scheduled_wallclock(now, specs["simulation"].scheduled_time)},
                {"trade_date": current_trade_date, "phase_name": "shadow", "scheduled_at": _scheduled_wallclock(now, specs["shadow"].scheduled_time)},
                {"trade_date": current_trade_date, "phase_name": "summary", "scheduled_at": _scheduled_wallclock(now, specs["summary"].scheduled_time)},
            ]
        )
    due: list[Dict[str, Any]] = []
    for item in candidates:
        if now < item["scheduled_at"]:
            continue
        state = _ensure_cycle_state(config, item["trade_date"], profile)
        if _phase_state_final(dict(state.get("phases", {}).get(item["phase_name"], {}) or {})):
            continue
        due.append(item)
    return sorted(due, key=lambda row: (row["scheduled_at"], PHASE_SEQUENCE.index(row["phase_name"])))


def _run_phase(
    config: Dict[str, Any],
    profile: str,
    trade_date: str,
    phase_name: str,
    scheduled_at: datetime,
) -> Dict[str, Any]:
    cycle_state = _ensure_cycle_state(config, trade_date, profile)
    specs = _phase_specs(config)
    _mark_phase_running(config, trade_date, profile, phase_name, scheduled_for=scheduled_at.isoformat(timespec="seconds"))
    if phase_name == "release":
        research_status = str(dict(cycle_state.get("phases", {}).get("research", {}) or {}).get("status", "") or "")
        if research_status in {"failed", "timeout"}:
            fallback = _find_fallback_source(config)
            cycle_state["fallback"] = {
                "active": bool(fallback.get("ok", False)),
                **fallback,
            }
            _save_cycle_state(config, cycle_state)
            if not bool(fallback.get("ok", False)):
                result = {
                    "status": "failed",
                    "return_code": None,
                    "release_id": "",
                    "warning_count": 0,
                    "error_message": str(fallback.get("fallback_reason", "") or "fallback_unavailable"),
                    "stdout_log": "",
                    "stderr_log": "",
                    "stdout_tail": [],
                    "stderr_tail": [],
                    "result_status": "fallback_unavailable",
                    "result_payload": fallback,
                }
                return _mark_phase_complete(config, trade_date, profile, phase_name, result)
        else:
            cycle_state["fallback"] = {"active": False}
            _save_cycle_state(config, cycle_state)
    elif phase_name == "simulation":
        if not str(cycle_state.get("release_id", "") or "").strip():
            result = {
                "status": "skipped",
                "return_code": None,
                "release_id": "",
                "warning_count": 0,
                "error_message": "no_formal_release_for_trade_date",
                "stdout_log": "",
                "stderr_log": "",
                "stdout_tail": [],
                "stderr_tail": [],
                "result_status": "no_formal_release",
                "result_payload": {},
            }
            return _mark_phase_complete(config, trade_date, profile, phase_name, result)
    elif phase_name == "shadow":
        if not str(cycle_state.get("release_id", "") or "").strip():
            raw = _subprocess_phase(
                config=config,
                trade_date=trade_date,
                phase_name=phase_name,
                command=[
                    sys.executable,
                    str(_repo_root() / "launch_canonical.py"),
                    "--profile",
                    str(profile or ""),
                    "--skip-preflight",
                    "--mode",
                    "execution_only",
                    "--gate-only",
                    "--execution-mode",
                    str(_scheduler_cfg(config).get("shadow_execution_mode", "precision") or "precision"),
                    "--precision-trade",
                    "on" if bool(_scheduler_cfg(config).get("shadow_precision_trade_enabled", True)) else "off",
                    "--execution-namespace",
                    str(_scheduler_cfg(config).get("shadow_namespace", "shadow") or "shadow"),
                ],
                timeout_minutes=specs[phase_name].timeout_minutes,
            )
            result = _normalise_phase_result(config, phase_name, trade_date, raw)
            return _mark_phase_complete(config, trade_date, profile, phase_name, result)
    if phase_name == "summary":
        result = _run_summary_phase(config=config, trade_date=trade_date, profile=profile, cycle_state=cycle_state)
        return _mark_phase_complete(config, trade_date, profile, phase_name, result)
    raw = _subprocess_phase(
        config=config,
        trade_date=trade_date,
        phase_name=phase_name,
        command=_phase_command(config=config, phase_name=phase_name, profile=profile, trade_date=trade_date, cycle_state=cycle_state),
        timeout_minutes=specs[phase_name].timeout_minutes,
    )
    result = _normalise_phase_result(config, phase_name, trade_date, raw)
    return _mark_phase_complete(config, trade_date, profile, phase_name, result)


def _stop_requested(config: Dict[str, Any]) -> bool:
    return _stop_request_path(config).exists()


def _sleep_with_stop_check(config: Dict[str, Any], total_seconds: int) -> None:
    remaining = max(int(total_seconds or 0), 0)
    while remaining > 0:
        if _stop_requested(config):
            return
        step = 2 if remaining >= 2 else remaining
        time.sleep(step)
        remaining -= step


def _scheduler_heartbeat_state(
    config: Dict[str, Any],
    profile: str,
    now: datetime,
) -> Dict[str, Any]:
    snapshot = trading_clock_snapshot(config=config, now=now)
    gate = assess_execution_gate(config=config, release_id="", ignore_window=False, now=now)
    safety = assess_system_safety(
        config=config,
        gate=gate,
        project_root=Path(__file__).resolve().parent.parent,
        service_name="trade_clock_service",
        current_mode="clock_scheduler",
        force_account_refresh=False,
    )
    due = _candidate_phases(config=config, profile=profile, now=now)
    next_due = due[0] if due else {}
    runtime_state = _load_json(_scheduler_runtime_state_path(config), default={})
    state = {
        "last_heartbeat_at": str(snapshot.get("now", "") or now.isoformat(timespec="seconds")),
        "heartbeat_time": str(snapshot.get("now", "") or now.isoformat(timespec="seconds")),
        "market_stage": str(snapshot.get("market_stage", "") or ""),
        "calendar_ok": bool(snapshot.get("calendar_ok", False)),
        "is_trading_day": bool(snapshot.get("is_trading_day", False)),
        "active_execution_window": snapshot.get("active_execution_window"),
        "gate": gate,
        "service_name": "trade_clock_service",
        "service_alive": True,
        "current_mode": "clock_scheduler",
        "scheduler_enabled": bool(_scheduler_cfg(config).get("enabled", True)),
        "scheduler_profile": str(profile or ""),
        "stop_requested": _stop_requested(config),
        "next_due_phase": str(next_due.get("phase_name", "") or ""),
        "next_due_trade_date": str(next_due.get("trade_date", "") or ""),
        "next_due_at": next_due.get("scheduled_at").isoformat(timespec="seconds") if next_due else "",
        "service_runtime_state_path": str(_scheduler_runtime_state_path(config)),
        "phase_state_root": str(_phase_state_root(config)),
        "locks_root": str(_locks_root(config)),
        "runtime_root": str(_runtime_root(config)),
        "automation_runs_root": str(_automation_runs_root(config)),
        "system_mode": str(safety.get("system_mode", "") or ""),
        "market_safety_regime": str(safety.get("market_safety_regime", "") or ""),
        "manual_halt": bool(safety.get("manual_halt", False)),
        "manual_reduce_only": bool(safety.get("manual_reduce_only", False)),
        "release_age_seconds": safety.get("state", {}).get("release_age_seconds"),
        "account_state_age_seconds": safety.get("state", {}).get("account_state_age_seconds"),
        "position_sync_age_seconds": safety.get("state", {}).get("position_sync_age_seconds"),
        "system_state_path": str(_trade_clock_root(config) / "system_safety_state.json"),
        "incident_log_path": str(_trade_clock_root(config) / "incident_log.jsonl"),
        "manual_overrides_path": str(_trade_clock_root(config) / "manual_overrides.json"),
        "runtime": runtime_state,
    }
    return state


def run_trade_clock(
    config_path: Path,
    profile: str,
    poll_seconds: int | None = None,
    once: bool = False,
) -> Dict[str, Any]:
    config = load_config(config_path)
    scheduler = _scheduler_cfg(config)
    sleep_seconds = int(poll_seconds if poll_seconds is not None else dict(config.get("trade_clock", {}) or {}).get("poll_seconds", 30) or 30)
    _sync_runtime_state(
        config=config,
        profile=profile,
        payload={
            "service_status": "starting",
            "config_path": str(config_path),
            "service_profile": str(profile or ""),
            "poll_seconds": int(sleep_seconds),
            "pid": os.getpid(),
        },
    )
    while True:
        config = load_config(config_path)
        now = clock_now(str(dict(config.get("trade_clock", {}) or {}).get("timezone", "Asia/Shanghai") or "Asia/Shanghai"))
        _clear_owned_locks(config)
        heartbeat = _scheduler_heartbeat_state(config=config, profile=profile, now=now)
        _write_json(_clock_state_path(config), heartbeat)
        if _stop_requested(config):
            _sync_runtime_state(
                config=config,
                profile=profile,
                payload={
                    "service_status": "stopping",
                    "stop_reason": "manual_stop_request",
                    "active_phase": "",
                },
            )
            _clear_owned_locks(config)
            return heartbeat
        executed_phase = False
        if bool(scheduler.get("enabled", True)):
            due = _candidate_phases(config=config, profile=profile, now=now)
            if due:
                candidate = due[0]
                _sync_runtime_state(
                    config=config,
                    profile=profile,
                    payload={
                        "service_status": "running_phase",
                        "active_phase": str(candidate["phase_name"]),
                        "active_trade_date": str(candidate["trade_date"]),
                    },
                )
                state = _run_phase(
                    config=config,
                    profile=profile,
                    trade_date=str(candidate["trade_date"]),
                    phase_name=str(candidate["phase_name"]),
                    scheduled_at=candidate["scheduled_at"],
                )
                phase_bucket = dict(state.get("phases", {}).get(str(candidate["phase_name"]), {}) or {})
                _sync_runtime_state(
                    config=config,
                    profile=profile,
                    payload={
                        "service_status": "idle",
                        "active_phase": "",
                        "active_trade_date": "",
                        "last_phase": str(candidate["phase_name"]),
                        "last_trade_date": str(candidate["trade_date"]),
                        "last_phase_status": str(phase_bucket.get("status", "") or ""),
                        "last_release_id": str(state.get("release_id", "") or ""),
                    },
                )
                executed_phase = True
        print(
            json.dumps(
                {
                    "heartbeat": heartbeat.get("last_heartbeat_at", ""),
                    "market_stage": heartbeat.get("market_stage", ""),
                    "next_due_phase": heartbeat.get("next_due_phase", ""),
                    "next_due_trade_date": heartbeat.get("next_due_trade_date", ""),
                    "system_mode": heartbeat.get("system_mode", ""),
                    "market_regime": heartbeat.get("market_safety_regime", ""),
                    "stop_requested": heartbeat.get("stop_requested", False),
                    "phase_executed": executed_phase,
                },
                ensure_ascii=False,
            )
        )
        if once:
            return heartbeat
        _sleep_with_stop_check(config, max(int(sleep_seconds or 0), 10))
