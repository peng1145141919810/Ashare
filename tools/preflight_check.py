from __future__ import annotations

import argparse
import importlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json_yaml(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_check(checks: List[Dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _compile_file(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)


def _subprocess_import_check(python_executable: Path, runtime_root: Path, module_name: str) -> subprocess.CompletedProcess[str]:
    command = [
        str(python_executable),
        "-c",
        (
            "import importlib, sys; "
            f"sys.path.insert(0, r'{runtime_root}'); "
            f"importlib.import_module('{module_name}')"
        ),
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def run_preflight(repo_root: Path, profile: str, mode: str, explicit_config: str = "") -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    manifest_path = repo_root / "SYSTEM_MANIFEST.yaml"
    profiles_path = repo_root / "RUN_PROFILES.yaml"
    manifest = _load_json_yaml(manifest_path)
    profiles_doc = _load_json_yaml(profiles_path)

    main_path = Path(str(manifest["canonical"]["wrapped_business_root_entry"]))
    runtime_root = Path(str(manifest["canonical"]["live_runtime_root"]))
    formal_output_root = Path(str(manifest["canonical"]["formal_output_root"]))

    _append_check(checks, "manifest_exists", manifest_path.exists(), str(manifest_path))
    _append_check(checks, "profiles_exists", profiles_path.exists(), str(profiles_path))
    _append_check(checks, "main_exists", main_path.exists(), str(main_path))
    _append_check(checks, "runtime_root_exists", runtime_root.exists(), str(runtime_root))
    _append_check(checks, "formal_output_parent_exists", formal_output_root.parent.exists(), str(formal_output_root.parent))

    allowed_profiles = dict(profiles_doc.get("allowed_profiles", {}) or {})
    allowed_modes = list(profiles_doc.get("allowed_modes", []) or [])
    _append_check(checks, "profile_allowed", profile in allowed_profiles, profile)
    _append_check(checks, "mode_allowed", mode in allowed_modes, mode)

    if explicit_config:
        config_path = Path(explicit_config).resolve()
        _append_check(checks, "explicit_config_exists", config_path.exists(), str(config_path))

    compile_targets = [
        repo_root / "launch_canonical.py",
        repo_root / "main_research_runner.py",
        repo_root / "trade_clock_service.py",
        repo_root / "tools" / "preflight_check.py",
        runtime_root / "hub_v6" / "local_settings.py",
        runtime_root / "hub_v6" / "config_builder.py",
        runtime_root / "hub_v6" / "supervisor.py",
        runtime_root / "hub_v6" / "portfolio_release.py",
        runtime_root / "hub_v6" / "trading_clock.py",
        runtime_root / "hub_v6" / "execution_manager.py",
        runtime_root / "hub_v6" / "clock_supervisor.py",
    ]
    for target in compile_targets:
        try:
            _compile_file(target)
            _append_check(checks, f"py_compile:{target.name}", True, str(target))
        except Exception as exc:
            _append_check(checks, f"py_compile:{target.name}", False, str(exc))

    sys.path.insert(0, str(runtime_root))
    import_targets = [
        "hub_v6.local_settings",
        "hub_v6.config_builder",
        "hub_v6.portfolio_release",
        "hub_v6.execution_manager",
    ]
    for target in import_targets:
        try:
            importlib.import_module(target)
            _append_check(checks, f"import:{target}", True, "ok")
        except Exception as exc:
            _append_check(checks, f"import:{target}", False, str(exc))

    try:
        local_settings = importlib.import_module("hub_v6.local_settings")
        research_python = Path(str(getattr(local_settings, "PYTHON_EXECUTABLE", "") or "").strip())
    except Exception as exc:
        research_python = Path("")
        _append_check(checks, "canonical_research_python", False, str(exc))
    else:
        _append_check(checks, "canonical_research_python", research_python.exists(), str(research_python))

    if str(research_python):
        proc = _subprocess_import_check(
            python_executable=research_python,
            runtime_root=runtime_root,
            module_name="hub_v6.supervisor",
        )
        if proc.returncode == 0:
            _append_check(checks, "import:hub_v6.supervisor@canonical_python", True, "ok")
        else:
            detail = (proc.stderr or proc.stdout or "import failed").strip()
            _append_check(checks, "import:hub_v6.supervisor@canonical_python", False, detail)

    ok = all(bool(item["ok"]) for item in checks)
    return {
        "ok": ok,
        "repo_root": str(repo_root),
        "mode": mode,
        "profile": profile,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight governance preflight checks")
    parser.add_argument("--repo-root", default="", help="Repository root; defaults to the parent of this tools directory")
    parser.add_argument("--profile", required=True, help="Profile to validate")
    parser.add_argument("--mode", required=True, help="Mode to validate")
    parser.add_argument("--config", default="", help="Optional explicit runtime config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if str(args.repo_root).strip() else Path(__file__).resolve().parents[1]
    report = run_preflight(
        repo_root=repo_root,
        profile=str(args.profile).strip(),
        mode=str(args.mode).strip(),
        explicit_config=str(args.config).strip(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(report.get("ok", False)) else 1)


if __name__ == "__main__":
    main()
