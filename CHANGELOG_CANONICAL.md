# Canonical Governance Changelog

This file tracks governance-layer additions that define the formal operator
path, runtime identity, and run-trace rules.

Detailed engineering handoff and runtime history remain in:
- `F:\quant_data\Ashare\CODEX_DEV_LOG.md`

## Entries

### 2026-03-21 19:29
- Phase: `1`
- Change:
  - Added the first law/manifest/profile/changelog governance files.
- Files:
  - `F:\quant_data\Ashare\PROJECT_LAW.md`
  - `F:\quant_data\Ashare\SYSTEM_MANIFEST.yaml`
  - `F:\quant_data\Ashare\RUN_PROFILES.yaml`
  - `F:\quant_data\Ashare\CHANGELOG_CANONICAL.md`
- Intent:
  - Freeze one live runtime identity and define archive/experiment semantics
    without changing business logic.

### 2026-03-21 19:36
- Phase: `2`
- Change:
  - Added the formal operator wrapper `launch_canonical.py`.
  - Added `tools/preflight_check.py` for lightweight path/profile/import/py_compile checks.
- Files:
  - `F:\quant_data\Ashare\launch_canonical.py`
  - `F:\quant_data\Ashare\tools\preflight_check.py`
  - `F:\quant_data\Ashare\AGENTS.md`
  - `F:\quant_data\Ashare\CHANGELOG_CANONICAL.md`
- Intent:
  - Make one operator-facing launch path real without changing the underlying business call chain.
  - Keep the wrapper thin by dispatching to the canonical research Python instead of assuming the shell Python is the live runtime.

### 2026-03-21 19:43
- Phase: `3`
- Change:
  - Added `tools/register_run.py`.
  - Made `launch_canonical.py` create one governance run directory and `run_manifest.json` for each formal run.
- Files:
  - `F:\quant_data\Ashare\tools\register_run.py`
  - `F:\quant_data\Ashare\launch_canonical.py`
  - `F:\quant_data\Ashare\AGENTS.md`
  - `F:\quant_data\Ashare\CHANGELOG_CANONICAL.md`
- Intent:
  - Ensure formal runs have one lightweight trace artifact without changing the business output chain.

### 2026-03-21 19:49
- Phase: `4`
- Change:
  - Added explicit non-canonical marker files for mixed output, example, and deprecated-entrypoint locations.
  - Corrected the law/manifest classification so `outputs\canonical_runs` is formal while the rest of `outputs\` remains non-canonical by default.
- Files:
  - `F:\quant_data\Ashare\PROJECT_LAW.md`
  - `F:\quant_data\Ashare\SYSTEM_MANIFEST.yaml`
  - `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321\CANONICAL_STATUS.md`
  - `F:\quant_data\Ashare\outputs\CANONICAL_STATUS.md`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\examples\CANONICAL_STATUS.md`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\docs\DEPRECATED_ENTRYPOINTS.md`
  - `F:\quant_data\Ashare\CHANGELOG_CANONICAL.md`
- Intent:
  - Mark non-live paths explicitly without moving files or changing imports.

### 2026-03-22 02:00
- Phase: `5`
- Change:
  - Added precision-style split modes `research_only`, `release_only`, and `execution_only`.
  - Added a portfolio release layer and a lightweight always-on trade clock service.
  - Added trade-clock autostart scripts for Windows logon startup.
- Files:
  - `F:\quant_data\Ashare\launch_canonical.py`
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\trade_clock_service.py`
  - `F:\quant_data\Ashare\RUN_PROFILES.yaml`
  - `F:\quant_data\Ashare\SYSTEM_MANIFEST.yaml`
  - `F:\quant_data\Ashare\PROJECT_LAW.md`
  - `F:\quant_data\Ashare\scripts\start_trade_clock.ps1`
  - `F:\quant_data\Ashare\scripts\stop_trade_clock.ps1`
  - `F:\quant_data\Ashare\scripts\install_trade_clock_autostart.ps1`
  - `F:\quant_data\Ashare\scripts\remove_trade_clock_autostart.ps1`
- Intent:
  - Split research production from execution timing without rewriting the live business chain.
  - Keep the release layer and clock supervisor additive so the integrated mode remains available for debug and rollback.

### 2026-03-22 02:15
- Phase: `6`
- Change:
  - Added explicit execution-account mode switching between simulation and precision profiles.
  - Added a precision-trade on/off switch so the clock service can stay online while still blocking real execution.
  - Registered the new precision account profile in the local gmtrade runtime template.
- Files:
  - `F:\quant_data\Ashare\launch_canonical.py`
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\trade_clock_service.py`
  - `F:\quant_data\Ashare\scripts\start_trade_clock.ps1`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_bridge_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_manager.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\gmtrade_runtime_config.local.json`
- Intent:
  - Prevent accidental precision execution while preserving one lightweight always-on clock process.
