# Codex Development Log

## Must Read First
- This file is the current handoff document for future Codex sessions in this project.
- If anything here conflicts with older `README` files, this file wins.
- Do not run full end-to-end validation by default. The user has explicitly said the full pipeline can run for hours and freeze Codex.
- Preferred validation is lightweight only: targeted file inspection, small probes, and `python -m py_compile` on touched files.
- Future Codex sessions must maintain this file. If they materially change behavior, configs, entrypoints, runtime assumptions, or safety rules, they must update this log before ending the turn.

## Log Maintenance Rule
- This is a living development log and handoff note.
- Do not treat it as archival-only documentation.
- After any material change, append or revise the relevant section here.
- Minimum content for a new change entry:
  - local timestamp
  - file or module path
  - change summary
  - intended effect
  - any new operator warning

## Engineering Bias
- Current operator preference:
  - future Codex work in this repo should lean slightly more aggressive and more willing to innovate on research-side architecture instead of over-preserving transitional compatibility layers
- Practical meaning:
  - do not keep adding half-unified wrappers, stale compatibility shims, or duplicate truths just to avoid touching internals
  - when a cleaner cross-layer redesign is clearly better and the user is explicitly pushing the system forward, prefer the cleaner unified path
  - avoid “留后路过多” style engineering that makes later integration harder than the current migration
- Guardrails:
  - aggressive does not mean reckless
  - preserve formal operator entry, release contract, safety layer, and gmtrade environment boundaries unless the user explicitly asks to redesign them
  - still do lightweight validation on touched paths and document behavioral truth in this log before ending the turn
  - if a proposed innovation increases research-side ambition but weakens execution-side safety, keep the safety boundary conservative and isolate the innovation upstream

## Latest Stable Snapshot
- Snapshot date: `2026-03-23`
- Formal operator entry: `F:\quant_data\Ashare\launch_canonical.py`
- Formal trade-clock service entry: `F:\quant_data\Ashare\trade_clock_service.py`
- Canonical business root entry: `F:\quant_data\Ashare\main_research_runner.py`
- Default mode: `integrated_supervisor`
- Default profile: `quick_test`
- Formal run-trace root:
  - `F:\quant_data\Ashare\outputs\canonical_runs`
- Root layout note:
  - active runtime code now remains under `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean`
  - the active V5.1 research brain is embedded under `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\v5_gpu_runtime`
  - archived legacy root packages were moved to `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321`
- Git sync behavior:
  - local commits auto-push to GitHub in this workspace through `.githooks/post-commit`
  - use `$env:DISABLE_AUTO_PUSH='1'; git commit -m "..."; Remove-Item Env:DISABLE_AUTO_PUSH` when a local-only commit is needed in PowerShell
- Google Drive dev-log mirror:
  - `CODEX_DEV_LOG.md` can be mirrored into `G:\我的云端硬盘\Ashare_backups\codex_dev_log_mirror`
  - versioned script snapshots now live under `G:\我的云端硬盘\Ashare_backups\script_versions`
  - snapshot catalog now lives at `G:\我的云端硬盘\Ashare_backups\VERSION_CATALOG.md`
  - current naming scheme:
    - `LEGACY-YYYYMMDD-RNNN` for milestone / layout-transition snapshots
    - `SCRIPT-YYYYMMDD-RNNN` for script-only repo snapshots without `data/`, `outputs/`, virtualenvs, caches, or private `local_settings.py`
  - watcher script: `F:\quant_data\Ashare\scripts\sync_codex_dev_log_to_gdrive.py`
  - start script: `F:\quant_data\Ashare\scripts\start_codex_dev_log_sync.ps1`
  - stop script: `F:\quant_data\Ashare\scripts\stop_codex_dev_log_sync.ps1`
  - autostart install script: `F:\quant_data\Ashare\scripts\install_codex_dev_log_sync_autostart.ps1`
  - autostart remove script: `F:\quant_data\Ashare\scripts\remove_codex_dev_log_sync_autostart.ps1`
  - scheduled task name: `Ashare Codex Dev Log Mirror`
  - startup behavior: logon trigger with 2-minute delay, hidden PowerShell launcher, low-priority watcher, 5-second poll interval
- Canonical research Python:
  - `C:\Users\Administrator\PyCharmMiscProject\.venv\Scripts\python.exe`
- Canonical Gmtrade Python:
  - `F:\quant_data\Ashare\venvs\gmtrade39\Scripts\python.exe`
- Current execution default:
  - account mode: `precision`
  - precision trade switch: `False`
  - operator implication: precision remains the default execution account binding, but the daily scheduler is now a manual-start process and does not auto-run after reboot
- Precision-style split:
  - `research_only` runs the research chain and publishes a portfolio release without directly calling the execution bridge.
  - `release_only` republishes the latest `portfolio_recommendation.json` and `target_positions.csv` into the formal release layer.
  - `execution_only` reads the latest published release, applies trading-clock gates, and only then calls the gmtrade bridge.
  - `trade_clock_service.py` is now the lightweight manual-start daily scheduler that watches phase timepoints, release state, and safety truth, then dispatches bounded subprocess phases.
- Trade clock runtime:
  - release root: `F:\quant_data\Ashare\data\trade_release_v1`
  - clock state root: `F:\quant_data\Ashare\data\trade_clock`
  - OMS truth root: `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1`
  - automation pack root: `F:\quant_data\Ashare\outputs\automation_runs`
  - latest release pointer: `F:\quant_data\Ashare\data\trade_release_v1\latest_release.json`
  - latest clock heartbeat: `F:\quant_data\Ashare\data\trade_clock\clock_state.json`
  - phase-state root: `F:\quant_data\Ashare\data\trade_clock\phase_state`
  - scheduler-runtime state: `F:\quant_data\Ashare\data\trade_clock\runtime\scheduler_runtime.json`
  - trade-clock runtime log root: `F:\quant_data\Ashare\data\trade_clock\runtime`
  - latest safety truth: `F:\quant_data\Ashare\data\trade_clock\system_safety_state.json`
  - incident log: `F:\quant_data\Ashare\data\trade_clock\incident_log.jsonl`
  - manual overrides: `F:\quant_data\Ashare\data\trade_clock\manual_overrides.json`
  - latest account health probe: `F:\quant_data\Ashare\data\trade_clock\latest_account_health.json`
  - latest OMS actual-state truth: `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\latest_actual_portfolio_state.json`
  - manual operator scripts:
    - start now: `F:\quant_data\Ashare\scripts\start_trade_clock.ps1`
    - stop now: `F:\quant_data\Ashare\scripts\stop_trade_clock.ps1`
    - show status: `F:\quant_data\Ashare\scripts\show_trade_clock_status.ps1`
  - legacy autostart scripts:
    - install: `F:\quant_data\Ashare\scripts\install_trade_clock_autostart.ps1` (deprecated; now prints guidance only)
    - remove: `F:\quant_data\Ashare\scripts\remove_trade_clock_autostart.ps1` (removes old scheduled task if one exists)
  - heartbeat/safety split:
    - `clock_state.json` is now the lightweight heartbeat
    - `system_safety_state.json` is the current safety truth
    - `incident_log.jsonl` is the append-only abnormal-event ledger
- Daily automation scheduler truth:
  - automatic production profile: `daily_production`
  - profile runtime weight:
    - above `quick_test`
    - below `overnight`
  - fixed V5 cycles: `3`
  - current phase order:
    - `research`
    - `release`
    - `preopen_gate`
    - `simulation`
    - `shadow`
    - `summary`
  - current production model:
    - manual PowerShell start
    - long-lived but lightweight scheduler process
    - child subprocesses for all heavy work
    - fallback release path if nightly research fails or times out
  - namespace split:
    - simulation OMS truth lives under `...\oms_v1\simulation`
    - shadow OMS truth lives under `...\oms_v1\shadow`
- Current recommended commands:
  - operator plain-language guide:
    - `F:\quant_data\Ashare\SYSTEM_DAILY_USAGE_GUIDE_CN.txt`
  - `python F:\quant_data\Ashare\launch_canonical.py`
  - `python F:\quant_data\Ashare\launch_canonical.py --profile overnight`
  - `python F:\quant_data\Ashare\launch_canonical.py --profile quick_test`
  - `python F:\quant_data\Ashare\launch_canonical.py --profile daily_production`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode industry_router_only --profile quick_test`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode resume_downstream --profile quick_test`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode research_only --profile quick_test`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode research_only --profile daily_production`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode release_only --profile quick_test`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode release_only --profile daily_production`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode execution_only --profile quick_test --gate-only`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode execution_only --profile daily_production --gate-only`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode execution_only --profile quick_test --execution-mode simulation --gate-only`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode execution_only --profile quick_test --execution-mode precision --precision-trade off --gate-only`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode execution_only --profile quick_test --execution-mode precision --precision-trade on`
  - `python F:\quant_data\Ashare\trade_clock_service.py --profile daily_production --once`
  - `powershell -ExecutionPolicy Bypass -File F:\quant_data\Ashare\scripts\start_trade_clock.ps1 -Profile daily_production`
  - `powershell -ExecutionPolicy Bypass -File F:\quant_data\Ashare\scripts\stop_trade_clock.ps1`
  - `powershell -ExecutionPolicy Bypass -File F:\quant_data\Ashare\scripts\show_trade_clock_status.ps1`
  - `notepad F:\quant_data\Ashare\data\trade_clock\manual_overrides.json`
- Runtime transparency:
  - `main_research_runner.py` now prints a stage preview before dispatch.
  - `supervisor_state.json` is updated incrementally during integrated runs instead of only at the end.
  - `supervisor_state.json` now carries `current_stage`, `stages`, and `stage_history` for operator inspection.
  - `supervisor_state.json` now also carries recent `runtime_notes` for selected long stages.
  - V6 now emits additive sidecar artifacts for `announcement_evidence_cards.json` and `manual_review_queue.json`.
  - V6 now also emits a formal `industry_router` research skeleton with stock-level signals, mechanism-state tables, and split backtest reports.
  - V6 now also emits a formal market-state layer with `latest_market_state.json`, `market_state_daily.csv`, and `market_state_explainer.json`.
  - Portfolio recommendation now emits a formal technical-confirmation layer with `latest_technical_confirmation.csv` and `technical_confirmation_summary.json`.
  - Technical confirmation now supports a softer `pilot` path for borderline new entries and a graded `reduce_watch` path for weak existing positions instead of relying only on hard zero/one gating.
  - Portfolio recommendation now supports post-filter reweighting so weak screening does not leave the system trapped in meaningless low-exposure books when capacity still exists under the current regime cap.
  - V5 completion can now emit `latest_v5_cycle_review.json` as a local post-cycle review.
  - execution runs now emit a portfolio-control audit layer with `position_state_before/after_plan/after_execution`, `rebalance_audit.json`, and `execution_feedback.json`.
  - the execution layer can now refresh a dedicated `Latest Live Portfolio Snapshot` block inside this dev log after successful bridge runs.
  - the trade clock now also emits a separate safety truth file, incident log, and lightweight gmtrade account-health probe.
- Local Ollama split:
  - strict local event extraction stays on `qwen2.5:7b`
  - local research fallback chain is now `deepseek-r1:14b -> qwen2.5:7b`
  - local announcement evidence-card generation and V5 cycle review use `deepseek-r1:14b` by default
  - local manual-review routing and runtime stage explainer use `qwen2.5:7b` by default
- Portfolio control V1:
  - current low-risk scope is `ledger + drift threshold + daily turnover budget + execution feedback + dev-log portfolio snapshot`
  - industry/theme exposure and staged build/reduce are intentionally deferred for now
- Portfolio V2A:
  - the research-side portfolio layer now has a formal deterministic V2A posture engine, lifecycle engine, and admission/replacement layer under `hub_v6\portfolio_v2a`
  - V2A now upgrades target generation from flat rank-weight output into:
    - portfolio posture
    - per-name lifecycle state
    - action intent
    - admission/replacement audit
  - current lifecycle states are:
    - `watch`
    - `pilot`
    - `build`
    - `hold`
    - `trim`
    - `exit`
  - current V2A scope is portfolio-construction-side and now explicitly separated from OMS truth:
    - it influences `portfolio_recommendation.json`, `target_positions.csv`, `release_manifest.json`, and execution audit metadata
    - it now prefers OMS actual-state truth for continuity when available
    - it still does not make intraday execution micro-decisions
- OMS narrow but complete truth layer:
  - a formal OMS package now lives under `hub_v6\oms`
  - authority split is now:
    - research/V2A owns `desired_state`
    - release owns contract freeze/versioning
    - safety/trade-clock owns permission to attempt execution
    - OMS owns broker/account truth, order/fill ledgers, desired-vs-actual gap truth, and `actual_state`
  - the execution bridge is now an operational arm of OMS/runtime, not the authoritative lifecycle ledger
  - `live_execution_bridge\runtime.py` is now a compatibility wrapper that delegates to OMS runtime
- Industry router Deepened Three-Mechanism Architecture:
  - static contracts now live under `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\industry_router`
  - runtime code is now split into `contracts\`, `core\`, and `mechanisms\`
  - top-level `hub_v6\industry_router\runtime.py` and `backtest.py` are orchestration wrappers only
  - `contracts\stock_profile_schema.py`, `mechanism_state_schema.py`, `signal_schema.py`, and `backtest_schema.py` now define the formal research-side schema
  - runtime artifacts now live under `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router`
  - current mechanism groups are fixed to:
    - `trend_capex`
    - `price_inventory`
    - `macro_style`
  - each mechanism now owns its own `profile_view.py`, `source_ingest.py`, `state_builder.py`, `mapping_rules.py`, `signal_builder.py`, `risk_rules.py`, `policy.py`, and `config.json`
  - formal runtime contracts now include:
    - `stock_profile.csv`
    - `mechanism_state_daily.csv`
    - `core_variable_daily.csv`
    - `stock_signal_daily.csv`
  - mechanism sidecar outputs now also include `<mechanism>_profile.csv`, `<mechanism>_state.csv`, `<mechanism>_core_variable.csv`, and `<mechanism>_signal.csv`
  - `source_contracts.json` is now a structured contract with `mode / url / keywords / weight`, not just a name list
  - official source fetch is wired into runtime and emits `source_state_daily.csv`, `source_snapshot_index.json`, and `source_snapshot_items.json`
  - split backtest attribution now preserves zero-trade candidate buckets so all three mechanisms remain visible in attribution outputs even when the latest dates lack future bars
  - stale legacy `industry_state_daily.csv` is actively removed during rebuild so the output directory does not keep dual truths
  - current phase still uses a seed pool, not a full-market classification engine
- Latest confirmed milestone:
  - V6 research plan generation confirmed on `2026-03-21 14:09:34`
  - quick_test V5 cycle observed generating new candidates on `2026-03-21 14:39:15`
  - downstream recovery plus fresh portfolio output confirmed on `2026-03-21 15:32`
  - execution bridge connectivity confirmed on `2026-03-21 15:39:49`
  - `industry_router_only` formal skeleton run confirmed on `2026-03-22 04:18:14`
  - deepened three-mechanism `industry_router_only` run confirmed on `2026-03-22 19:49:01`
  - `plan_only` integration with deepened industry-router context confirmed on `2026-03-22 19:47:22`
  - market-state and technical-confirmation portfolio integration confirmed on `2026-03-22 21:06:22`
  - market-state-aware `execution_only --gate-only` release check confirmed on `2026-03-22 21:06:22`
  - softened market-state / technical-confirmation posture plus post-filter reweight confirmed on `2026-03-22 21:38:38`
  - Portfolio V2A posture/lifecycle/admission integration plus release/gate compatibility confirmed on `2026-03-22 22:17:56`
  - OMS authority refactor plus broker-truth-first artifact emission probe confirmed on `2026-03-22 23:29:33`
  - post-OMS `execution_only --gate-only` split-path compatibility reconfirmed on `2026-03-22 23:30:06`
- Current truth:
  - old V6 readmes pointing to `run_v6_full_cycle_real.py` are stale
  - this log is the current source of truth

## Latest Live Portfolio Snapshot
<!-- LIVE_PORTFOLIO_SNAPSHOT_START -->
- Updated at: `20260323_002316`
- Source report: `F:\quant_data\Ashare\data\live_execution_bridge\execution_report_20260323_002316.json`
- Account: `oms_...ount`
- NAV: `10000.0000`
- Cash: `8000.0000`
- Positions: `1`
- Target names: `1`
- Orders/Fills: `1` / `0`
- Turnover raw/final: `0.2000` / `0.2000`
- Drift skipped: `0`
- Turnover adjustments: `0`
- Execution status summary: `success=0 partial=0 failed=1 skipped=0`
- Top holdings:
- `600010.SH`: weight=0.2000, shares=200, price=10.0000
<!-- LIVE_PORTFOLIO_SNAPSHOT_END -->
## Session Start Checklist
- Read `Latest Stable Snapshot`, `Latest Live Portfolio Snapshot`, `Known Dangerous Operations`, and `Known Issues` before touching code.
- If the task touches precise-style execution, also inspect `data\trade_release_v1\latest_release.json` and `data\trade_clock\clock_state.json` first.
- Confirm whether the user has explicitly allowed any long-running integrated run in the current session.
- Use `launch_canonical.py` plus the documented profile for formal operator runs.
- Use `main_research_runner.py` when you need to inspect or reason about the wrapped business chain directly.
- If a change alters current runtime truth, update the stable sections first, then append a new change-log entry.

## Run Profile Quick Reference
| Profile | Intent | V6 Plan Reuse | V5 Cycles | Use Case | Notes |
| --- | --- | --- | --- | --- | --- |
| `daily_production` | bounded daily production research | `24h` | `3` | stable daily automation chain, release generation, and next-day execution prep | automatic mainline profile; designed to finish on time and tolerate fallback |
| `overnight` | full nightly research | `24h` | `8` | sleep-time full research pass | heavy nightly mode; highest runtime cost |
| `quick_test` | minimal full-chain debug | `24h` | `1` | faster debugging of the integrated chain | current code default; not a smoke test |

## Runtime Mode Matrix
- Current mode axes:
  - `profile`: runtime weight selector
  - `mode`: which chain segment to run
  - `execution_mode`: which gmtrade account profile to bind
  - `precision_trade`: whether precision mode is allowed to actually submit the execution bridge
- Current code defaults:
  - profile: `quick_test`
  - mode: `integrated_supervisor`
  - execution_mode: `precision`
  - precision_trade: `False`

| Mode | What It Runs | Typical Use | Direct Execution Behavior |
| --- | --- | --- | --- |
| `integrated_supervisor` | market pipeline -> strategy feedback -> V6 plan -> V5.1 -> portfolio recommendation -> optional execution bridge | integration debug and old one-shot chain | simulation can still execute; precision is blocked by default unless `ALLOW_INTEGRATED_PRECISION_EXECUTION=True` |
| `research_only` | market pipeline -> strategy feedback -> V6 plan -> V5.1 -> portfolio recommendation -> release publish | formal research-side production; daily scheduler uses this for nightly bounded research | does not directly execute |
| `release_only` | republishes latest portfolio artifacts into `trade_release_v1` | refresh release after portfolio files changed, after recovery, or from explicit fallback source artifacts | does not directly execute |
| `execution_only` | reads latest release -> checks trade gate/window -> dispatches execution bridge if allowed | formal execution-side entry | simulation is release-driven; precision is time-gated and obeys `precision_trade` |
| oms_validate | bounded OMS synthetic/replay validation harness | OMS hardening regression checks after refactors | no execution bridge; no live broker required |
| `resume_downstream` | restart from portfolio recommendation and optionally rerun execution | downstream recovery after V5 finished but later stages failed | precision direct execution is blocked by default |
| `full_cycle` | ingest -> extract -> industry router -> gap -> plan -> bridge artifacts | V6 subchain inspection | no execution bridge |
| `ingest_only` | market/base-table refresh and raw event ingest | upstream source debugging | no execution bridge |
| `extract_only` | ingest + structured event extraction | event extraction debugging | no execution bridge |
| `gap_only` | ingest + extract + data-gap analysis | data-gap debugging | no execution bridge |
| `industry_router_only` | rebuild stock profile, mechanism state, core variables, signal table, and unified split backtests from existing event history | targeted validation of the three-mechanism unified research architecture | no execution bridge |
| `plan_only` | ingest + extract + industry router + gap + research plan generation | V6 planning debugging with mechanism-aware context | no execution bridge |
| `bridge_only` | ingest + extract + industry router + gap + plan + bridge artifacts | bridge payload debugging with mechanism-aware context | no execution bridge |

## Mode And Config Control Surface
- Formal operator entry:
  - `F:\quant_data\Ashare\launch_canonical.py`
- Wrapped business root:
  - `F:\quant_data\Ashare\main_research_runner.py`
- Long-term defaults are hand-edited in:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
- Teammate-safe template lives in:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
- Gmtrade account binding and broker parameters are hand-edited in:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\gmtrade_runtime_config.local.json`
- Do not hand-edit generated runtime files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\hub_config.v6.runtime.<profile>.json`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\gmtrade_runtime_config.autogen.json`

| What You Want To Change | Where To Change It | Current Truth |
| --- | --- | --- |
| default run mode | `hub_v6/local_settings.py` -> `RUN_MODE` | `integrated_supervisor` |
| default profile | `hub_v6/local_settings.py` -> `DEFAULT_RUN_PROFILE` | `quick_test` |
| one-off run mode override | `launch_canonical.py --mode ...` or `main_research_runner.py --mode ...` | operator override only; does not rewrite defaults |
| one-off profile override | `launch_canonical.py --profile overnight|daily_production|quick_test` | operator override only |
| default account mode | `hub_v6/local_settings.py` -> `EXECUTION_ACCOUNT_MODE` | `precision` |
| one-off account mode override | `--execution-mode simulation|precision` | operator override only |
| precision live-trade master switch | `hub_v6/local_settings.py` -> `PRECISION_TRADE_ENABLED` | `False` |
| one-off precision-trade override | `--precision-trade on|off` | operator override only |
| manual scheduler start/stop | `scripts/start_trade_clock.ps1`, `scripts/stop_trade_clock.ps1` | manual operator process; no autostart |
| scheduler status view | `scripts/show_trade_clock_status.ps1` | concise read-only operator summary over clock / release / safety / phase / OMS |
| scheduler default profile | `hub_v6/local_settings.py` -> `TRADE_CLOCK_SCHEDULER_PROFILE` | `daily_production` |
| scheduler phase times | `hub_v6/local_settings.py` -> `TRADE_CLOCK_PHASE_*_TIME` | research=15:05, release=15:10, preopen_gate=09:20, simulation=09:30:35, shadow=09:35, summary=15:20 |
| scheduler phase timeouts | `hub_v6/local_settings.py` -> `TRADE_CLOCK_*_TIMEOUT_MINUTES` | current local values; bounded per phase |
| automation pack root | `hub_v6/local_settings.py` -> `AUTOMATION_RUNS_ROOT` | `F:\quant_data\Ashare\outputs\automation_runs` |
| scheduler simulation/shadow namespace split | `hub_v6/local_settings.py` -> `TRADE_CLOCK_SIMULATION_NAMESPACE`, `TRADE_CLOCK_SHADOW_NAMESPACE` | `simulation` / `shadow` |
| one-off execution namespace override | `--execution-namespace ...` | operator override only |
| one-off shadow execution flag | `--shadow-run` | operator override only |
| whether integrated modes may directly execute precision | `hub_v6/local_settings.py` -> `ALLOW_INTEGRATED_PRECISION_EXECUTION` | `False` |
| simulation / precision account ids | `configs/gmtrade_runtime_config.local.json` -> `broker.account_profiles` | simulation=`4d74746e-243c-11f1-a169-00163e022aa6`; precision=`e18905e4-254f-11f1-b37d-00163e022aa6` |
| broker endpoint / buy-sell ratios / lot size / min trade value / cash reserve | `configs/gmtrade_runtime_config.local.json` | current live local template |
| trade release validity time | `hub_v6/local_settings.py` -> `TRADE_RELEASE_VALID_AFTER_TIME`, `TRADE_RELEASE_EXPIRES_AT_TIME` | `09:30:30` / `15:00:00` |
| clock polling interval | `hub_v6/local_settings.py` -> `TRADE_CLOCK_POLL_SECONDS` | `30` |
| automatic execution windows | `hub_v6/local_settings.py` -> `TRADE_CLOCK_EXECUTION_WINDOWS` | `09:30:30-10:00:00` |
| industry-router master switch | `hub_v6/local_settings.py` -> `ENABLE_INDUSTRY_ROUTER` | `True` |
| industry-router contract root | `hub_v6/local_settings.py` -> `INDUSTRY_ROUTER_CONTRACT_ROOT` | `...\configs\industry_router` |
| industry-router artifact root | `hub_v6/local_settings.py` -> `INDUSTRY_ROUTER_OUTPUT_ROOT` | `data\event_lake_v6\research\industry_router` |
| industry-router history window | `hub_v6/local_settings.py` -> `INDUSTRY_ROUTER_HISTORY_LOOKBACK_DAYS` | `14` |
| industry-router split backtest config | `hub_v6/local_settings.py` -> `INDUSTRY_ROUTER_ENABLE_BACKTEST`, `INDUSTRY_ROUTER_BACKTEST_HORIZONS`, `INDUSTRY_ROUTER_BACKTEST_TOP_K` | `True / [1, 2] / 3` |

- Practical operator rule:
  - for formal precise-style operation, prefer `research_only -> release_only if needed -> execution_only`
  - treat `integrated_supervisor` as an integration/debug mode, not the primary precision-trading production path

## Industry Router Snapshot
- Formal module root:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router`
- Formal internal layout:
  - `contracts\records.py`, `contracts\interfaces.py`, and the schema files define the unified contracts and policy interface
  - `core\` holds loaders, event normalization, source fetch, runtime engine, calendar alignment, metrics, signal loading, and backtest engine
  - `mechanisms\trend_capex|price_inventory|macro_style` each now own:
    - `config.json`
    - `profile_view.py`
    - `source_ingest.py`
    - `state_builder.py`
    - `mapping_rules.py`
    - `signal_builder.py`
    - `risk_rules.py`
    - `policy.py`
  - top-level `runtime.py` and `backtest.py` are thin orchestration entrypoints only
- Static contracts:
  - `stock_master.seed.csv`
  - `mechanism_map.seed.csv`
  - `event_taxonomy.json`
  - `source_contracts.json`
- Runtime outputs:
  - `stock_master.csv`
  - `mechanism_map.csv`
  - `stock_profile.csv`
  - `event_instances.csv`
  - `event_stock_mapping.csv`
  - `mechanism_state_daily.csv`
  - `source_state_daily.csv`
  - `core_variable_daily.csv`
  - `stock_signal_daily.csv`
  - `latest_stock_signal.csv`
  - `source_snapshots\source_snapshot_index.json`
  - `source_snapshots\source_snapshot_items.json`
  - `backtests\backtest_<mechanism>_*.json/csv`
  - `<mechanism>_profile.csv`
  - `<mechanism>_state.csv`
  - `<mechanism>_core_variable.csv`
  - `<mechanism>_signal.csv`
- Current Three-Mechanism Deepened truth:
  - the core subject is explicitly `stock`, not `industry` and not `event`
  - event records are treated as triggers and normalized before entering mechanism-specific mapping rules
  - the formal signal-production chain is now:
    - `stock_profile -> mechanism_state_daily -> core_variable_daily -> stock_signal_daily`
  - each mechanism now has its own:
    - stock-profile enrichment logic
    - source summarization logic
    - state builder
    - event-to-stock mapping rules
    - signal builder
    - risk / confirmation layer
    - policy / attribution rules
  - the unified backtest engine no longer hardcodes mechanism logic; it only aligns time, loads signals, evaluates policy hooks, and writes attribution
  - company-event sources bind to the existing announcement/news ingest chain through `existing_event_ingest` contracts
  - official industry/macro sources bind to a dedicated `source_fetch` sidecar and feed `source_state_daily.csv`
  - `trend_capex`, `price_inventory`, and `macro_style` all now have runnable source coverage, runtime outputs, and unified-backtest attribution rows
  - the rebuild path removes stale legacy `industry_state_daily.csv` and stale empty backtest detail files so output directories do not keep mixed schemas
  - the source layer still uses curated official page URLs, so freshness is controlled by decay weights rather than a rolling agency crawler

## Known Dangerous Operations
- Do not run the full integrated pipeline just to validate a small code edit.
- Do not bypass `launch_canonical.py` for a formal operator run unless the user explicitly asks to use the wrapped business root directly.
- Do not switch the Gmtrade bridge off `gmtrade39`.
- Do not hand-edit generated runtime configs such as `hub_config.v6.runtime.*.json`; they are regenerated.
- Do not assume `F:\quant_data\Ashare` is a normal git repository.
- Do not forget this workspace auto-pushes after `git commit` unless `DISABLE_AUTO_PUSH=1` is set.
- Do not assume the Google Drive mirror is active after a reboot unless the watcher has been started again.
- Do not restore archived legacy packages back into the repo root unless you are intentionally undoing the March 21 legacy-code cleanup.
- If the dev-log mirror appears broken after boot, check the scheduled task `Ashare Codex Dev Log Mirror` before changing the sync scripts.
- Do not spam Tushare news endpoints when quotas are already tight.
- Do not echo tokens or duplicate secrets into normal user-facing output.
- Do not treat `quick_test` as a seconds-level smoke test; it is still a real integrated run.
- Do not use `--resume-execution` casually after a prior bridge run; it can replay simulated orders on refreshed portfolio files.
- Do not treat the whole `outputs\` tree as canonical; only `outputs\canonical_runs` is the formal wrapper trace root unless another local marker says otherwise.
- Do not treat `integrated_supervisor` as the formal precise-trading path; it remains mainly a debug/integration mode.
- Do not use `--ignore-window` on `execution_only` unless the user explicitly wants to bypass the market-time gate.
- Do not let `trade_clock_service.py` and a manual `execution_only` call fight over the same release/window without checking `data\trade_clock\clock_state.json` first.

## Project Overview
- This is a complex medium-sized A-share research and execution system, not a small script project.
- The production chain is:
  1. Market data update and train-table append
  2. Raw event ingest from announcements plus Tushare news
  3. Structured event extraction
  4. Industry-router stock/mechanism signal layer and split backtest sidecar
  5. Market-state / capital-flow regime layer
  6. Data-gap analysis
  7. V6 research brief generation
  8. V5.1 GPU iterative research
  9. Portfolio V2A posture / lifecycle / admission-replacement construction
  10. Portfolio recommendation with technical confirmation and market-aware sizing
  11. Portfolio release publish
  12. Time-gated execution with safety guard and market-state turnover/reduce-only constraints
  13. Daily performance feedback back into the next run

## Authority Structure
- Research power:
  - `hub_v6/portfolio_recommendation.py`, `hub_v6/portfolio_v2a/*`, V6, and V5.1 may define hypotheses, ranking, desired target weights, and desired lifecycle suggestions.
  - Research is not allowed to define broker/account/order/fill truth.
- Release power:
  - `hub_v6/portfolio_release.py` freezes the contract for execution consumption.
  - Release does not learn from broker noise and does not derive actual-state truth.
- Safety/gate power:
  - `trade_clock_service.py`, `execution_manager.py`, and the safety truth files decide whether execution may be attempted.
  - Safety does not author account truth or lifecycle truth.
- OMS power:
  - `hub_v6/oms/*` owns:
    - broker/account snapshots
    - position/order/fill ledgers
    - desired-vs-actual gap
    - actual-state derivation
    - intent lifecycle truth
  - OMS consumes gate truth but does not replace the global safety layer.
- Execution power:
  - `live_execution_bridge` submits orders and reads broker replies under OMS governance.
  - `live_execution_bridge/runtime.py` is now a compatibility wrapper; the canonical execution-side truth runtime is `hub_v6/oms/runtime.py`.
- Feedback buckets:
  - Bucket A `truth feedback` stays inside OMS ledgers/artifacts only.
  - Bucket B `control feedback` flows from OMS into V2A posture pacing.
  - Bucket C `research meta feedback` flows only as aggregated summary into the research context pack.
  - Bucket D `narrative feedback` is non-authoritative and for human reading only.

## Active Runtime vs Archived Roots
- Active root entry:
  - `F:\quant_data\Ashare\main_research_runner.py`
- Active runtime package tree:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean`
- Active V6 orchestrator and supervisor:
  - `...\hub_v6`
- Active V5.1 research brain and iterator:
  - `...\v5_gpu_runtime`
  - launcher: `...\v5_gpu_runtime\run_research_hub_v5_1_local.py`
  - main control: `...\v5_gpu_runtime\hub\cli_v5.py`
- Active downstream outputs:
  - `F:\quant_data\Ashare\data`
- Archived root-level legacy packages:
  - `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321\quant_research_hub_v5`
  - `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321\quant_research_hub_v5_1`
- Operator warning:
  - The archived root-level `quant_research_hub_v5*` folders are not the live research engine anymore.
  - If you need the real research brain, inspect `v5_gpu_runtime` inside the active package tree, not the archived root copies.

## Core Module Relationship Map
- Root entry and runtime config:
  - `launch_canonical.py` is the formal operator entry.
  - `main_research_runner.py` is the wrapped business root entry.
  - `launch_canonical.py` also writes `run_manifest.json` through `tools/register_run.py` under `outputs\canonical_runs\<run_id>\`.
  - The wrapper does not replace business logic; it only performs governance-layer selection and preflight before dispatching to the wrapped root.
  - It reads `hub_v6/local_settings.py`, calls `hub_v6/config_builder.py`, and generates `configs/hub_config.v6.runtime.<profile>.json`.
  - It then dispatches by mode:
    - `integrated_supervisor` -> `hub_v6/supervisor.py`
    - `resume_downstream` -> `hub_v6/supervisor.py::run_resume_downstream`
    - other V6 partial modes -> `hub_v6/orchestrator_v6.py`
- Supervisor layer:
  - `hub_v6/supervisor.py` is the daily conductor for the full chain.
  - It calls `hub_v6/market_pipeline.py`, refreshes strategy feedback, decides whether V6 planning should run, launches V5 GPU research, then calls portfolio recommendation and the execution bridge.
  - It writes top-level run state to `data/event_lake_v6/research/supervisor/supervisor_state.json`.
  - During runtime it now updates `current_stage`, per-stage status, and recent stage history incrementally so operators can see where the chain is currently sitting.
- Market and event layer:
  - `hub_v6/market_pipeline.py` updates HS300, enriched daily files, price snapshots, and the training table.
  - `hub_v6/event_ingest.py` collects raw announcements and Tushare news into the event lake.
  - `hub_v6/event_extract.py` converts raw event text into structured event objects with quality and anti-overfit metadata.
  - `hub_v6/industry_router/` now sits between extracted events and the higher research context as the formal stock/mechanism skeleton.
  - `hub_v6/industry_router/runtime.py` builds `stock_master`, `event_instances`, `event_stock_mapping`, `mechanism_state_daily`, and `stock_signal_daily`.
  - `hub_v6/industry_router/backtest.py` runs the split mechanism backtest skeleton over the generated signal table.
  - `hub_v6/market_state/runtime.py` builds market-state regime truth from trend, breadth, liquidity proxies, style balance, and industry-router bias.
  - `hub_v6/data_gap_engine.py` identifies missing derived features or refresh actions.
  - `hub_v6/context_pack.py` now merges extracted events, industry-router summary, market-state truth, gap findings, and bridge context into `research_context_pack.json`.
- V6 planning layer:
  - `hub_v6/research_brief_engine.py` reads the context pack, including market-state truth and mechanism-aware context, and produces `research_brief.json`.
  - `hub_v6/llm_router.py` is the provider abstraction for OpenAI / DeepSeek / local Ollama.
  - `hub_v6/v5_bridge.py` converts the research brief into bridge artifacts such as `candidate_override.json` under `data/event_lake_v6/bridge`.
- Bridge boundary between V6 and V5:
  - `data/event_lake_v6/bridge` is the main handoff directory between the supervisor/V6 side and the V5 research engine.
  - Important bridge files include:
    - `candidate_override.json`
    - `enriched_context.json`
    - `performance_feedback.json`
    - `last_token_plan.json`
- V5.1 GPU research runtime:
  - The live V5.1 runtime is the embedded package tree `quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\v5_gpu_runtime`.
  - The archived root-level `quant_research_hub_v5` and `quant_research_hub_v5_1` directories are not imported by `main_research_runner.py`.
  - `v5_gpu_runtime/run_research_hub_v5_1_local.py` launches the research brain and immediately hands off to `v5_gpu_runtime/hub/cli_v5.py`.
  - `v5_gpu_runtime/hub/cli_v5.py` drives the adaptive research loop and writes `controller_state.json` plus per-cycle `cycle_summary.json`.
  - `candidate_factory.py` creates route/model/feature candidates, partly influenced by bridge overrides and performance feedback.
  - `codegen.py` builds candidate-local feature/model/training files in each lab workspace.
  - `single_run_v5.py` runs one candidate end-to-end.
  - `training_engine.py` handles dataset prep, feature selection, model fit, prediction, and train summaries.
  - `model_families.py` defines actual model backends such as `xgboost_gpu`, stabilized `ridge_ranker`, and fallbacks.
  - `registry.py` appends run-level results into `registry/experiment_registry.csv`.
  - `strategy_family.py` and deployment gates summarize family-level status after each cycle.
- Portfolio recommendation layer:
  - `hub_v6/portfolio_v2a/` is now the formal deterministic posture/state/admission engine between raw candidate ranking and final target-book publication.
  - `portfolio_v2a/exposure_engine.py` turns market-state + safety linkage + current book posture into total-cap, new-entry-budget, add-budget, and rebalance mode.
  - `portfolio_v2a/lifecycle_engine.py` turns each candidate into lifecycle state, action intent, size confidence, target cap, and proposal weight.
  - `portfolio_v2a/admission_engine.py` decides new-entry admission and replacement pressure when slots are limited.
  - `portfolio_v2a/runtime.py` writes formal sidecars:
    - `latest_portfolio_posture.json`
    - `latest_position_lifecycle.csv`
    - `position_lifecycle_daily.csv`
    - `admission_replacement_audit.json`
    - `portfolio_control_summary.json`
  - `hub_v6/technical_confirmation/runtime.py` scores candidate-level trend, volume, stretch, and hold-health confirmation before final portfolio selection.
  - `hub_v6/portfolio_recommendation.py` reads the latest valid V5 result, resolves the corresponding run directory, loads `latest_portfolio_v1.csv`, applies market-state-aware sizing plus technical confirmation, attaches price context, and writes:
    - `portfolio_recommendation.json`
    - `target_positions.csv`
    - `rebalance_orders.csv`
  - It consumes both V5 outputs and the latest market snapshot files, reads `performance_feedback.json` for posture overrides, and also reads:
    - latest market-state truth
    - latest technical-confirmation table
    - latest industry-router signal context
  - Market state can now directly tighten `max_names`, `total_exposure_cap`, and `single_name_cap`.
  - Technical confirmation can now directly reject weak entries or weight down marginal candidates before the release layer snapshots the final target book.
  - V2A now directly decides:
    - lifecycle state
    - action intent
    - target-weight cap
    - proposal target weight
    - admission/replacement outcome
- Release layer:
  - `hub_v6/portfolio_release.py` is the new middle layer between research and execution.
  - It reads the latest portfolio recommendation artifacts and publishes a versioned release under `data\trade_release_v1\releases\<release_id>\`.
  - The release manifest now also snapshots market-state truth and technical-confirmation summary so execution can read one coherent release contract instead of recomputing posture.
  - It now also snapshots V2A posture and lifecycle-sidecar pointers so release consumers can inspect the portfolio state machine without rerunning research logic.
  - It also maintains:
    - `latest_release.json`
    - `latest\release_manifest.json`
    - `latest\target_positions.csv`
- Precision execution gate:
  - `hub_v6/execution_manager.py` owns `execution_only`.
  - It reads the published release, checks trading day plus execution window, evaluates safety, applies market-state turnover and reduce-only posture, and only then dispatches the execution bridge.
  - `hub_v6/trading_clock.py` owns A-share clock windows and the cached trade-calendar check.
  - `hub_v6/clock_supervisor.py` is the lightweight heartbeat loop used by `trade_clock_service.py`.
- Execution layer:
  - `hub_v6/oms/runtime.py` is now the canonical execution-side truth runtime.
  - It owns:
    - account ledger
    - position ledger
    - intent ledger
    - order ledger
    - fill ledger
    - desired-vs-actual gap
    - actual-state derivation
  - `live_execution_bridge/runtime.py` remains as a compatibility wrapper and simply delegates into OMS runtime.
  - `live_execution_bridge/portfolio_control.py` is now a deterministic dispatch-planning and audit utility under OMS governance, not the authoritative source of actual state.
  - `portfolio_control.py` still carries through V2A metadata from `target_positions.csv` into execution-side position-state audit rows so operator audit can see lifecycle state and action intent next to share math.
  - `live_execution_bridge/brokers/gmtrade_sim_broker.py` is the real gmtrade simulation adapter currently in use.
  - The OMS/execution layer consumes the published release target book plus broker/account truth, then writes:
    - `latest_actual_portfolio_state.json`
    - `desired_vs_actual_gap.csv`
    - `oms_summary.json`
    - `intent_ledger_latest.csv`
    - `order_ledger_latest.csv`
    - `fill_ledger_latest.csv`
    - `actual_state_daily.csv`
    - `execution_report_*.json`
    - `orders_*.csv`
    - `fills_*.csv`
    - `latest_account_state.json` (compatibility snapshot only; OMS artifacts are authoritative)
    - `equity_curve.csv`
- Feedback loop:
  - After execution, `hub_v6/supervisor.py` reads `equity_curve.csv` and writes `performance_feedback.json`.
  - OMS also emits:
    - Bucket A truth feedback into OMS ledgers/artifacts only
    - Bucket B control feedback into `control_feedback_latest.json` for V2A posture pacing
    - Bucket C research meta feedback into `research_meta_feedback_latest.json`, which is now injected into the research context pack as aggregated execution realism
    - Bucket D narrative feedback into a non-authoritative human-readable sidecar
  - That feedback is then consumed on the next run by:
    - `hub_v6/supervisor.py` itself
    - `v5_gpu_runtime/hub/candidate_factory.py`
    - `hub_v6/portfolio_recommendation.py`
- Downstream recovery mode:
  - `resume_downstream` intentionally skips market update, V6 planning, and V5 research.
  - It starts from the latest completed V5 artifacts and reruns only:
    - portfolio recommendation
    - optional execution bridge
  - Use this when V5 already finished but downstream outputs failed or need to be replayed.

## Artifact Registry
| Artifact | Producer | Consumer | Path | Format | Notes |
| --- | --- | --- | --- | --- | --- |
| `supervisor_state.json` | integrated supervisor | operator / debugging | `F:\quant_data\Ashare\data\event_lake_v6\research\supervisor\supervisor_state.json` | JSON | first stop for top-level step status; now includes `current_stage`, `stages`, and `stage_history` during runtime |
| `runtime_stage_notes.json` | supervisor local explainer sidecar | operator / debugging | `F:\quant_data\Ashare\data\event_lake_v6\research\supervisor\runtime_stage_notes.json` | JSON | rolling operator notes for selected long stages with suggested watch files |
| `market_pipeline_report.json` | market pipeline | operator / debugging | `F:\quant_data\Ashare\data\daily_cache_v6\market_pipeline_report.json` | JSON | shows data sync and train append status |
| `research_context_pack.json` | context pack builder | research brief engine / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\context_pack\research_context_pack.json` | JSON | full evidence pack |
| `announcement_evidence_cards.json` | V6 additive local evidence-card sidecar | research brief engine / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\evidence_cards\announcement_evidence_cards.json` | JSON | compact high-value announcement evidence cards; additive only |
| `research_brief.json` | V6 research planner | V5 bridge / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\briefs\research_brief.json` | JSON | core planning artifact |
| `manual_review_queue.json` | event-extract local review-router sidecar | operator / debugging | `F:\quant_data\Ashare\data\event_lake_v6\research\extract_summary\manual_review_queue.json` | JSON | compact queue of events worth manual review; additive only |
| `run_manifest.json` | formal governance wrapper | operator / debugging | `F:\quant_data\Ashare\outputs\canonical_runs\<run_id>\run_manifest.json` | JSON | run id, operator entry, runtime root, mode/profile, and trace metadata |
| `industry_router_summary.json` | industry-router runtime | context pack / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\industry_router_summary.json` | JSON | high-level summary over mechanism groups, latest active signals, and split-backtest status |
| `latest_market_state.json` | market-state runtime | context pack / portfolio recommendation / release / execution gate / operator | `F:\quant_data\Ashare\data\market_state_v6\latest_market_state.json` | JSON | current market regime truth with regime score, style bias, mechanism bias, exposure/turnover multipliers, and entry posture |
| `market_state_daily.csv` | market-state runtime | operator / debugging | `F:\quant_data\Ashare\data\market_state_v6\market_state_daily.csv` | CSV | historical daily market-state rows with sub-score decomposition and final posture fields |
| `market_state_explainer.json` | market-state runtime | operator / debugging | `F:\quant_data\Ashare\data\market_state_v6\market_state_explainer.json` | JSON | concise explanation of the latest regime drivers, thresholds, and policy decisions |
| `stock_master.csv` | industry-router runtime | event mapper / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\stock_master.csv` | CSV | resolved stock master with mechanism and subchain tags |
| `mechanism_map.csv` | industry-router runtime | event mapper / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\mechanism_map.csv` | CSV | symbol-to-mechanism mapping contract used by the router |
| `stock_profile.csv` | industry-router runtime | mechanism policies / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\stock_profile.csv` | CSV | resolved stock profile contract after merging stock master and mechanism map |
| `<mechanism>_profile.csv` | industry-router runtime | mechanism debugging / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\<mechanism>_profile.csv` | CSV | mechanism-sliced profile view after profile enrichment |
| `event_instances.csv` | industry-router runtime | mapper / backtest / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\event_instances.csv` | CSV | normalized event instances with resolved mechanism group and direction |
| `event_stock_mapping.csv` | industry-router runtime | scorer / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\event_stock_mapping.csv` | CSV | event-to-stock mapping with score, reason, and exposure level |
| `mechanism_state_daily.csv` | industry-router runtime | scorer / context pack / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\mechanism_state_daily.csv` | CSV | unified mechanism-state contract with `scope_type`, state score, source score, heat, and regime label |
| `<mechanism>_state.csv` | industry-router runtime | mechanism debugging / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\<mechanism>_state.csv` | CSV | mechanism-sliced state output with sub-state columns and drivers |
| `source_state_daily.csv` | industry-router source sidecar | scorer / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\source_state_daily.csv` | CSV | mechanism-level official-source state rows derived from structured source contracts |
| `core_variable_daily.csv` | industry-router runtime | signal policy / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\core_variable_daily.csv` | CSV | per-stock per-date core variables before policy scoring and risk filtering |
| `<mechanism>_core_variable.csv` | industry-router runtime | mechanism debugging / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\<mechanism>_core_variable.csv` | CSV | mechanism-sliced core-variable table before signal generation |
| `source_snapshot_index.json` | industry-router source sidecar | operator / debugging | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\source_snapshots\source_snapshot_index.json` | JSON | compact fetch index with ok/error counts and `as_of_date` |
| `source_snapshot_items.json` | industry-router source sidecar | operator / debugging | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\source_snapshots\source_snapshot_items.json` | JSON | per-source fetch results, extracted publish dates, keyword hits, and signal scores |
| `stock_signal_daily.csv` | industry-router runtime | backtest / context pack / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\stock_signal_daily.csv` | CSV | unified stock-level policy output with `pre_risk_score`, `final_score`, `signal_state`, `allow_entry`, and attribution bucket |
| `<mechanism>_signal.csv` | industry-router runtime | mechanism debugging / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\<mechanism>_signal.csv` | CSV | mechanism-sliced signal table with penalty / confirmation / attribution columns |
| `latest_stock_signal.csv` | industry-router runtime | context pack / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\latest_stock_signal.csv` | CSV | latest-date slice of stock signals, useful for quick inspection |
| `backtest_trend_capex_*` | industry-router backtest skeleton | operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\backtests\backtest_trend_capex_*` | JSON/CSV | split backtest outputs for trend-capex seeds |
| `backtest_price_inventory_*` | industry-router backtest skeleton | operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\backtests\backtest_price_inventory_*` | JSON/CSV | split backtest outputs for price/inventory seeds |
| `backtest_macro_style_*` | industry-router backtest skeleton | operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\backtests\backtest_macro_style_*` | JSON/CSV | split backtest outputs for macro/style seeds |
| `backtest_combined_*` | industry-router backtest skeleton | operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\backtests\backtest_combined_*` | JSON/CSV | combined equal-weight summary over the split mechanism picks |
| `backtest_attribution_*` | industry-router backtest skeleton | operator | `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router\backtests\backtest_attribution_*` | JSON/CSV | component-level attribution summary by mechanism; zero-trade candidate buckets are preserved when future bars are unavailable |
| `candidate_override.json` | V5 bridge | V5.1 runtime | `F:\quant_data\Ashare\data\event_lake_v6\bridge\candidate_override.json` | JSON | tells V5 what routes, models, labels to favor |
| `latest_v5_cycle_review.json` | supervisor local V5 review sidecar | operator / debugging | `F:\quant_data\Ashare\data\research_hub_v5_1_gpu_integrated\reviews\latest_v5_cycle_review.json` | JSON | concise local review over latest completed V5 cycle; additive only |
| `portfolio_recommendation.json` | portfolio recommendation layer | operator / execution bridge | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\portfolio_recommendation.json` | JSON | summary of selected strategy and portfolio state, now also including market-state posture, technical-confirmation summary, and post-filter reweight totals |
| `latest_portfolio_posture.json` | portfolio V2A runtime | operator / release / audit | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\portfolio_v2a\latest_portfolio_posture.json` | JSON | portfolio-level posture contract with exposure cap, new-entry budget, add budget, rebalance mode, safety linkage, and replacement aggressiveness |
| `latest_position_lifecycle.csv` | portfolio V2A runtime | operator / release / audit | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\portfolio_v2a\latest_position_lifecycle.csv` | CSV | latest per-name lifecycle state, action intent, proposal weight, final weight, cap, and reason fields |
| `position_lifecycle_daily.csv` | portfolio V2A runtime | operator / replay / audit | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\portfolio_v2a\position_lifecycle_daily.csv` | CSV | rolling history of lifecycle states for replay and threshold scans |
| `admission_replacement_audit.json` | portfolio V2A runtime | operator / release / audit | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\portfolio_v2a\admission_replacement_audit.json` | JSON | explains which new names were admitted, denied, or used to replace weaker incumbents |
| `portfolio_control_summary.json` | portfolio V2A runtime | operator / release / audit | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\portfolio_v2a\portfolio_control_summary.json` | JSON | V2A state counts, exposure usage, replacement counts, and soft-crowding snapshot |
| `latest_technical_confirmation.csv` | technical-confirmation runtime | portfolio recommendation / release / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\technical_confirmation\latest_technical_confirmation.csv` | CSV | latest candidate-level technical gate output with entry allow flag, gate reason, and weight multiplier |
| `technical_confirmation_daily.csv` | technical-confirmation runtime | operator / debugging | `F:\quant_data\Ashare\data\event_lake_v6\research\technical_confirmation\technical_confirmation_daily.csv` | CSV | rolling technical-confirmation history across candidate symbols and dates |
| `technical_confirmation_summary.json` | technical-confirmation runtime | portfolio recommendation / release / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\technical_confirmation\technical_confirmation_summary.json` | JSON | latest confirmation counts, strictness level, and summary posture |
| `target_positions.csv` | portfolio recommendation layer | Gmtrade execution bridge | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\target_positions.csv` | CSV | target holdings with price fields, technical-confirmation fields, market-aware sizing metadata, post-filter reweighted portfolio weights, and V2A lifecycle/action-intent fields |
| `rebalance_orders.csv` | portfolio recommendation layer | operator / execution bridge | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\rebalance_orders.csv` | CSV | delta orders relative to prior holdings |
| `latest_release.json` | portfolio release layer | execution gate / trade clock / operator | `F:\quant_data\Ashare\data\trade_release_v1\latest_release.json` | JSON | pointer to the current formal release |
| `release_manifest.json` | portfolio release layer | execution gate / operator | `F:\quant_data\Ashare\data\trade_release_v1\releases\<release_id>\release_manifest.json` | JSON | versioned trade-date-scoped release contract between research and execution; now also snapshots market-state truth and technical-confirmation summary |
| `latest_actual_portfolio_state.json` | OMS runtime | operator / V2A continuity / audit | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\latest_actual_portfolio_state.json` | JSON | authoritative OMS actual-state snapshot derived from broker/account truth plus open intents/orders |
| `desired_vs_actual_gap.csv` | OMS runtime | operator / audit / control feedback | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\desired_vs_actual_gap.csv` | CSV | authoritative desired-vs-actual gap table between the published release book and broker truth |
| `oms_summary.json` | OMS runtime | operator / postmortem | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\oms_summary.json` | JSON | OMS-level summary over authority ownership, gap size, intent status, dispatch count, and overrides applied |
| `intent_ledger_latest.csv` | OMS runtime | operator / audit / replay | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\ledgers\intent_ledger_latest.csv` | CSV | authoritative intent ledger with status lifecycle from `planned` through terminal states |
| `order_ledger_latest.csv` | OMS runtime | operator / audit / replay | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\ledgers\order_ledger_latest.csv` | CSV | authoritative order ledger with broker ids, status, remaining quantity, and intent linkage |
| `fill_ledger_latest.csv` | OMS runtime | operator / audit / replay | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\ledgers\fill_ledger_latest.csv` | CSV | authoritative fill ledger keyed by broker execution ids when available |
| `actual_state_daily.csv` | OMS runtime | operator / replay / research continuity fallback | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\actual_state_daily.csv` | CSV | rolling actual-state history derived from broker truth, not from the research-side target book |
| latest_open_intents.json | OMS runtime | operator / resume logic / audit | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\latest_open_intents.json | JSON | latest unresolved open intents after final ledger deduplication; first-stop snapshot for cross-session continuity |
| latest_intent_continuity_report.json | OMS runtime | operator / OMS debugging | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\latest_intent_continuity_report.json | JSON | start-of-session continuity classification over carried intents, superseded intents, cancel requests, and reconcile-only blocks |
| session_resume_audit.json | OMS runtime | operator / recovery | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\session_resume_audit.json | JSON | bounded session-resume audit with ignored stale orders, carried symbols, replacement-required symbols, and cancel requests |
| cancel_replace_audit.json | OMS runtime | operator / postmortem | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\cancel_replace_audit.json | JSON | explicit cancel/replace lineage over cancel requests/results and old_intent_id -> new_intent_id replacements |
| latest_manual_intervention_state.json | OMS runtime | operator / audit | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\snapshots\latest_manual_intervention_state.json | JSON | latest applied OMS intervention state with active override summary, applied counts, and current override payload hash |
| `control_feedback_latest.json` | OMS runtime | V2A posture engine / operator | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\feedback\control_feedback_latest.json` | JSON | Bucket B feedback for new-entry/add completion, turnover truncation, and persistent gap pressure |
| gap_control_metrics_daily.csv | OMS runtime | V2A posture engine / operator | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\feedback\gap_control_metrics_daily.csv | CSV | rolling control-feedback history with completion ratios, convergence, replacement churn, and partial-stuck metrics |
| `research_meta_feedback_latest.json` | OMS runtime | context pack / research meta weighting / operator | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\feedback\research_meta_feedback_latest.json` | JSON | Bucket C aggregated execution-realism feedback for research-side consumption only |
| mechanism_realism_rollup.csv | OMS runtime | research context / operator | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\feedback\mechanism_realism_rollup.csv | CSV | rolling mechanism-level realizability and convergence rollup over 20/40/60-run windows |
| `narrative_feedback_latest.json` | OMS runtime | operator | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\feedback\narrative_feedback_latest.json` | JSON | Bucket D non-authoritative human-readable summary; never mutates truth |
| manual_overrides.json | OMS operator | OMS runtime / continuity / cancel-replace | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\manual_overrides.json | JSON | operator-facing OMS intervention file for intent-level, symbol-level, and session-level overrides; distinct from trade-clock safety overrides |
| manual_override_history.jsonl | OMS runtime | operator / audit | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\history\manual_override_history.jsonl | JSONL | append-only history of OMS override payload hashes and applied intervention summaries |
| oms_validation_report.json | OMS validation harness | operator / Codex regression checks | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\validation\oms_validation_report.json | JSON | bounded synthetic/replay validation report over reconciliation, continuity, lifecycle, and recovery scenarios |
| oms_validation_summary.md | OMS validation harness | operator / Codex regression checks | F:\quant_data\Ashare\data\live_execution_bridge\oms_v1\validation\oms_validation_summary.md | Markdown | short human-readable pass/fail summary for the OMS validation harness |
| `clock_state.json` | trade clock supervisor | operator / debugging | `F:\quant_data\Ashare\data\trade_clock\clock_state.json` | JSON | heartbeat, gate status, active window, and last dispatch state |
| `scheduler_runtime.json` | trade clock supervisor | operator / debugging | `F:\quant_data\Ashare\data\trade_clock\runtime\scheduler_runtime.json` | JSON | scheduler self-state with process pid, active phase, stop reason, and last update time |
| `phase_state\YYYYMMDD.json` | trade clock supervisor | operator / debugging / postmortem | `F:\quant_data\Ashare\data\trade_clock\phase_state\YYYYMMDD.json` | JSON | per-day phase truth for `research/release/preopen_gate/simulation/shadow/summary` with status, timestamps, release id, and stdout/stderr paths |
| `system_safety_state.json` | safety guard | operator / execution gate / trade clock | `F:\quant_data\Ashare\data\trade_clock\system_safety_state.json` | JSON | current execution safety truth including system mode, market regime, manual overrides, release validation, and freshness markers |
| `incident_log.jsonl` | safety guard | operator / debugging / postmortem | `F:\quant_data\Ashare\data\trade_clock\incident_log.jsonl` | JSONL | append-only abnormal-event log with before/after safety modes and action taken |
| `manual_overrides.json` | operator + safety guard | trade clock / execution gate | `F:\quant_data\Ashare\data\trade_clock\manual_overrides.json` | JSON | operator-editable `manual_halt` / `manual_reduce_only` kill-switch file |
| `manual_override_history.jsonl` | safety guard | operator / postmortem | `F:\quant_data\Ashare\data\trade_clock\manual_override_history.jsonl` | JSONL | audit trail when manual override values change |
| `latest_account_health.json` | gmtrade health probe sidecar | safety guard / operator | `F:\quant_data\Ashare\data\trade_clock\latest_account_health.json` | JSON | latest fresh or cached account/position/order health snapshot fetched via `gmtrade39` |
| `latest_execution_dispatch.json` | execution gate / trade clock supervisor | operator / debugging | `F:\quant_data\Ashare\data\trade_clock\latest_execution_dispatch.json` | JSON | latest release-triggered execution dispatch outcome |
| `latest_execution_dispatch.<namespace>.json` | execution gate / trade clock supervisor | operator / debugging | `F:\quant_data\Ashare\data\trade_clock\latest_execution_dispatch.<namespace>.json` | JSON | namespace-isolated dispatch result for simulation or shadow execution lines |
| `outputs\automation_runs\YYYYMMDD\run_manifest.json` | trade clock summary packager | operator / debugging / daily audit | `F:\quant_data\Ashare\outputs\automation_runs\YYYYMMDD\run_manifest.json` | JSON | per-day automation bundle manifest with release id, namespaces, report path, and copied phase artifacts |
| `outputs\automation_runs\YYYYMMDD\daily_report.txt` | trade clock summary packager | operator | `F:\quant_data\Ashare\outputs\automation_runs\YYYYMMDD\daily_report.txt` | Text | plain-language daily automation summary over phase results, release, OMS sidecars, warnings, and critical flags |
| `execution_report_*.json` | Gmtrade execution bridge | operator / supervisor feedback | `F:\quant_data\Ashare\data\live_execution_bridge\execution_report_*.json` | JSON | execution summary per run |
| `latest_account_state.json` | live execution compatibility writer | operator / legacy downstream readers | `F:\quant_data\Ashare\data\live_execution_bridge\latest_account_state.json` | JSON | compatibility-only latest account snapshot; OMS ledgers and `latest_actual_portfolio_state.json` are now authoritative |
| `position_state_before.json` | portfolio control V1 | operator / audit | `F:\quant_data\Ashare\data\live_execution_bridge\portfolio_control_runs\<timestamp>\position_state_before.json` | JSON | planned-trade ledger snapshot before controls are applied |
| `position_state_after_plan.json` | portfolio control V1 | operator / audit | `F:\quant_data\Ashare\data\live_execution_bridge\portfolio_control_runs\<timestamp>\position_state_after_plan.json` | JSON | target vs actual vs pending plan after drift/budget controls |
| `position_state_after_execution.json` | portfolio control V1 | operator / audit | `F:\quant_data\Ashare\data\live_execution_bridge\portfolio_control_runs\<timestamp>\position_state_after_execution.json` | JSON | actual vs target vs unfinished-order pending effect after execution |
| `rebalance_audit.json` | portfolio control V1 | operator / audit | `F:\quant_data\Ashare\data\live_execution_bridge\portfolio_control_runs\<timestamp>\rebalance_audit.json` | JSON | explains drift skips, turnover truncation, and final control decisions |
| `execution_feedback.json` | portfolio control V1 | operator / audit | `F:\quant_data\Ashare\data\live_execution_bridge\portfolio_control_runs\<timestamp>\execution_feedback.json` | JSON | normalizes planned/submitted/filled/skipped order outcomes |
| `equity_curve.csv` | Gmtrade execution bridge | supervisor feedback | `F:\quant_data\Ashare\data\live_execution_bridge\equity_curve.csv` | CSV | feeds daily performance feedback |
| `performance_feedback.json` | supervisor | candidate factory / portfolio recommendation / operator | `F:\quant_data\Ashare\data\event_lake_v6\bridge\performance_feedback.json` | JSON | next-day regime and route bias feedback |

## Config Surface
| Config | Location | Current | Impact |
| --- | --- | --- | --- |
| `DEFAULT_RUN_PROFILE` | `hub_v6/local_settings.py` | `quick_test` | changes which runtime profile is used when no `--profile` is passed |
| `TOKEN_PLAN_MIN_INTERVAL_HOURS` | `hub_v6/local_settings.py` | `24` | controls V6 research-plan reuse frequency |
| `OVERNIGHT_V5_GPU_MAX_CYCLES_PER_TICK` | `hub_v6/local_settings.py` | `8` | controls overnight runtime and research depth |
| `QUICK_TEST_V5_GPU_MAX_CYCLES_PER_TICK` | `hub_v6/local_settings.py` | `1` | controls quick_test runtime and debugging speed |
| `ENABLE_EXECUTION_BRIDGE` | `hub_v6/local_settings.py` | `True` | determines whether simulated execution runs after portfolio generation |
| `PORTFOLIO_ENABLE_POST_FILTER_REWEIGHT` | `hub_v6/local_settings.py` | `True` | allows filtered target weights to be re-expanded toward a meaningful exposure floor when regime capacity still exists |
| `PORTFOLIO_MIN_EXPOSURE_FILL_RATIO` | `hub_v6/local_settings.py` | `0.75` | target fraction of the current total-exposure cap used by post-filter reweighting when the filtered book is too sparse |
| `ENABLE_PORTFOLIO_V2A` | `hub_v6/local_settings.py` | `True` | enables the deterministic V2A posture/lifecycle/admission engine inside portfolio recommendation |
| `PORTFOLIO_ENABLE_LIFECYCLE_STATE_MACHINE` | `hub_v6/local_settings.py` | `True` | enables lifecycle-state assignment before final target-book publication |
| `PORTFOLIO_ENABLE_ADMISSION_REPLACEMENT` | `hub_v6/local_settings.py` | `True` | enables new-entry admission and weak-incumbent replacement logic when slots are constrained |
| `PORTFOLIO_ENABLE_SOFT_CROWDING_PENALTY` | `hub_v6/local_settings.py` | `True` | enables soft crowding penalties as a ranking/weight modifier instead of hard blocking |
| `PORTFOLIO_ENABLE_RICH_PORTFOLIO_AUDIT` | `hub_v6/local_settings.py` | `True` | enables richer V2A sidecars and release/audit metadata |
| `PORTFOLIO_V2A_PILOT_MAX_WEIGHT` | `hub_v6/local_settings.py` | `0.04` | maximum default size for `pilot` positions before post-filter reweight and single-name caps are applied |
| `PORTFOLIO_V2A_BUILD_SPEED` / `PORTFOLIO_V2A_TRIM_SPEED` | `hub_v6/local_settings.py` | `1.25 / 0.72` | controls how quickly V2A expands strong incumbents and trims weakening names |
| `PORTFOLIO_V2A_REPLACEMENT_IMPROVEMENT_THRESHOLD` | `hub_v6/local_settings.py` | `0.08` | minimum admission-vs-retention improvement needed before replacing a weaker incumbent |
| `PORTFOLIO_V2A_SOFT_CROWDING_PENALTY_STRENGTH` | `hub_v6/local_settings.py` | `0.08` | strength of soft crowding penalties inside V2A admission/retention scoring |
| `ENABLE_PORTFOLIO_CONTROL` | `hub_v6/local_settings.py` | `True` | enables the low-risk portfolio control audit and constraint layer inside execution |
| `PORTFOLIO_CONTROL_DRIFT_THRESHOLD` | `hub_v6/local_settings.py` | `0.005` | small weight gaps below this threshold are skipped instead of traded |
| `PORTFOLIO_CONTROL_MAX_DAILY_TURNOVER_RATIO` | `hub_v6/local_settings.py` | `0.25` | caps planned daily turnover and truncates lower-priority orders when exceeded |
| `PORTFOLIO_CONTROL_ENABLE_EXECUTION_FEEDBACK` | `hub_v6/local_settings.py` | `True` | writes normalized planned/submitted/filled/skipped execution feedback |
| `PORTFOLIO_CONTROL_ENABLE_DEV_LOG_SNAPSHOT` | `hub_v6/local_settings.py` | `True` | refreshes the live portfolio snapshot block inside `CODEX_DEV_LOG.md` after execution |
| `PORTFOLIO_CONTROL_DEV_LOG_TOP_HOLDINGS` | `hub_v6/local_settings.py` | `8` | controls how many top holdings are written into the dev-log snapshot |
| `PORTFOLIO_CONTROL_ALLOW_ODD_LOT_EXIT` | `hub_v6/local_settings.py` | `True` | allows cleanup of residual odd-lot sell quantities in the control layer |
| `ENABLE_OMS` | `hub_v6/local_settings.py` | `True` | turns the broker-truth-first OMS layer on beneath `execution_only` and the gmtrade bridge |
| `OMS_OUTPUT_ROOT` | `hub_v6/local_settings.py` | `F:\quant_data\Ashare\data\live_execution_bridge\oms_v1` | authoritative OMS ledger/artifact root |
| `OMS_USE_BROKER_TRUTH_FOR_V2A_CONTINUITY` | `hub_v6/local_settings.py` | `True` | makes V2A prefer OMS actual-state truth over previous target/lifecycle sidecars when continuity is available |
| `OMS_INTENT_EXPIRY_DAYS` | `hub_v6/local_settings.py` | `3` | default expiry window for OMS intents before they should be treated as stale |
| `OMS_CONTROL_FEEDBACK_LOOKBACK_RUNS` / `OMS_RESEARCH_META_LOOKBACK_RUNS` | `hub_v6/local_settings.py` | `20 / 60` | lookback windows for Bucket B control feedback and Bucket C research meta feedback aggregation |
| `OMS_COMPAT_WRITE_LATEST_ACCOUNT_STATE` | `hub_v6/local_settings.py` | `True` | keeps writing `latest_account_state.json` for legacy consumers even though OMS artifacts are authoritative |
| OMS_ENABLE_BROKER_CANCEL | hub_v6/local_settings.py | True | allows OMS cancel/replace logic to request broker-side order_cancel when unresolved open orders are superseded or operator-cancelled |
| `EXECUTION_ACCOUNT_MODE` | `hub_v6/local_settings.py` | `precision` | selects which gmtrade account profile is active by default: `simulation` or `precision` |
| `PRECISION_TRADE_ENABLED` | `hub_v6/local_settings.py` | `False` | when `False`, precision mode still refreshes heartbeat/gate logs but refuses to call the execution bridge |
| `ALLOW_INTEGRATED_PRECISION_EXECUTION` | `hub_v6/local_settings.py` | `False` | keeps `integrated_supervisor` and `resume_downstream` from directly executing against the precision account unless explicitly allowed |
| `ENABLE_TRADE_RELEASE` | `hub_v6/local_settings.py` | `True` | turns the middle release layer on after research-side portfolio generation |
| `TRADE_RELEASE_VALID_AFTER_TIME` / `TRADE_RELEASE_EXPIRES_AT_TIME` | `hub_v6/local_settings.py` | `09:30:30 / 15:00:00` | defines the default release validity window consumed by `execution_only` |
| `ENABLE_TRADE_CLOCK` | `hub_v6/local_settings.py` | `True` | enables the lightweight always-on clock-service path |
| `ENABLE_SAFETY_LAYER` | `hub_v6/local_settings.py` | `True` | enables fail-closed safety evaluation, manual overrides, incident logging, and gmtrade health probes on the execution side |
| `SAFETY_HEALTH_PROBE_INTERVAL_SECONDS` | `hub_v6/local_settings.py` | `300` | controls how often the clock may refresh the broker/account health snapshot via the dedicated `gmtrade39` probe |
| `SAFETY_ACCOUNT_STATE_MAX_AGE_SECONDS` / `SAFETY_POSITION_SYNC_MAX_AGE_SECONDS` | `hub_v6/local_settings.py` | `900 / 900` | if account/position truth is older than this, execution flips to `HALT` instead of guessing |
| `SAFETY_FAIL_ON_UNFINISHED_ORDERS` / `SAFETY_FAIL_ON_UNKNOWN_ORDER_STATUS` | `hub_v6/local_settings.py` | `True / True` | blocks new execution when broker order truth is incomplete or not understood |
| `SAFETY_CAUTION_*` / `SAFETY_PANIC_*` | `hub_v6/local_settings.py` | current local values | market-safety thresholds for broad-selloff detection from daily snapshot + HS300 |
| `SAFETY_EXECUTION_FAIL_RATIO_DEGRADED` / `SAFETY_EXECUTION_FAIL_RATIO_HALT` | `hub_v6/local_settings.py` | `0.35 / 0.75` | recent execution failure ratios above these thresholds degrade or halt execution-side behavior |
| `INDUSTRY_ROUTER_ENABLE_SOURCE_FETCH` | `hub_v6/local_settings.py` | `True` | turns the official-source sidecar on for the industry router |
| `INDUSTRY_ROUTER_SOURCE_FETCH_TIMEOUT_SECONDS` | `hub_v6/local_settings.py` | `8` | caps per-source official-page fetch time so the sidecar stays lightweight |
| `INDUSTRY_ROUTER_SOURCE_FETCH_CACHE_HOURS` | `hub_v6/local_settings.py` | `12` | controls reuse of cached official-source snapshots under `source_snapshots` |
| `INDUSTRY_ROUTER_SOURCE_FETCH_MAX_SOURCES_PER_RUN` | `hub_v6/local_settings.py` | `9` | caps the number of official sources fetched in a single `industry_router` run |
| `TRADE_CLOCK_POLL_SECONDS` | `hub_v6/local_settings.py` | `30` | controls the sleeping heartbeat interval of the clock supervisor |
| `TRADE_CLOCK_EXECUTION_WINDOWS` | `hub_v6/local_settings.py` | `[{label=morning_primary,start=09:30:30,end=10:00:00}]` | defines the only windows in which `execution_only` may auto-dispatch |
| `manual_halt` / `manual_reduce_only` | `data\\trade_clock\\manual_overrides.json` | `False / False` | operator-facing runtime kill switches; `manual_halt` blocks all new orders, `manual_reduce_only` keeps the bridge sell-only |
| `ENABLE_INDUSTRY_ROUTER` | `hub_v6/local_settings.py` | `True` | turns the stock/mechanism skeleton and split-backtest sidecar on inside V6 |
| `INDUSTRY_ROUTER_CONTRACT_ROOT` / `INDUSTRY_ROUTER_OUTPUT_ROOT` | `hub_v6/local_settings.py` | `...\configs\industry_router / data\event_lake_v6\research\industry_router` | separates static contracts from runtime artifacts |
| `INDUSTRY_ROUTER_HISTORY_LOOKBACK_DAYS` | `hub_v6/local_settings.py` | `14` | controls how much event-store history is pulled into the router |
| `INDUSTRY_ROUTER_ENABLE_BACKTEST` | `hub_v6/local_settings.py` | `True` | enables the split mechanism backtest skeleton after signal generation |
| `INDUSTRY_ROUTER_BACKTEST_HORIZONS` / `INDUSTRY_ROUTER_BACKTEST_TOP_K` | `hub_v6/local_settings.py` | `[1, 2] / 3` | controls the minimal forward-return horizons and per-day top-k used by the split backtest |
| `INDUSTRY_ROUTER_ENABLE_CONTEXT_PACK` | `hub_v6/local_settings.py` | `True` | allows the router summary to be injected into `research_context_pack.json` |
| `ENABLE_MARKET_STATE_ENGINE` | `hub_v6/local_settings.py` | `True` | turns the formal market-state regime layer on inside V6 planning and downstream portfolio/release/execution consumers |
| `MARKET_STATE_USE_ROUTER_BIAS` | `hub_v6/local_settings.py` | `True` | allows mechanism leadership from the industry-router output to influence market-state mechanism bias |
| `MARKET_STATE_ROOT` | `hub_v6/local_settings.py` | `F:\quant_data\Ashare\data\market_state_v6` | controls where market-state artifacts are written and read |
| `MARKET_STATE_CONFIG_PATH` | `hub_v6/local_settings.py` | `...\configs\market_state\default.json` | policy thresholds and regime-band definitions for the market-state engine |
| `ENABLE_TECHNICAL_CONFIRMATION` | `hub_v6/local_settings.py` | `True` | turns the formal candidate-level technical gate on during portfolio recommendation |
| `PORTFOLIO_MARKET_STATE_AWARE_SIZING` | `hub_v6/local_settings.py` | `True` | allows market-state truth to tighten portfolio exposure, name count, and single-name caps |
| `PORTFOLIO_TECHNICAL_CONFIRMATION_GATE` | `hub_v6/local_settings.py` | `True` | allows technical confirmation to reject or down-weight new candidates before target positions are written |
| `TECHNICAL_CONFIRMATION_ROOT` | `hub_v6/local_settings.py` | `F:\quant_data\Ashare\data\event_lake_v6\research\technical_confirmation` | controls where technical-confirmation artifacts are written and read |
| `TECHNICAL_CONFIRMATION_CONFIG_PATH` | `hub_v6/local_settings.py` | `...\configs\technical_confirmation\default.json` | scoring thresholds and gating rules for the technical-confirmation layer |
| `ENABLE_DAILY_STRATEGY_FEEDBACK` | `hub_v6/local_settings.py` | `True` | determines whether prior-day performance changes route and portfolio posture |
| `ENABLE_TUSHARE_NEWS` / `ENABLE_TUSHARE_MAJOR_NEWS` | `hub_v6/local_settings.py` | `True / True` | enables Tushare message-layer inputs |
| `TUSHARE_NEWS_MAX_SOURCES_PER_RUN` | `hub_v6/local_settings.py` | `1` | affects short-news breadth vs quota safety |
| `TUSHARE_MAJOR_NEWS_MAX_SOURCES_PER_RUN` | `hub_v6/local_settings.py` | `3` | affects major-news breadth vs quota safety |
| `OLLAMA_EVENT_EXTRACT_MODEL` / `OLLAMA_RESEARCH_MODEL` | `hub_v6/local_settings.py` | `qwen2.5:7b / deepseek-r1:14b` | splits strict event extraction from local research fallback |
| `OLLAMA_EVIDENCE_CARD_*` | `hub_v6/local_settings.py` | `enabled / deepseek-r1:14b / 180s / 2 items` | controls additive announcement evidence-card generation |
| `OLLAMA_REVIEW_ROUTER_*` | `hub_v6/local_settings.py` | `enabled / qwen2.5:7b / 90s / 6 items` | controls additive manual-review queue routing |
| `OLLAMA_RUNTIME_EXPLAINER_*` | `hub_v6/local_settings.py` | `enabled / qwen2.5:7b / 45s` | controls long-stage operator notes and watch-file hints |
| `OLLAMA_V5_REVIEW_*` | `hub_v6/local_settings.py` | `enabled / deepseek-r1:14b / 180s` | controls additive V5 post-cycle local review |
| `V5_PROJECT_ROOT` | `hub_v6/local_settings.py` | `F:\quant_data\Ashare\quant_research_hub_v5_1_gpu_integrated` | legacy-named metadata field passed into V5 config; not the actual script launch path |
| `V5_HUB_OUTPUT_ROOT` | `hub_v6/local_settings.py` | `F:\quant_data\Ashare\data\research_hub_v5_1_gpu_integrated` | current V5 output root consumed by registry, cycle summaries, and portfolio recommendation |

## Known Issues
- OpenAI upstream network resets can still happen occasionally; the client now retries transient failures and auto-drops unsupported `reasoning.effort`.
- Tushare news can still return zero rows when upstream quota is exhausted even after local quota guarding.
- V5.1 runtime exposes sparse heartbeat artifacts while a cycle is running; operators often need to infer progress from candidate file timestamps.
- The new local evidence-card, review-router, runtime-explainer, and V5-review layers are additive sidecars; if local Ollama is unavailable, they now fail fast and their artifacts can be empty or stale without blocking the main chain.
- The deepened three-mechanism router still uses a hand-curated seed pool, not a full-market automatic stock master.
- The current event taxonomy and mapping rules are intentionally conservative and still vulnerable to routing edge cases; the new three-mechanism modules are structurally complete but not yet production-grade in coverage.
- `industry_router_only` currently rebuilds from `event_store.jsonl` history; if that history is too short or too noisy, split-backtest results will be sparse or unstable.
- The unified split backtest engine now respects mechanism policy hooks, but it still enters on the first trading close on or after the signal date and remains a research explainer, not a production execution simulation.
- Attribution output now preserves zero-trade candidate buckets, but when the latest signal dates are at the end of the available enriched-daily history the realized-trade rows can still be empty even though candidate rows exist.
- The official-source layer currently relies on curated fixed article/index URLs in `source_contracts.json`, not a rolling agency-discovery crawler; freshness is partially mitigated by decay weights.
- The source fetcher now falls back to an unverified SSL context for a subset of official China government sites because this environment rejects some certificate chains; keep the source list restricted to explicit official domains and do not broaden it casually.
- `main_research_runner.py` should be launched with the canonical research Python. If it is invoked with a bare shell `python` from the wrong interpreter, imports such as `requests` can fail even though the code is fine.
- `research_only` in `quick_test` can still exceed a 15-20 minute terminal window because V5 codegen / workspace validation dominates runtime; if new files keep appearing under `data\research_hub_v5_1_gpu_integrated\cycles\<cycle_id>\`, treat it as slow-progress rather than an immediate hang.
- Shared generated runtime config paths can still be touched by other live processes. Atomic writes now reduce truncation risk, but a concurrent holder on `hub_config.v6.runtime.<profile>.json` can still produce `PermissionError` during replacement if another run or editor keeps the file open.
- Portfolio control V1 still exists as a compatibility dispatch/audit utility, but it is no longer the authoritative truth owner for actual state or ledgers; OMS now owns those truths.
- The dev-log live portfolio snapshot is refreshed only by execution runs; if execution is disabled or skipped, that section can lag behind the latest research-side target portfolio.
- `trade_clock_service.py` is implemented as a user-session process plus a Windows logon task, not a native Windows service.
- The trading-day check depends on the cached `trading_calendar_a_share.csv` file plus Tushare refresh; if both are unavailable, the clock gate will block rather than guess a holiday schedule.
- There is no code-level guarantee against third-party security software terminating the clock process; the current mitigation is low resource usage plus manual restart scripts and clear runtime logs.
- In the current Codex / IDE environment, direct complex foreground Python invocations can occasionally fail with `ACP process` initialization errors or exit code `0xC000013A`; for operator automation and in-session validation, prefer the PowerShell start/stop scripts or `Start-Process` with stdout/stderr redirected to files.
- The safety layer is intentionally fail-closed. Recent execution failures, stale account truth, unfinished broker orders, or release validation failures can push `system_safety_state.json` to `HALT` and block new execution until the operator intervenes.
- The initial market-safety thresholds are deliberately conservative and can classify a broad market selloff day as `PANIC`; treat them as operational guardrails, not a final market-timing model.
- The active precision/simulation account mapping currently lives in `configs\gmtrade_runtime_config.local.json`; if that file is changed manually, make sure the `account_profiles` block stays aligned with `EXECUTION_ACCOUNT_MODE`.
- `hub_v6/local_settings.py` still contains legacy V5 naming such as `V5_PROJECT_ROOT`, which can mislead readers into thinking a root-level package is launched directly.
- The actual V5 launcher path is package-local `...\v5_gpu_runtime\run_research_hub_v5_1_local.py`; treat `project_root` inside V5 JSON as required config metadata, not launch-path truth.
- `deepseek-r1:14b` is currently configured only as a local research fallback model, not as the default title-extraction worker; strict event JSON extraction still stays on `qwen2.5:7b` until dedicated validation proves otherwise.
- Market-state and technical-confirmation defaults are now less defensive than the first rollout, but they are still posture controls, not a finished alpha model.
- The new `pilot` entry path and post-filter reweight can restore opportunity set and exposure, but they also make the system more sensitive to upstream candidate-quality errors; do not mistake higher capital usage for validated edge.
- Portfolio V2A now prefers OMS `latest_actual_portfolio_state.json` for continuity, but it still falls back to the previous target book plus previous lifecycle sidecar when OMS truth is absent or stale.
- `hub_v6/oms/runtime.py` is execution-environment code and imports the gmtrade stack through the broker adapter; use the lightweight OMS state-reader interfaces or `oms_validate` from the research Python, and keep full OMS runtime execution in `gmtrade39`.
- Portfolio V2A currently uses lightweight admission/replacement logic and soft crowding, not a full exposure/risk model; it is intentionally more capable than V1 but still not a production-grade optimizer or OMS.
- OMS now has deterministic cross-session intent continuity, formal manual intervention layering, first-pass broker-aware cancel/replace, and a bounded synthetic/replay validation harness; it is still not a full broker-agnostic intraday EMS.
- There are now two override files with different powers: data\trade_clock\manual_overrides.json owns global kill-switch behavior, while data\live_execution_bridge\oms_v1\manual_overrides.json owns OMS-local continuity/cancel/repair behavior; do not mix them.
- oms_validate proves OMS state transitions and replay continuity, not live broker connectivity or market-session correctness.
- In strongly defensive posture, V2A can still legitimately produce `new_entry_count = 0` if market-state is weak and the candidate set is poor. That is not automatically a bug; it must be judged against the candidate universe quality.
- `runtime_stage_notes.json` only covers selected long stages by default (`v6_planning`, `v5_gpu`, `portfolio_recommendation`, `execution_bridge`), not every short stage.

## Deferred Work
- Improve multi-source news coverage beyond Tushare quota limits.
- Refine entity linking between announcements and news items.
- Add message-derived features deeper into downstream data construction if they are not already present.

## Decision Log
- Decision: `launch_canonical.py` is the formal operator entry while `main_research_runner.py` remains the wrapped business root.
  - Reason: the project needed a single operator-facing entry without rewriting the existing business chain.
  - Alternatives considered: keep `main_research_runner.py` as both operator entry and business root, or refactor the core call chain directly.
  - Consequence: future formal runs should default to the wrapper, while code-level debugging can still inspect the wrapped root directly.
- Decision: every formal wrapper run should emit one `run_manifest.json` under `outputs\canonical_runs`.
  - Reason: the project needed a minimal, non-invasive trace artifact for operator audit and rollback-friendly run identification.
  - Alternatives considered: store no wrapper-level trace at all, or push run registration into the business chain itself.
  - Consequence: `launch_canonical.py` now owns run-level trace registration while the business chain remains unchanged.
- Decision: `main_research_runner.py` is the canonical root entry.
  - Reason: multiple legacy V6 entrypoints were diverging; one root entry is easier to operate.
  - Alternatives considered: keep `run_v6_full_cycle_real.py` as primary.
  - Consequence: old V6 runtime readmes are now historical.
- Decision: documentation must reflect the current code default profile as `quick_test`.
  - Reason: `hub_v6/local_settings.py` currently sets `DEFAULT_RUN_PROFILE = "quick_test"`; handoff notes must track code truth, not stale intent.
  - Alternatives considered: keep the docs saying `overnight` because that was the earlier operator preference.
  - Consequence: operators must pass `--profile overnight` explicitly when they want the heavy nightly path.
- Decision: keep dual Python environments.
  - Reason: Gmtrade does not support the main Python runtime used by the research stack.
  - Alternatives considered: unify all runtimes under one interpreter.
  - Consequence: execution-bridge environment must be protected from accidental switching.
- Decision: the archived root-level `quant_research_hub_v5*` directories are now historical only; the live research brain is the embedded `v5_gpu_runtime`.
  - Reason: `main_research_runner.py` and `hub_v6/supervisor.py` launch the package-local V5 runtime directly.
  - Alternatives considered: treat the archived root-level copy as still active.
  - Consequence: future debugging must inspect `v5_gpu_runtime` inside the active package tree rather than the archived root packages.
- Decision: message-layer logic now uses evidence quality and anti-overfit weighting instead of raw title importance alone.
  - Reason: title-driven signals were too easy to overfit.
  - Alternatives considered: keep simple rule score or hardcode more keywords.
  - Consequence: context pack and research brief now rely on quality-weighted message evidence.
- Decision: downstream recovery should be a first-class mode instead of a manual one-off script.
  - Reason: V5 can finish successfully while portfolio recommendation or execution fails later; rerunning the whole chain is wasteful.
  - Alternatives considered: keep manual ad hoc Python snippets for recovery.
  - Consequence: `resume_downstream` is now an operator-facing mode in `main_research_runner.py`.
- Decision: precise-style operation is now split into `research_only -> release layer -> execution_only`.
  - Reason: the integrated chain is too heavy and too timing-sensitive to be the formal precise-trading path.
  - Alternatives considered: keep one giant integrated supervisor as both research engine and time-gated execution manager.
  - Consequence: future precise trading should treat portfolio releases as the contract boundary between research and execution.
- Decision: OMS, not portfolio control or release, now owns `actual_state` and desired-vs-actual truth.
  - Reason: broker/account/order/fill truth must come from the execution-side reconciliation layer, not from research-side sidecars or release snapshots.
  - Alternatives considered: keep extending `portfolio_control.py` into a hidden pseudo-OMS, or continue letting V2A continuity infer actual state from prior target books.
  - Consequence: `hub_v6/oms/*` is now the canonical truth owner for actual holdings, intent/order/fill lifecycle, and actual-state artifacts.
- Decision: feedback is explicitly bucketed by authority level instead of flowing as one undifferentiated execution blob.
  - Reason: raw broker/account outcomes should not directly rewrite research logic, while portfolio pacing still needs operational feedback.
  - Alternatives considered: keep sending only `performance_feedback.json`, or let execution-side artifacts be consumed ad hoc by whichever module wants them.
  - Consequence: OMS now emits:
    - Bucket A truth feedback in ledgers/artifacts only
    - Bucket B control feedback for V2A posture pacing
    - Bucket C research meta feedback for aggregated research-side consumption
    - Bucket D narrative feedback for human reading only
- Decision: execution-side abnormal handling is now fail-closed and centered on one safety truth file plus one incident log.
    - Reason: scattered heartbeat/gate prints were not enough to answer whether the system should still trade after broker, file, or market anomalies.
    - Alternatives considered: keep only `clock_state.json`, or add multiple extra daemons for separate health checking.
    - Consequence: `execution_only` and `trade_clock_service.py` now read/write `system_safety_state.json`, `incident_log.jsonl`, and `manual_overrides.json`; one lightweight clock process remains the main resident service.
  - Decision: `industry_router` now uses one unified engine plus three mechanism policy modules instead of one monolithic runtime script.
    - Reason: the project needed a clean contract boundary between normalized inputs, mechanism state, stock profile, core variables, and signal policy so later mechanism refinement does not keep growing one file.
    - Alternatives considered: continue patching the old `runtime.py`, or bolt on extra sidecars while keeping the old runtime as the real engine.
    - Consequence: `contracts\`, `core\`, and `mechanisms\` are now the only formal implementation path; legacy top-level `contracts.py` and `source_ingest.py` were removed.
- Decision: the deepened three-mechanism architecture keeps one shared runtime/backtest engine, but pushes profile/state/mapping/signal/risk/policy logic fully down into mechanism packages.
  - Reason: the project needed to stop growing compatibility sludge around a single router file while still avoiding three separate backtest runtimes.
  - Alternatives considered: keep generic scoring and only add more thresholds, or fork three separate end-to-end routers.
  - Consequence: future mechanism refinement should happen inside `mechanisms\<name>\*` or static contracts, while `core\runtime_engine.py` and `core\backtest_engine.py` should remain mechanism-agnostic.
- Decision: market-state and technical-confirmation are now formal shared contracts, not optional post-hoc execution filters.
  - Reason: regime posture and entry confirmation need to influence research context, portfolio sizing, release truth, and execution behavior consistently instead of being recomputed ad hoc at one downstream point.
  - Alternatives considered: keep market-state only inside safety/clock logic, or add technical confirmation only as a local scoring sidecar that does not affect target positions.
  - Consequence: `market_state` and `technical_confirmation` artifacts are now first-class runtime outputs consumed by `context_pack`, `portfolio_recommendation`, `portfolio_release`, and `execution_manager`.
- Decision: Portfolio V2A is implemented as a deterministic sublayer inside portfolio recommendation, not as a parallel portfolio system and not as a heavy optimizer.
  - Reason: the system needed posture, lifecycle, staged sizing, and admission/replacement behavior without breaking the current research/release/execution law or introducing another source of truth.
  - Alternatives considered: keep only flat market-aware weights plus technical gate, or jump straight to a full optimizer / OMS rewrite.
  - Consequence: `hub_v6\portfolio_v2a` now owns research-side posture/state/admission logic, while `target_positions.csv`, release manifests, and execution audit remain the canonical downstream carriers.

## What To Inspect After A Run
- Supervisor state:
  - `F:\quant_data\Ashare\data\event_lake_v6\research\supervisor\supervisor_state.json`
  - inspect `current_stage` while a run is active
  - inspect `stages` and `stage_history` to see which phase completed, failed, or was skipped
- Market pipeline report:
  - `F:\quant_data\Ashare\data\daily_cache_v6\market_pipeline_report.json`
- Research brief:
  - `F:\quant_data\Ashare\data\event_lake_v6\research\briefs\research_brief.json`
- Portfolio outputs:
  - `F:\quant_data\Ashare\data\portfolio_recommendation_v6\portfolio_recommendation.json`
  - `F:\quant_data\Ashare\data\portfolio_recommendation_v6\target_positions.csv`
  - `F:\quant_data\Ashare\data\portfolio_recommendation_v6\rebalance_orders.csv`
- Execution outputs:
  - `F:\quant_data\Ashare\data\live_execution_bridge\latest_account_state.json`
  - `F:\quant_data\Ashare\data\live_execution_bridge\equity_curve.csv`
  - `F:\quant_data\Ashare\data\live_execution_bridge\execution_report_*.json`
- Dynamic strategy feedback:
  - `F:\quant_data\Ashare\data\event_lake_v6\bridge\performance_feedback.json`

## Token And API Notes
- Tushare is wired into `hub_v6/local_settings.py` and passed through `hub_v6/config_builder.py`.
- OpenAI and DeepSeek are expected through environment variables:
  - `OPENAI_API_KEY`
  - `DEEPSEEK_API_KEY`
- Tushare rate limits matter in this project:
  - `news`: effectively 1 call per minute on the current plan
  - `major_news`: effectively 4 calls per hour on the current plan
- Tushare news fetch state is persisted at:
  - `F:\quant_data\Ashare\data\daily_cache_v6\tushare_fetch_state.json`

## README Integration Summary
### V5 and V5.1 README themes
- `quant_research_hub_v5` explains the original move away from the broken V4 cold-start loop.
- `quant_research_hub_v5_1` and `v5_gpu_runtime/README.md` define V5.1 as the GPU-aware automatic research controller with resource guardrails and candidate workspaces.
- `data\quant_research_hub_v5_1\quant_research_hub_v5_1_xgb_lock\README.md` documents the XGBoost-GPU-locked variant.

### V6 README themes
- `docs\README.md` and `docs\README_RUNTIME_PATCH.md` describe the original V6 announcement-driven runtime package.
- `docs\README_COST_PATCH.md` adds the lower-cost event-extract plus portfolio-recommendation framing.
- `docs\README_PYCHARM.md` says V6 was intended to be launched from PyCharm and configured from `hub_v6/local_settings.py`.
- `docs\README_接入说明.md` and the Gmtrade bridge readmes describe the execution-layer integration.

### Current truth after integration
- Older V6 readmes still point to `run_v6_full_cycle_real.py`. Treat that as historical documentation.
- The real root entry in the current integrated system is `main_research_runner.py`.
- V6 is now the orchestrator and supervisor layer.
- V5.1 remains the heavy research engine, but the live copy is the embedded `v5_gpu_runtime`, not the archived root-level `quant_research_hub_v5_1`.
- The execution bridge is now part of the main supervised flow, not a separate side package.
- Portfolio recommendation is a first-class output in the daily chain.

## Change Log Format
- Entries from `2026-03-21 14:49` onward should use this structure where practical:
  - `Timestamp`
  - `Type`
  - `Scope`
  - `Files`
  - `Change`
  - `Impact`
  - `Validation`
  - `Compatibility`
  - `Rollback`
- Older entries below are legacy short-form entries and may omit some fields.

## Change Log Entry Template
```md
### YYYY-MM-DD HH:MM
- Type:
  - `bugfix|feature|refactor|config|runtime|docs|ops`
- Scope:
  - `data|event|research|portfolio|execution|infra`
- Files:
  - `absolute\path\to\file`
- Change:
  - What changed.
- Impact:
  - Which downstream modules, outputs, or operator workflows are affected.
- Validation:
  - Lightweight checks performed.
  - Explicitly say if no full pipeline run was executed.
- Compatibility:
  - Backward compatible or not, plus any stale docs/configs/output-schema notes.
- Rollback:
  - How to revert or disable the change if needed.
```

## Change Log
All timestamps below are local file write times in the current workspace and should be read as Asia/Shanghai local time.

### 2026-03-23 03:05
- Type:
  - `feature`
  - `ops`
  - `bugfix`
- Scope:
  - `execution`
  - `infra`
  - `operator`
- Files:
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\launch_canonical.py`
  - `F:\quant_data\Ashare\trade_clock_service.py`
  - `F:\quant_data\Ashare\RUN_PROFILES.yaml`
  - `F:\quant_data\Ashare\SYSTEM_DAILY_USAGE_GUIDE_CN.txt`
  - `F:\quant_data\Ashare\scripts\start_trade_clock.ps1`
  - `F:\quant_data\Ashare\scripts\stop_trade_clock.ps1`
  - `F:\quant_data\Ashare\scripts\show_trade_clock_status.ps1`
  - `F:\quant_data\Ashare\scripts\install_trade_clock_autostart.ps1`
  - `F:\quant_data\Ashare\scripts\remove_trade_clock_autostart.ps1`
  - `F:\quant_data\Ashare\tools\preflight_check.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\runtime_profiles.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\clock_supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_utils.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_bridge_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_manager.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\oms\runtime.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Upgraded `trade_clock_service.py` plus `hub_v6\clock_supervisor.py` from a simple execution-window trigger into a manual-start daily scheduler with six formal phases:
    - `research`
    - `release`
    - `preopen_gate`
    - `simulation`
    - `shadow`
    - `summary`
  - Added the new formal runtime profile `daily_production` and fixed its bounded research weight to `V5 cycles = 3`.
  - Kept heavy work out of the resident process: the scheduler now launches all phase work through short-lived subprocesses, writes per-day phase state under `data\trade_clock\phase_state`, stores runtime/log state under `data\trade_clock\runtime`, and writes packaged daily operator bundles under `outputs\automation_runs\YYYYMMDD`.
  - Added simulation/shadow namespace isolation so both execution lines consume the same release but keep separate dispatch and OMS truth roots.
  - Added fallback-release support: `release_only` can now publish from explicit portfolio artifacts when `research_only` fails or times out, instead of collapsing the whole next-day chain.
  - Added BOM-tolerant config reads (`utf-8-sig`) so PowerShell-written temp configs and override payloads do not fail JSON decoding.
  - Reworked the operator surface:
    - `start_trade_clock.ps1` / `stop_trade_clock.ps1` are now the primary manual lifecycle scripts
    - old trade-clock autostart install/remove scripts are deprecated into legacy-cleanup helpers
    - new `show_trade_clock_status.ps1` gives a concise read-only summary over service state, release, safety, phase status, OMS sidecars, and daily pack location
    - `SYSTEM_DAILY_USAGE_GUIDE_CN.txt` now documents the automation chain, config surface, file meanings, and routine operator commands in plain language
  - Documented and worked around the current IDE/Codex ACP instability by favoring background subprocesses with file-based stdout/stderr instead of long foreground Python sessions for automation validation.
- Impact:
  - Stable-running test phase now has one manual-start scheduler path that can keep running without depending on IDE foreground control.
  - The daily automation chain is now explicit and inspectable even if one phase fails.
  - Operators can inspect status through:
    - `clock_state.json`
    - `scheduler_runtime.json`
    - `phase_state\YYYYMMDD.json`
    - `outputs\automation_runs\YYYYMMDD`
    without digging directly through raw subprocess output first.
  - Research and execution authority boundaries remain intact; this is a scheduling/operability upgrade, not a power-structure rollback.
- Validation:
  - Ran `python -m py_compile` on all touched Python files, including the scheduler, profile helper, config readers, execution namespace plumbing, and release fallback paths.
  - Ran `python F:\quant_data\Ashare\trade_clock_service.py --profile daily_production --once --skip-preflight`; confirmed `clock_state.json` now reports scheduler roots and `scheduler_profile=daily_production`.
  - Ran a synthetic automation probe through `Start-Process` with redirected logs; confirmed:
    - `daily_production` resolves to `V5 cycles = 3`
    - Windows lock detection prevents double acquisition
    - fallback source discovery returns a usable source
    - daily packager writes `run_manifest.json`, `phase_status.json`, and `daily_report.txt`
  - Ran a namespace-isolation probe and confirmed simulation/shadow output roots and OMS roots differ.
  - Ran a fallback-release probe with explicit summary/target inputs; after the BOM fix, `release_only` successfully published a release with `source_mode=fallback_release`.
  - Ran `powershell -ExecutionPolicy Bypass -File F:\quant_data\Ashare\scripts\start_trade_clock.ps1 -Profile daily_production` and `stop_trade_clock.ps1`; confirmed graceful stop after the idle-sleep interrupt fix.
  - Ran `powershell -ExecutionPolicy Bypass -File F:\quant_data\Ashare\scripts\show_trade_clock_status.ps1` and confirmed it renders current service, release, safety, phase, OMS, and daily-pack paths.
  - No full integrated pipeline and no real broker execution were run.
- Compatibility:
  - Additive at the operator/protocol level.
  - Existing `quick_test` and `overnight` profiles remain valid.
  - Existing `execution_only`, `research_only`, and `release_only` interfaces remain valid, but `release_only` now supports explicit fallback-source inputs.
  - Trade-clock autostart is no longer the recommended operating model; legacy cleanup scripts remain only to remove older scheduled tasks.
- Rollback:
  - Revert `trade_clock_service.py`, `hub_v6\clock_supervisor.py`, and the new profile/helper files to restore the older clock behavior.
  - Remove `show_trade_clock_status.ps1` and restore the earlier operator guide if the new manual-scheduler operating model is abandoned.
  - If only the fallback or BOM handling needs rollback, revert the corresponding changes in `supervisor.py`, `config_utils.py`, `main_research_runner.py`, and `launch_canonical.py`.

### 2026-03-22 18:47
- Type:
  - `refactor`
- Scope:
  - `research`
  - `infra`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\runtime.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\backtest.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\contracts\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\core\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\mechanisms\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\industry_router\event_taxonomy.json`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\industry_router\source_contracts.json`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Rebuilt `industry_router` into a unified three-mechanism architecture.
  - Added formal contract modules, shared runtime/backtest engines, and separate policy modules for `trend_capex`, `price_inventory`, and `macro_style`.
  - Collapsed the old monolithic runtime into orchestration-only wrappers and removed legacy top-level `contracts.py` / `source_ingest.py`.
  - Promoted `stock_profile.csv`, `mechanism_state_daily.csv`, and `core_variable_daily.csv` to formal runtime artifacts.
  - Added stale-artifact cleanup so legacy `industry_state_daily.csv` and obsolete backtest detail files do not remain in the output directory after a rebuild.
- Impact:
  - `industry_router_only` now runs through one shared engine with mechanism-specific policies instead of ad hoc branching inside one file.
  - Context-pack consumers keep the same high-level payload shape, but the underlying research artifacts are now contract-driven and easier to extend by mechanism.
  - Output schema changed on the research side: `mechanism_state_daily.csv` replaces the old `industry_state_daily.csv` as the formal state table.
- Validation:
  - Ran `python -m py_compile` on the new `industry_router` subtree.
  - Ran `python F:\quant_data\Ashare\tools\preflight_check.py --profile quick_test --mode industry_router_only`.
  - Ran `python F:\quant_data\Ashare\launch_canonical.py --profile quick_test --mode industry_router_only --skip-preflight` twice; the first run exposed and fixed a `pandas.Series` truthiness bug in `resolve_mechanism`, the second run completed successfully.
  - Verified generated artifacts and backtest outputs under `F:\quant_data\Ashare\data\event_lake_v6\research\industry_router`.
  - No full integrated pipeline run was executed.
- Compatibility:
  - Not backward compatible at the internal module-layout level inside `hub_v6\industry_router`.
  - Upstream orchestration remains compatible because `build_industry_router_artifacts(...)` stayed the formal entrypoint.
  - Research output truth changed: `industry_state_daily.csv` is now treated as legacy and removed on rebuild.
- Rollback:
  - Restore the deleted legacy files and revert the new `contracts\`, `core\`, and `mechanisms\` subtree if the unified architecture must be abandoned.
  - If only the new schema is a problem, pin downstream consumers to the previous commit before `2026-03-22 18:47`.

### 2026-03-21 18:06
- Type:
  - `docs`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
  - `F:\quant_data\Ashare\AGENTS.md`
- Change:
  - Re-audited the real integrated call chain and corrected stale documentation that still implied `overnight` was the default profile.
  - Added an explicit `Active Runtime vs Archived Roots` section clarifying that the live research brain is the embedded `v5_gpu_runtime`, while root-level `quant_research_hub_v5*` copies are now archived only.
  - Documented the remaining naming debt around `V5_PROJECT_ROOT` and `project_root` so future sessions do not mistake config metadata for the actual launcher path.
- Impact:
  - Future Codex sessions and the operator now have a cleaner source of truth for where the live modules actually run.
  - No runtime behavior changed.
- Validation:
  - Manually re-read `main_research_runner.py`, `hub_v6/supervisor.py`, `hub_v6/local_settings.py`, and `v5_gpu_runtime/hub/config_utils.py`.
  - Verified the active V5 launcher path still resolves inside `quant_research_hub_v6_repacked_clean\...\v5_gpu_runtime`.
  - No code execution and no full pipeline run.
- Compatibility:
  - Backward compatible.
  - This change corrects stale handoff text only.
- Rollback:
  - Revert the doc updates if a later code change restores a different default profile or reintroduces a root-level V5 runtime as the live path.

### 2026-03-21 15:52
- Type:
  - `docs`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a dedicated `Core Module Relationship Map` section describing how the root entry, supervisor, V6 planning modules, V5 runtime, portfolio layer, execution bridge, and feedback loop connect.
- Impact:
  - Improves handoff quality and makes cross-module debugging easier for future Codex sessions and for the operator.
  - No runtime behavior changed.
- Validation:
  - Manual markdown review after patching.
  - No code execution and no full pipeline run.
- Compatibility:
  - Backward compatible.
- Rollback:
  - Remove or shorten the section if it becomes outdated or too heavy to maintain.

### 2026-03-21 15:32
- Type:
  - `bugfix`
- Scope:
  - `research`
  - `portfolio`
  - `execution`
  - `infra`
- Files:
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\llm_router.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\research_brief_engine.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_recommendation.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\v5_gpu_runtime\hub\model_families.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\v5_gpu_runtime\hub\training_engine.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\v5_gpu_runtime\hub\codegen.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\v5_gpu_runtime\hub\registry.py`
- Change:
  - Fixed OpenAI research fallback so unsupported `reasoning.effort` is avoided for non-reasoning models and auto-retried away if the API still rejects it; added transient network retry for OpenAI Responses calls.
  - Reworked XGBoost GPU prediction to use explicit `Booster.predict(DMatrix)` on the V5 side, avoiding the sklearn wrapper's device-mismatch fallback warning.
  - Replaced bare `Ridge` with a stabilized ridge wrapper that standardizes features and prunes constant / near-collinear columns before fit.
  - Fixed portfolio recommendation run selection to prefer the latest completed cycle and skip stale registry rows whose run directories no longer exist.
  - Prevented the integrated supervisor from executing the bridge after portfolio generation failure.
  - Added `resume_downstream` mode to continue from finished V5 artifacts without rerunning V6/V5, and sanitized generated lab code to strip markdown fences.
- Impact:
  - Affects V6 research planning reliability, V5 prediction stability, downstream portfolio selection, and operator recovery workflow.
  - No backward-incompatible change to the main `integrated_supervisor` entrypoint.
- Validation:
  - `py_compile` passed on all touched Python files.
  - Targeted probe confirmed the XGBoost prediction helper emits zero device-mismatch warnings.
  - Targeted probe confirmed the OpenAI client retries once without `reasoning` after a simulated unsupported-parameter error.
  - Ran `main_research_runner.py --mode resume_downstream --profile quick_test`; it generated a fresh portfolio recommendation from run `20260321_150548_398753e3`.
  - No full integrated pipeline run.
- Compatibility:
  - Backward compatible for normal daily runs.
  - Adds a new operator mode: `resume_downstream`.
- Rollback:
  - Revert the touched files if the recovery mode or model-stability changes prove undesirable; for operations, stop using `resume_downstream` and return to full reruns only.

### 2026-03-21 14:53
- Type:
  - `docs`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
  - `F:\quant_data\Ashare\AGENTS.md`
- Change:
  - Added a session-start checklist, a run-profile quick reference, and a reusable structured change-log template.
  - Tightened the rule that stable sections should be refreshed before appending a new historical entry when current truth changes.
- Impact:
  - Improves future Codex handoff quality and reduces the chance that later sessions skip key operating constraints.
  - No runtime behavior or output schema changed.
- Validation:
  - Manual markdown review after patching.
  - No code execution and no full pipeline run.
- Compatibility:
  - Backward compatible.
  - Existing log entries remain valid.
- Rollback:
  - Remove the added stable sections and template if the documentation is judged too heavy for future maintenance.

### 2026-03-21 14:49
- Type:
  - `docs`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
  - `F:\quant_data\Ashare\AGENTS.md`
- Change:
  - Reorganized the development log into a stable snapshot, artifact registry, config surface, known issues, deferred work, and decision log layout.
  - Tightened AGENTS guidance so future Codex sessions keep the log structured instead of appending free-form notes only.
- Impact:
  - Affects documentation and future handoff quality only.
  - No runtime behavior or output schema changed.
- Validation:
  - Manual read-through of the updated markdown structure.
  - No code execution or full pipeline run.
- Compatibility:
  - Backward compatible.
  - Old entries remain valid but are now treated as legacy short-form entries.
- Rollback:
  - Restore the previous simpler markdown layout if the extra structure becomes too heavy to maintain.

### 2026-03-21 09:48
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\tushare_client.py`
- Change:
  - Added more robust Tushare token handling, retry/backoff behavior, and client-side throttling behavior.
- Intended effect:
  - Make daily data sync and Tushare-backed pulls more stable under real quota pressure.

### 2026-03-21 09:50
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\v5_gpu_runtime\hub\candidate_factory.py`
- Change:
  - Added support for dynamic performance feedback input from the execution side.
- Intended effect:
  - Let previous-day performance influence route and candidate generation posture.

### 2026-03-21 09:53
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_recommendation.py`
- Change:
  - Added price enrichment from price snapshots and local daily data, plus feedback-aware portfolio controls.
- Intended effect:
  - Prevent target positions and rebalance orders from being unusable due to missing prices.

### 2026-03-21 09:54
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\gmtrade_runtime_config.local.json`
- Change:
  - Unified execution-bridge paths around the current portfolio output and price snapshot locations.
- Intended effect:
  - Keep Gmtrade simulation reading the right files from the integrated chain.

### 2026-03-21 11:53
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
- Change:
  - Fixed the `token_plan_min_interval_hours = 0` bug by preserving explicit zero instead of collapsing it through `or 24`.
  - Added dynamic strategy feedback handling around `performance_feedback.json`.
- Intended effect:
  - Make the V6 plan interval obey configuration exactly and let the strategy react to prior-day returns.

### 2026-03-21 12:01
- File:
  - `F:\quant_data\Ashare\main_research_runner.py`
- Change:
  - Promoted this file to the canonical root entry.
  - Defaulted the root chain to `integrated_supervisor`.
  - Added `--profile overnight` and `--profile quick_test`.
  - Added runtime-config generation per profile.
- Intended effect:
  - Provide one-click daily operation with a heavy overnight mode and a minimal end-to-end debug mode.

### 2026-03-21 12:08
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
- Change:
  - Restored `TOKEN_PLAN_MIN_INTERVAL_HOURS = 24`.
  - Added dual-profile runtime settings for `overnight` and `quick_test`.
  - Preserved the dual-Python arrangement for research vs. Gmtrade bridge.
  - Added explicit Tushare news source and rate-limit settings into runtime config generation.
- Intended effect:
  - Keep the daily default stable while still allowing fast debugging without editing multiple config files by hand.

### 2026-03-21 12:09
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\event_ingest.py`
- Change:
  - Reworked Tushare news ingestion to persist fetch state.
  - Added local rate-limit guarding.
  - Switched `major_news` to supported source names.
  - Added detailed per-source news logging.
  - Preserved the announcement path while making its logs clearer.
- Intended effect:
  - Fix the misleading `Tushare 新闻抓取完成，条数=0` behavior when the real issue was source validity or quota exhaustion.

### 2026-03-21 12:49
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\event_extract.py`
- Change:
  - Rebuilt the event-extraction layer around source reliability, recency, corroboration, entity specificity, and anti-overfit weighting.
  - Added duplicate/cluster control so repeated or routine titles do not dominate the selected event set.
  - Added richer `structured_facts` metadata such as evidence quality, source diversity, corroboration count, impact scope, and anti-overfit weight.
- Intended effect:
  - Make message-side influence depend on evidence quality instead of crude title-level importance alone.

### 2026-03-21 12:50
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\context_pack.py`
- Change:
  - Rebuilt the context pack so `priority_events` are sorted by robust research priority instead of raw importance only.
  - Added `message_evidence_profile` and stronger summary metrics like confirmed events, weak-signal ratio, and average anti-overfit weight.
- Intended effect:
  - Give the research planner a higher-quality summary of the message layer and reduce the chance of headline noise steering the whole plan.

### 2026-03-21 12:53
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\research_brief_engine.py`
- Change:
  - Reworked the rule-based research-brief generator so message-side signals affect planning only after passing quality and anti-overfit gates.
  - Switched default feature derivation toward quality-weighted, confirmation-aware, and noise-filter features instead of title-like one-off ideas.
  - Added stronger stop conditions and ban items against single-headline overfitting.
- Intended effect:
  - Let the message layer materially influence research direction while staying inside a more defensible generalization regime.

### 2026-03-21 12:54
- File:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\tushare_client.py`
- Change:
  - Changed rate-limit handling so Tushare news calls fail fast on quota errors instead of spending a long time retrying.
- Intended effect:
  - Avoid wasting V6 planning time when the upstream Tushare news quota is already exhausted.

### 2026-03-21 17:26
- Type:
  - `config`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\.gitignore`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
- Change:
  - Added a root `.gitignore` to keep runtime data, logs, virtual environments, local configs, autogen configs, and machine-specific runtime JSON files out of Git.
  - Added a safe `local_settings.example.py` so the repository keeps a usable template without publishing local tokens or machine paths from the real `local_settings.py`.
- Impact:
  - Git staging now focuses on source code, docs, and safe examples only.
  - Runtime artifacts under `data/`, local execution configs, and real local settings remain local-only.
- Validation:
  - Cleared the existing staged index with `git rm -r --cached -- .`.
  - Restaged only the source/doc whitelist and inspected `git status --short`.
  - Checked staged config files for secret-bearing local settings and machine-specific runtime JSON.
  - No commit, push, or full pipeline run was performed.
- Compatibility:
  - Backward compatible for local runtime behavior.
  - New clones will require developers to create their own local settings/runtime configs from example files.
- Rollback:
  - Remove or relax the new `.gitignore` rules and restage the previously excluded files if a private-only repository later needs to track them.

### 2026-03-21 17:41
- Type:
  - `ops`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\.githooks\post-commit`
  - `F:\quant_data\Ashare\.gitattributes`
  - `F:\quant_data\Ashare\AGENTS.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a local `post-commit` hook under `.githooks` so this workspace auto-pushes to GitHub after each successful local commit.
  - Added an opt-out switch for PowerShell: `$env:DISABLE_AUTO_PUSH='1'; git commit -m "..."; Remove-Item Env:DISABLE_AUTO_PUSH`.
  - Added `.gitattributes` rule forcing LF line endings under `.githooks` so the shell hook stays executable on Windows Git.
  - Documented the new auto-push behavior in the stable snapshot, dangerous-operations section, and AGENTS handoff notes.
- Impact:
  - Normal `git commit` now also triggers a GitHub push in this workspace.
  - Operators need to remember that a commit is no longer local-only unless they explicitly disable auto-push for that command.
- Validation:
  - Wrote the hook file and configured `core.hooksPath` locally.
  - Staged the hook/doc updates.
  - No extra commit, no test push, and no pipeline run were performed.
- Compatibility:
  - Backward compatible with current repo contents.
  - Hook activation is local to this workspace because `core.hooksPath` is stored in local git config.
- Rollback:
  - Run `git config --unset core.hooksPath` and remove `.githooks/post-commit` if you want to return to manual push behavior.

### 2026-03-21 17:38
- Type:
  - `ops`
- Scope:
  - `docs`
- Files:
  - `F:\quant_data\Ashare\scripts\sync_codex_dev_log_to_gdrive.py`
  - `F:\quant_data\Ashare\scripts\start_codex_dev_log_sync.ps1`
  - `F:\quant_data\Ashare\scripts\stop_codex_dev_log_sync.ps1`
  - `F:\quant_data\Ashare\AGENTS.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a lightweight watcher that mirrors `CODEX_DEV_LOG.md` into Google Drive whenever the file changes.
  - Added PowerShell start/stop scripts for the watcher and documented the mirror path plus operational caveats.
  - Started the watcher in the current session after validating one-shot sync behavior.
- Impact:
  - Future edits to `CODEX_DEV_LOG.md` in this session will automatically produce both a latest copy and timestamped history copies under Google Drive.
  - The mirror is session-local unless the watcher is started again after reboot.
- Validation:
  - `python -m py_compile` passed for `scripts/sync_codex_dev_log_to_gdrive.py`.
  - One-shot sync copied the log to `G:\我的云端硬盘\Ashare_backups\codex_dev_log_mirror`.
  - Verified a running background watcher process: `pythonw.exe` with `sync_codex_dev_log_to_gdrive.py`.
- Compatibility:
  - Backward compatible with current repo behavior.
  - No change to the research/execution pipeline.
- Rollback:
  - Stop the watcher with `scripts/stop_codex_dev_log_sync.ps1` and delete the `scripts/sync_codex_dev_log_to_gdrive.py` family if the mirror is no longer wanted.

### 2026-03-21 17:50
- Type:
  - `ops`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\scripts\sync_codex_dev_log_to_gdrive.py`
  - `F:\quant_data\Ashare\scripts\start_codex_dev_log_sync.ps1`
  - `F:\quant_data\Ashare\scripts\stop_codex_dev_log_sync.ps1`
  - `F:\quant_data\Ashare\scripts\install_codex_dev_log_sync_autostart.ps1`
  - `F:\quant_data\Ashare\scripts\remove_codex_dev_log_sync_autostart.ps1`
  - `F:\quant_data\Ashare\AGENTS.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Lowered watcher runtime overhead by switching to 5-second polling, file-stat checks before hashing, and idle process priority on Windows.
  - Added a Windows single-instance mutex so duplicate watcher launches exit safely.
  - Added logon autostart install/remove scripts using Task Scheduler instead of Startup-folder style persistence.
  - Registered the scheduled task `Ashare Codex Dev Log Mirror` in the current workspace.
  - Routed autostart through hidden PowerShell plus the existing start script because direct scheduled launches of the watcher were less stable.
- Impact:
  - `CODEX_DEV_LOG.md` mirror now survives reboot/login without manual startup, provided the scheduled task remains enabled.
  - Resource usage stays very low and the launcher relies on Windows-native scheduling primitives, which should be less fragile than ad-hoc autorun methods in a 360-enabled environment.
- Validation:
  - `python -m py_compile` passed for `scripts/sync_codex_dev_log_to_gdrive.py`.
  - Manual watcher start successfully produced a stable background `python.exe` process.
  - `schtasks /Run /TN "Ashare Codex Dev Log Mirror"` successfully launched the watcher.
  - `schtasks /Query /TN "Ashare Codex Dev Log Mirror" /V /FO LIST` ended in `Status: Ready` and `Last Result: 0`.
- Compatibility:
  - Backward compatible with the existing Google Drive mirror path and start/stop scripts.
  - The scheduled task is local-machine state; cloning the repo elsewhere will not recreate it automatically unless the install script is run there.
- Rollback:
  - Run `scripts/remove_codex_dev_log_sync_autostart.ps1`, then use `scripts/stop_codex_dev_log_sync.ps1` if you want to disable both autostart and the current watcher instance.

### 2026-03-21 18:03
- Type:
  - `refactor`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v5`
  - `F:\quant_data\Ashare\quant_research_hub_v5_1`
  - `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Moved the unused root-level legacy packages `quant_research_hub_v5` and `quant_research_hub_v5_1` out of the repo root.
  - Archived them under `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321` so the repo root keeps only the current live chain plus tooling.
- Impact:
  - The repo root is materially cleaner and less ambiguous.
  - Future work on the live chain is less likely to accidentally touch obsolete package trees.
  - A future git commit from this workspace will record these legacy roots as deletions from the repository.
- Validation:
  - Confirmed `main_research_runner.py` still points only at `quant_research_hub_v6_repacked_clean`.
  - Re-listed the repo root after the move and verified both legacy directories are now absent there.
  - Re-listed `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321` and confirmed both archived directories landed there.
  - No full pipeline run was performed.
- Compatibility:
  - Backward compatible for the current live chain.
  - Any manual workflow that still expected the old root-level `quant_research_hub_v5*` directories must now read them from the archive location instead.
- Rollback:
  - Move `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321\quant_research_hub_v5*` back into `F:\quant_data\Ashare\` if the cleanup must be reverted locally.

### 2026-03-21 19:29
- Type:
  - `docs`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\PROJECT_LAW.md`
  - `F:\quant_data\Ashare\SYSTEM_MANIFEST.yaml`
  - `F:\quant_data\Ashare\RUN_PROFILES.yaml`
  - `F:\quant_data\Ashare\CHANGELOG_CANONICAL.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a low-risk governance layer that defines the formal operator entry, wrapped business root entry, unique live runtime root, formal run-trace output root, and archive/experiment/deprecated directory meanings.
  - Added a canonical governance changelog separate from the broader Codex handoff log.
- Impact:
  - Improves operator clarity and reduces ambiguity about which paths are live versus historical or experimental.
  - No runtime behavior, imports, or business logic changed in this phase.
- Validation:
  - Manual file review after creation.
  - Confirmed the law files match the current documented live chain and default profile.
  - No full pipeline run was performed.
- Compatibility:
  - Backward compatible.
  - Phase 1 is documentation/governance only.
- Rollback:
  - Remove the new governance files and this entry if the law layer is not wanted.

### 2026-03-21 19:36
- Type:
  - `ops`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\launch_canonical.py`
  - `F:\quant_data\Ashare\tools\preflight_check.py`
  - `F:\quant_data\Ashare\AGENTS.md`
  - `F:\quant_data\Ashare\CHANGELOG_CANONICAL.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added `launch_canonical.py` as the formal governance wrapper around `main_research_runner.py`.
  - Added `tools/preflight_check.py` for lightweight path, profile, import, and `py_compile` checks without running heavy business workloads.
  - Made the wrapper dispatch to the canonical research Python from `hub_v6/local_settings.py` so the formal operator path does not depend on whichever shell Python launched the wrapper.
  - Updated AGENTS and the stable log sections so future sessions distinguish the formal operator entry from the wrapped business root.
- Impact:
  - Formal operator runs now have one wrapper entrypoint without changing the business call chain.
  - Preflight failures can stop obvious path/profile/import mistakes before a real run starts.
- Validation:
  - `python -m py_compile launch_canonical.py tools\preflight_check.py`
  - `python tools\preflight_check.py --profile quick_test --mode integrated_supervisor`
  - `python launch_canonical.py --preflight-only --profile quick_test --mode integrated_supervisor`
  - No full pipeline run was performed in this phase.
- Compatibility:
  - Backward compatible for the underlying business chain.
  - Direct `main_research_runner.py` usage still works when explicitly chosen.
- Rollback:
  - Remove `launch_canonical.py` and `tools/preflight_check.py`, then restore the log/AGENTS wording if the wrapper layer is not wanted.

### 2026-03-21 19:43
- Type:
  - `ops`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\tools\register_run.py`
  - `F:\quant_data\Ashare\launch_canonical.py`
  - `F:\quant_data\Ashare\AGENTS.md`
  - `F:\quant_data\Ashare\CHANGELOG_CANONICAL.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added `tools\register_run.py` as a thin governance utility for creating and updating `run_manifest.json`.
  - Updated `launch_canonical.py` so every formal run gets a dedicated `outputs\canonical_runs\<run_id>\` directory and a wrapper-level run manifest.
  - Recorded the new trace artifact in AGENTS and the stable dev-log sections.
- Impact:
  - Formal runs now have a lightweight audit trail with run id, timestamps, entrypoint, runtime root, data root, output root, and optional git commit.
  - No business module imports, runtime config generation, or downstream logic were moved into the governance layer.
- Validation:
  - `python -m py_compile launch_canonical.py tools\register_run.py`
  - `python tools\register_run.py --profile quick_test --mode integrated_supervisor`
  - `python launch_canonical.py --preflight-only --profile quick_test --mode integrated_supervisor`
  - No full pipeline run was performed in this phase.
- Compatibility:
  - Backward compatible for direct `main_research_runner.py` usage.
  - Wrapper-launched formal runs now create one extra governance output directory under `outputs\canonical_runs`.
- Rollback:
  - Remove `tools\register_run.py`, remove the wrapper manifest calls from `launch_canonical.py`, and delete the generated `outputs\canonical_runs` directories if the trace layer is not wanted.

### 2026-03-21 19:49
- Type:
  - `docs`
- Scope:
  - `infra`
- Files:
  - `F:\quant_data\Ashare\PROJECT_LAW.md`
  - `F:\quant_data\Ashare\SYSTEM_MANIFEST.yaml`
  - `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321\CANONICAL_STATUS.md`
  - `F:\quant_data\Ashare\outputs\CANONICAL_STATUS.md`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\examples\CANONICAL_STATUS.md`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\docs\DEPRECATED_ENTRYPOINTS.md`
  - `F:\quant_data\Ashare\CHANGELOG_CANONICAL.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added explicit marker files for mixed-output, examples, and deprecated-entrypoint locations so operators can see which paths are non-live.
  - Corrected the governance documents so `outputs\canonical_runs` is treated as formal while the rest of `outputs\` remains experimental by default.
- Impact:
  - Reduces the chance of accidental work in non-current paths without moving any legacy or example files.
  - No runtime behavior or import structure changed in this phase.
- Validation:
  - Manual file review after patching.
  - Confirmed the markers point operators back to `launch_canonical.py` and `main_research_runner.py`.
  - No full pipeline run was performed in this phase.
- Compatibility:
  - Backward compatible.
  - Marker files are additive only.
- Rollback:
  - Remove the marker files and restore the prior law/manifest wording if the explicit path-status layer is not wanted.

### 2026-03-21 20:01
- Type:
  - `ops`
- Scope:
  - `runtime`
  - `observability`
- Files:
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\orchestrator_v6.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added stage-preview output in `main_research_runner.py` so the operator can see the planned chain before execution starts.
  - Added clearer step-by-step V6 progress logs in `orchestrator_v6.py` with stage counters and elapsed time.
  - Reworked `supervisor.py` so `supervisor_state.json` is written incrementally during runtime and now records `current_stage`, per-stage status, and recent stage history.
  - Added clearer start/finish/skip/fail messages around market sync, strategy feedback, V6 planning, V5 GPU research, portfolio generation, execution bridge, and resume-downstream steps.
  - Added V5 visibility hints pointing operators to `controller_state.json`, `registry/experiment_registry.csv`, and `cycles/*/cycle_summary.json` while the GPU stage is running.
- Impact:
  - The integrated chain is less black-box during long runs.
  - Operators can inspect both the terminal output and `supervisor_state.json` to see what is currently running and what already finished.
  - No business logic, data logic, or import structure changed.
- Validation:
  - `python -m py_compile main_research_runner.py quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\orchestrator_v6.py quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `python tools\preflight_check.py --profile quick_test --mode integrated_supervisor`
  - `python launch_canonical.py --preflight-only --profile quick_test --mode integrated_supervisor`
  - No full pipeline run was performed.
- Compatibility:
  - Backward compatible.
  - Existing downstream consumers of `supervisor_state.json` should continue to work because the new fields are additive.
- Rollback:
  - Remove the new progress-printing helpers and incremental stage bookkeeping if the extra observability is judged too noisy.

### 2026-03-21 20:23
- Type:
  - `docs`
- Scope:
  - `collaboration`
  - `github`
- Files:
  - `F:\quant_data\Ashare\README.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a root `README.md` aimed at GitHub collaborators so the repository homepage now exposes the current formal entrypoint, required local setup, excluded data scope, and the recommended preflight-first startup path.
  - Explicitly documented that large training tables and other `data/` artifacts are expected to be delivered out of band rather than through the repository.
- Impact:
  - New collaborators can clone the repo and see the current runnable chain without having to infer it from older nested docs.
  - No code path, import structure, runtime behavior, or data contract changed.
- Validation:
  - Manual review of the new root README against the current canonical entrypoints and profiles documented in this log.
  - No integrated runtime was executed.
- Compatibility:
  - Backward compatible.
  - Documentation only.
- Rollback:
  - Remove `README.md` and this log entry if a root collaborator-facing readme is no longer desired.

### 2026-03-22 00:19
- Type:
  - `ops`
- Scope:
  - `runtime`
  - `local_llm`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\json_parse_utils.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\llm_router.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_ollama_worker.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\event_extract.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Split the local Ollama runtime into two explicit roles: a strict event-extraction model path and a separate local research fallback path.
  - Kept local event extraction on `qwen2.5:7b`, and configured the local research fallback chain as `deepseek-r1:14b` first with `qwen2.5:7b` behind it.
  - Added a loose JSON-object parser so local Ollama responses can still be accepted when they wrap a valid JSON object with extra text or markdown fences.
  - Updated event-extract logging and summary output so operators can see which local extraction model was used.
- Impact:
  - The local 14B model is now integrated into the V6 local research fallback path without disturbing the stricter event-extract worker path.
  - Local Ollama parsing is more tolerant of verbose or fenced responses, reducing brittle failures when a model includes extra wrapper text around JSON.
  - No supervisor flow, V5 main chain, or cloud-provider priority order was changed.
- Validation:
  - Lightweight static validation only.
  - `python -m py_compile` on touched Python files.
  - `python tools\preflight_check.py --profile quick_test --mode integrated_supervisor`
  - targeted config inspection of the generated `local_ollama` runtime section.
  - No full integrated runtime was executed.
- Compatibility:
  - Backward compatible.
  - Legacy `OLLAMA_MODEL` / `OLLAMA_TIMEOUT_SECONDS` remain as fallback defaults if the dedicated split settings are absent.
- Rollback:
  - Remove the dedicated `OLLAMA_EVENT_*` / `OLLAMA_RESEARCH_*` settings, revert `config_builder.py`, `event_extract.py`, and the loose local JSON parsing helpers if the split local-model strategy is not wanted.

### 2026-03-22 00:33
- Type:
  - `ops`
- Scope:
  - `runtime`
  - `observability`
  - `local_llm`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_augmentations.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\context_pack.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\research_brief_engine.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\orchestrator_v6.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added `local_augmentations.py` as an additive sidecar layer for four local-Ollama tasks: announcement evidence cards, manual-review queue routing, long-stage runtime notes, and V5 post-cycle review.
  - Added a short cached Ollama health check inside the sidecar layer so these augmentations skip quickly when the local service is unavailable instead of waiting for full per-role timeouts.
  - Wired V6 ingest to generate `announcement_evidence_cards.json` from high-value announcements, then merge compact evidence cards into the research context pack before brief generation.
  - Wired V6 extract to generate `manual_review_queue.json` and print queue summary output so the operator can see how many items need manual attention.
  - Wired `supervisor.py` to emit `runtime_stage_notes.json` plus terminal hints at selected long stages, and to generate `latest_v5_cycle_review.json` after a successful V5 run without changing the V5 control flow.
  - Added explicit local-Ollama role configs for evidence-card, review-router, runtime-explainer, and V5-review tasks.
- Impact:
  - The local 14B model now has two higher-value sidecar jobs beyond research fallback: announcement evidence-card generation and V5 cycle review.
  - The local 7B model now has two lighter operator-support jobs: manual-review routing and long-stage runtime explanations.
  - Runtime transparency is improved through extra terminal output, supervisor state notes, and additive JSON sidecar artifacts.
  - No portfolio decision rule, execution rule, import tree, or main stage order was rewritten.
- Validation:
  - Lightweight validation only.
  - `python -m py_compile` on touched V6 modules and `local_augmentations.py`.
  - `python tools\preflight_check.py --profile quick_test --mode integrated_supervisor`
  - No full integrated runtime was executed.
- Compatibility:
  - Backward compatible.
  - All new local-Ollama features are best-effort sidecars and should not block the main chain if local inference fails.
- Rollback:
  - Remove `local_augmentations.py`, remove the additive calls from `orchestrator_v6.py` and `supervisor.py`, and remove the new role-specific local-Ollama config keys if the sidecar enhancement layer is not wanted.

### 2026-03-22 00:43
- Type:
  - `ops`
- Scope:
  - `backup`
  - `versioning`
- Files:
  - `G:\我的云端硬盘\Ashare_backups\script_versions\Ashare_script_20260322_r001_pre_big_update`
  - `G:\我的云端硬盘\Ashare_backups\VERSION_CATALOG.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Created a new Google Drive script snapshot version `SCRIPT-20260322-R001` before the next planned large update.
  - Stored the snapshot under `G:\我的云端硬盘\Ashare_backups\script_versions\Ashare_script_20260322_r001_pre_big_update` without modifying the existing legacy backup folder.
  - Added a catalog file at `G:\我的云端硬盘\Ashare_backups\VERSION_CATALOG.md` to classify the existing pre-restructure snapshot as legacy and the new one as the active script snapshot.
  - Packed three recovery layers into the new version folder:
    - `repo_scripts\` for direct file browsing
    - `manifest\repo_history.bundle` for committed git history
    - `manifest\working_tree.patch` for the exact uncommitted delta on top of git head `0488e87`
  - Excluded `data/`, `outputs/`, `.git/`, virtualenvs, caches, and private `local_settings.py` from the copied script snapshot.
- Impact:
  - There is now a frozen, non-overwriting pre-big-update script backup in Google Drive that can be restored either as plain files or through git history plus patch.
  - Sensitive local runtime settings were intentionally kept out of the copied script tree.
- Validation:
  - Confirmed the new version folder and manifest files exist.
  - Confirmed key operator files such as `launch_canonical.py`, `main_research_runner.py`, `tools\preflight_check.py`, and `hub_v6\local_augmentations.py` exist inside the snapshot.
  - Confirmed `local_settings.py` does not exist inside the copied `repo_scripts\` tree.
- Compatibility:
  - Additive only.
  - Existing Google Drive backup folders were left untouched.
- Rollback:
  - Delete `G:\我的云端硬盘\Ashare_backups\script_versions\Ashare_script_20260322_r001_pre_big_update` and remove the matching catalog entry if this snapshot should not be kept.

### 2026-03-22 01:12
- Type:
  - `ops`
- Scope:
  - `execution`
  - `portfolio_control`
  - `observability`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\portfolio_control.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\dev_log_snapshot.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\runtime.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\brokers\gmtrade_sim_broker.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\gmtrade_runtime_config.example.json`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\gmtrade_runtime_config.local.json`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a low-risk portfolio control V1 layer inside the execution bridge instead of introducing a separate OMS.
  - Current scope is deliberately limited to:
    - unified position-state ledger
    - drift-threshold rebalance suppression
    - daily turnover budgeting with truncation
    - normalized execution feedback
    - dev-log live portfolio snapshot refresh
  - Execution runs now write per-run control artifacts under `data\live_execution_bridge\portfolio_control_runs\<timestamp>\`.
  - `gmtrade_sim_broker.py` now queries day orders and unfinished orders so execution feedback is based on actual gmtrade order objects rather than only local intent files.
  - Added a dedicated `Latest Live Portfolio Snapshot` section in the dev log maintenance flow; successful execution runs can refresh only that small block instead of rewriting unrelated log content.
  - Kept industry/theme exposure and staged entry/exit out of this version on purpose to avoid hardening the downstream chain too early.
- Impact:
  - The execution layer is no longer a pure black-box rebalance submitter; it now leaves a stable audit trail for target, actual, pending, and post-execution estimated positions.
  - Small weight drift can now be skipped, and large one-day target changes can be truncated by turnover budget instead of forcing full turnover in one run.
  - Future Codex sessions and collaborators can inspect the dev log first and immediately see the latest live holdings snapshot when recent execution data exists.
- Validation:
  - `python -m py_compile` on the new control-layer modules plus touched runtime/config files.
  - `python tools\preflight_check.py --profile quick_test --mode integrated_supervisor`
  - A targeted dry planning probe using existing `latest_account_state.json` and `target_positions.csv`, which produced:
    - `n_raw_orders=24`
    - `n_final_orders=4`
    - `raw_turnover_ratio≈0.9602`
    - `final_turnover_ratio≈0.2490`
  - No live or full integrated runtime was executed in this session.
- Compatibility:
  - Backward compatible at the chain level.
  - The supervisor, portfolio recommendation layer, and gmtrade bridge entrypoints remain unchanged.
  - New artifacts and report fields are additive.
- Rollback:
  - Remove `portfolio_control.py` and `dev_log_snapshot.py`, revert the execution-bridge runtime/broker patches, and delete the new `portfolio_control` config section if the V1 control layer is not wanted.

### 2026-03-22 02:00
- Type:
  - `architecture`
- Scope:
  - `research_execution_split`
  - `release_layer`
  - `trade_clock`
- Files:
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\launch_canonical.py`
  - `F:\quant_data\Ashare\trade_clock_service.py`
  - `F:\quant_data\Ashare\RUN_PROFILES.yaml`
  - `F:\quant_data\Ashare\SYSTEM_MANIFEST.yaml`
  - `F:\quant_data\Ashare\PROJECT_LAW.md`
  - `F:\quant_data\Ashare\CHANGELOG_CANONICAL.md`
  - `F:\quant_data\Ashare\README.md`
  - `F:\quant_data\Ashare\scripts\start_trade_clock.ps1`
  - `F:\quant_data\Ashare\scripts\stop_trade_clock.ps1`
  - `F:\quant_data\Ashare\scripts\install_trade_clock_autostart.ps1`
  - `F:\quant_data\Ashare\scripts\remove_trade_clock_autostart.ps1`
  - `F:\quant_data\Ashare\tools\preflight_check.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_release.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\trading_clock.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_manager.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\clock_supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_bridge_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\runtime.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added three additive precise-style modes:
    - `research_only`
    - `release_only`
    - `execution_only`
  - Added a formal portfolio release layer that versions `portfolio_recommendation.json` and `target_positions.csv` into `data\trade_release_v1\releases\<release_id>\`.
  - Added a lightweight trade clock that watches trading day plus execution windows and triggers execution from the published release instead of from the live research chain.
  - Added a root-level `trade_clock_service.py` plus Windows start/stop/install/remove scripts.
  - Installed the Windows logon scheduled task `Ashare Trade Clock` and started the clock process in the current session.
- Impact:
  - Research production and execution timing are now separated by a physical file contract instead of a single heavy integrated run.
- The execution side now knows:
  - which account mode is active (`simulation` or `precision`)
  - whether precision execution is globally enabled for this run
  - whether a release exists
  - what trade date it belongs to
  - whether the current time is inside the configured execution window
  - The always-on process is intentionally thin: heartbeat plus gate check only, no heavy research work.
- Validation:
  - `python -m py_compile` on all new root/runtime modules and touched wrappers.
  - `python tools\preflight_check.py --profile quick_test --mode research_only`
  - `python tools\preflight_check.py --profile quick_test --mode execution_only`
  - `python launch_canonical.py --preflight-only --profile quick_test --mode research_only`
  - `python launch_canonical.py --preflight-only --profile quick_test --mode execution_only`
  - `python launch_canonical.py --profile quick_test --mode release_only --skip-preflight`
  - `python launch_canonical.py --profile quick_test --mode execution_only --gate-only --skip-preflight`
  - `python trade_clock_service.py --profile quick_test --once --skip-preflight`
  - `powershell -ExecutionPolicy Bypass -File scripts\start_trade_clock.ps1 -Profile quick_test`
  - `powershell -ExecutionPolicy Bypass -File scripts\stop_trade_clock.ps1`
  - `powershell -ExecutionPolicy Bypass -File scripts\install_trade_clock_autostart.ps1`
  - `schtasks /Query /TN "Ashare Trade Clock"`
  - No full integrated pipeline and no real timed execution dispatch were run in this session.
- Compatibility:
  - Additive only.
  - `integrated_supervisor` and `resume_downstream` remain available.
  - The live research chain, import tree, and gmtrade execution core were not rewritten.
- Rollback:
  - Remove the new trade clock scripts and `trade_clock_service.py`.
  - Remove `portfolio_release.py`, `trading_clock.py`, `execution_manager.py`, `clock_supervisor.py`, and `execution_bridge_runner.py`.
  - Remove the new modes from `main_research_runner.py`, `launch_canonical.py`, and `RUN_PROFILES.yaml`.
  - Remove the scheduled task `Ashare Trade Clock` and stop the running clock process.

### 2026-03-22 02:15
- Type:
  - `ops`
- Scope:
  - `execution_policy`
  - `account_switching`
  - `precision_safety_gate`
- Files:
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\launch_canonical.py`
  - `F:\quant_data\Ashare\trade_clock_service.py`
  - `F:\quant_data\Ashare\scripts\start_trade_clock.ps1`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_bridge_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_manager.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_release.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\gmtrade_runtime_config.local.json`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\gmtrade_runtime_config.example.json`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a real execution-policy surface with:
    - `EXECUTION_ACCOUNT_MODE`
    - `PRECISION_TRADE_ENABLED`
    - `ALLOW_INTEGRATED_PRECISION_EXECUTION`
  - Added CLI override entries:
    - `--execution-mode simulation|precision`
    - `--precision-trade on|off`
  - Registered the new precision account profile in the local gmtrade template and kept the simulation profile in place.
  - Tightened the default local policy to:
    - `precision`
    - `precision_trade_enabled = False`
  - `execution_only` now distinguishes:
    - simulation mode: release-driven, no trading-time gate
    - precision mode: trading-time gated, and blocked entirely when precision trade is off
  - `integrated_supervisor` and `resume_downstream` now skip direct execution against the precision account unless `ALLOW_INTEGRATED_PRECISION_EXECUTION=True`.
- Impact:
  - You now have one explicit entry to switch account profiles and one explicit entry to allow or deny precision execution.
  - The clock service can remain online in precision mode without risking accidental order submission.
  - Simulation-mode debug behavior stays available without needing to rewrite the execution bridge.
- Validation:
  - `python -m py_compile` on all touched wrappers/runtime modules.
  - `python tools\preflight_check.py --profile quick_test --mode execution_only`
  - `python launch_canonical.py --preflight-only --profile quick_test --mode execution_only --execution-mode precision --precision-trade off`
  - `python launch_canonical.py --profile quick_test --mode release_only --skip-preflight --execution-mode precision --precision-trade off`
  - `python launch_canonical.py --profile quick_test --mode execution_only --gate-only --skip-preflight --execution-mode simulation`
  - `python launch_canonical.py --profile quick_test --mode execution_only --gate-only --skip-preflight --execution-mode precision --precision-trade off`
  - `python trade_clock_service.py --profile quick_test --once --skip-preflight --execution-mode precision --precision-trade off`
  - `powershell -ExecutionPolicy Bypass -File scripts\start_trade_clock.ps1 -Profile quick_test`
  - Verified `clock_state.json` shows `account_mode=precision`, `precision_trade_enabled=false`, and `reason=precision_trade_disabled`.
- Compatibility:
  - Additive at the interface level.
  - Existing simulation-style runs can still be invoked explicitly through `--execution-mode simulation`.
- Rollback:
  - Remove the new execution-policy fields and CLI overrides.
  - Restore the old single-account broker template if dual-account switching is no longer wanted.

### 2026-03-22 02:20
- Type:
  - `backup`
- Scope:
  - `gdrive_snapshot`
- Files:
  - `G:\我的云端硬盘\Ashare_backups\script_versions\Ashare_script_20260322_r002_precision_clock_split`
  - `G:\我的云端硬盘\Ashare_backups\VERSION_CATALOG.md`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Created a new frozen script snapshot version `SCRIPT-20260322-R002`.
  - Stored it under `script_versions\Ashare_script_20260322_r002_precision_clock_split`.
  - Included:
    - `repo_scripts\`
    - `manifest\repo_history.bundle`
    - `manifest\working_tree.patch`
    - `manifest\snapshot_manifest.json`
    - `manifest\git_status.txt`
    - `manifest\git_diff_stat.txt`
- Impact:
  - There is now a restorable pre-github-update snapshot covering the precise clock split and dual-account execution-policy changes.
- Validation:
  - Confirmed the new snapshot folder exists and the catalog entry was appended.
- Compatibility:
  - Additive only.
- Rollback:
  - Delete the `SCRIPT-20260322-R002` folder and remove its catalog row if the snapshot should not be kept.

### 2026-03-22 02:45
- Type:
  - `docs`
- Scope:
  - `mode_matrix`
  - `config_surface_clarification`
- Files:
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a dedicated stable section documenting the full runtime mode matrix.
  - Split the explanation into:
    - profile layer
    - run-mode layer
    - execution-account layer
    - precision-trade switch layer
  - Added a dedicated control-surface section showing exactly where each operator-facing default is changed:
    - `hub_v6/local_settings.py`
    - `hub_v6/local_settings.example.py`
    - `configs/gmtrade_runtime_config.local.json`
    - CLI override flags on `launch_canonical.py` / `main_research_runner.py`
  - Marked the generated runtime JSON files as non-hand-edit targets.
- Impact:
  - Future Codex sessions and collaborators now have one fixed section to understand:
    - what each mode does
    - which mode is formal vs debug-oriented
    - which file controls long-term defaults
    - which switches are safe for one-off runtime overrides
- Validation:
  - Re-read current mode definitions from `main_research_runner.py`.
  - Re-read current default values from `hub_v6/local_settings.py`.
  - Re-read current account mapping from `configs/gmtrade_runtime_config.local.json`.
- Compatibility:
  - Documentation only; no runtime behavior changed.
- Rollback:
  - Remove the new stable sections and this change-log entry if a different documentation structure is preferred.

### 2026-03-22 04:25
- Type:
  - `feature`
- Scope:
  - `industry_router_phase1`
  - `cross_layer_research_contracts`
- Files:
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\RUN_PROFILES.yaml`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\orchestrator_v6.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\context_pack.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\research_brief_engine.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\__init__.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\contracts.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\runtime.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\backtest.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\industry_router\stock_master.seed.csv`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\industry_router\mechanism_map.seed.csv`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\industry_router\event_taxonomy.json`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\industry_router\source_contracts.json`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a formal `hub_v6/industry_router/` module as the Phase 1 stock/mechanism skeleton.
  - Locked the three initial mechanism groups to:
    - `trend_capex`
    - `price_inventory`
    - `macro_style`
  - Landed the required static contracts:
    - `stock_master.seed.csv`
    - `mechanism_map.seed.csv`
    - `event_taxonomy.json`
    - `source_contracts.json`
  - Added a new targeted runtime mode:
    - `industry_router_only`
  - Hooked the router into V6 so that `plan_only`, `bridge_only`, and `full_cycle` now pass through industry-router generation before higher-layer research context packaging.
  - Extended `research_context_pack.json` and the LLM compact context with `industry_router` summary fields.
  - Added split mechanism backtest outputs and attribution reports under `data\event_lake_v6\research\industry_router\backtests`.
- Why cross-layer change was necessary:
  - the old structure treated messages mainly as one shared evidence pool with unified priority scoring
  - that structure could not express:
    - stock as the final scoring subject
    - mechanism as interpretation context
    - event-to-stock mapping as an explicit contract
    - split mechanism backtests as first-class outputs
  - keeping the old contract would have forced future mechanism work into sidecar patches and made later integration harder
- Impact:
  - The main system now has a formal, durable place for stock/mechanism research contracts instead of only unified message scoring.
  - V6 planning can now consume a mechanism-aware summary without rewriting the V5 runtime or execution bridge.
  - Operators now have a lightweight way to rebuild only the industry-router artifacts and split backtests from existing event history.
- Validation:
  - `python -m py_compile` on all touched wrappers/runtime modules and the new `industry_router` package.
  - `python tools\preflight_check.py --profile quick_test --mode industry_router_only`
  - `python launch_canonical.py --preflight-only --profile quick_test --mode industry_router_only`
  - `python launch_canonical.py --profile quick_test --mode industry_router_only --skip-preflight`
  - Confirmed runtime outputs:
    - `industry_router_summary.json`
    - `stock_signal_daily.csv`
    - `latest_stock_signal.csv`
    - `backtest_trend_capex_*`
    - `backtest_price_inventory_*`
    - `backtest_macro_style_*`
    - `backtest_combined_*`
    - `backtest_attribution_*`
  - Confirmed the latest successful `industry_router_only` run at `2026-03-22 04:18:14`.
- Compatibility:
  - baseline execution / release / trade-clock behavior remains unchanged
  - the router is additive to execution behavior in Phase 1 and currently changes research-side context and artifacts only
  - `integrated_supervisor`, `research_only`, `release_only`, and `execution_only` remain available
- Rollback:
  - remove `hub_v6/industry_router/` and `configs/industry_router/`
  - remove `industry_router_only` from `main_research_runner.py` and `RUN_PROFILES.yaml`
  - remove the `industry_router` config block from `config_builder.py` and the related flags from `local_settings.py`
  - remove the router injection from `orchestrator_v6.py`, `context_pack.py`, and `research_brief_engine.py`

### 2026-03-22 04:47
- Type:
  - `feature`
  - `bugfix`
- Scope:
  - `industry_router_source_fetch`
  - `research_data_contracts`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\context_pack.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\source_ingest.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\runtime.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\industry_router\source_contracts.json`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Upgraded `source_contracts.json` from simple source-name lists into structured contracts with `mode`, `url`, keyword lists, and weights.
  - Added `industry_router.source_fetch` runtime config and local settings controls:
    - `INDUSTRY_ROUTER_ENABLE_SOURCE_FETCH`
    - `INDUSTRY_ROUTER_SOURCE_FETCH_TIMEOUT_SECONDS`
    - `INDUSTRY_ROUTER_SOURCE_FETCH_CACHE_HOURS`
    - `INDUSTRY_ROUTER_SOURCE_FETCH_MAX_SOURCES_PER_RUN`
  - Wired a dedicated official-source sidecar into `industry_router` so formal runs now fetch mechanism-specific official pages and emit:
    - `source_state_daily.csv`
    - `source_snapshots\source_snapshot_index.json`
    - `source_snapshots\source_snapshot_items.json`
  - Merged source-side state into stock scoring as a separate `source_state_score` channel while keeping event impact primarily on event mapping / industry-event rows.
  - Extended context payload and summary output so operators can see per-mechanism source coverage, source counts, and top contributing official pages.
  - Fixed two runtime bugs during integration:
    - added a certificate-verification fallback for a subset of official China government sites rejected by this environment
    - tightened publish-date extraction so PBC pages no longer pick the wrong date from unrelated page text
- Impact:
  - All three mechanism groups now have actual official-source coverage inside the runnable Phase 1 path, not just placeholder source names.
  - `industry_router_only` and upstream planning modes now expose source-aware mechanism state to operators and to the research context pack.
  - Source influence is additive and bounded; it does not replace event scoring or the existing event-stock mapping contract.
- Validation:
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\industry_router\\source_ingest.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\industry_router\\runtime.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\config_builder.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\context_pack.py`
  - `python tools\\preflight_check.py --profile quick_test --mode industry_router_only`
  - `python launch_canonical.py --preflight-only --profile quick_test --mode industry_router_only`
  - `python launch_canonical.py --profile quick_test --mode industry_router_only --skip-preflight`
  - Confirmed latest successful source-aware run at `2026-03-22 04:47:04`.
  - Confirmed current source-fetch result:
    - `source_snapshot_ok_count = 9`
    - `source_snapshot_error_count = 0`
    - `source_state_rows = 3`
  - No full integrated pipeline or full-cycle validation was executed.
- Compatibility:
  - Backward compatible at the operator-entry level.
  - Old Phase 1 artifacts remain, but `source_contracts.json` schema is now materially richer and should be treated as the new source-of-truth contract.
  - Existing event ingestion, V5 runtime, release layer, and execution layer were not rewritten.
- Rollback:
  - Set `INDUSTRY_ROUTER_ENABLE_SOURCE_FETCH = False` to disable the official-source sidecar without removing the rest of the industry router.
  - Restore the old simpler `source_contracts.json` if only name-level placeholders are desired again.
  - Remove `source_ingest.py` integration from `industry_router/runtime.py` if source-state scoring needs to be fully backed out.

### 2026-03-22 14:03
- Type:
  - `bugfix`
  - `runtime`
- Scope:
  - `validation_pass`
  - `config_write_race`
  - `research_only_label_truth`
- Files:
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Performed a segmented validation sweep across the current runtime modes instead of a single hours-long full integrated run.
  - Confirmed all key modes pass `preflight_check.py` under `quick_test`:
    - `integrated_supervisor`
    - `research_only`
    - `release_only`
    - `execution_only`
    - `resume_downstream`
    - `ingest_only`
    - `extract_only`
    - `industry_router_only`
    - `gap_only`
    - `plan_only`
    - `bridge_only`
    - `full_cycle`
  - Fixed a real runtime race where concurrent launcher invocations could truncate `hub_config.v6.runtime.<profile>.json` mid-read.
    - root cause: direct `write_text(...)` on the shared runtime config path
    - fix: switched to temp-file + `os.replace(...)` atomic writes in both `main_research_runner.py` and `hub_v6/config_builder.py`
  - Fixed a truth-label bug where `research_only` internally reused `run_integrated_supervisor(...)` and therefore wrote:
    - `supervisor_state.json.run_mode = integrated_supervisor`
    - release `source_mode = integrated_supervisor`
    even though the operator entry was `research_only`
  - `run_integrated_supervisor(...)` now accepts explicit mode/source labels so `research_only` can preserve correct operator truth.
- Impact:
  - Manual operator runs and the always-on clock path are less likely to collide on a half-written runtime config file.
  - Downstream state files and release manifests are now more trustworthy for future AI handoff and operator debugging when the entry mode is `research_only`.
  - This change does not alter strategy logic, model logic, or execution rules.
- Validation:
  - `python -m py_compile main_research_runner.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\config_builder.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\supervisor.py`
  - Actual segmented runs completed:
    - `python launch_canonical.py --profile quick_test --mode full_cycle --skip-preflight`
    - `python launch_canonical.py --profile quick_test --mode release_only --skip-preflight`
    - `python launch_canonical.py --profile quick_test --mode execution_only --gate-only --skip-preflight`
    - `python launch_canonical.py --profile quick_test --mode execution_only --execution-mode simulation --gate-only --skip-preflight`
    - `python launch_canonical.py --profile quick_test --mode resume_downstream --skip-preflight`
  - `full_cycle` completed in logs and reached bridge artifact generation; the terminal wrapper itself hit the Codex tool timeout before it could print the final epilogue, so log inspection was used to verify completion.
  - `research_only` was allowed to run until V5 candidate workspaces were actively updating under `data\research_hub_v5_1_gpu_integrated\cycles\cycle_001_20260322_134321\`; it was then manually stopped to avoid an unbounded long validation run.
  - No single full end-to-end integrated run was left running to completion in this session.
- Compatibility:
  - Backward compatible for normal single-operator usage.
  - The new atomic write behavior changes only how runtime configs hit disk, not their schema.
  - `research_only` now records truer mode labels; any downstream tooling that previously assumed `research_only` would masquerade as `integrated_supervisor` should be considered stale.
- Rollback:
  - Revert the atomic write helpers in `main_research_runner.py` and `hub_v6/config_builder.py` if direct writes are ever intentionally preferred again.
  - Revert the `run_integrated_supervisor(...)` signature change and the `run_research_only(...)` call-site labels if you intentionally want the older `integrated_supervisor` labels back.

### 2026-03-22 14:36
- Type:
  - `feature`
  - `runtime`
  - `ops`
- Scope:
  - `execution`
  - `infra`
- Files:
  - `F:\quant_data\Ashare\trade_clock_service.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_bridge_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_manager.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\clock_supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\safety_guard.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\run_gmtrade_health_probe.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\health_probe.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\brokers\gmtrade_sim_broker.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\portfolio_control.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added an execution-side safety layer centered on:
    - `data\trade_clock\system_safety_state.json`
    - `data\trade_clock\incident_log.jsonl`
    - `data\trade_clock\manual_overrides.json`
    - `data\trade_clock\latest_account_health.json`
  - Added a dedicated lightweight gmtrade health probe executed through the required `gmtrade39` Python:
    - new script `run_gmtrade_health_probe.py`
    - new module `live_execution_bridge\health_probe.py`
    - broker now exposes `load_order_health()` for normalized day-order / unfinished-order / unknown-status snapshots
  - `execution_only` now performs a formal safety preflight before any real bridge dispatch:
    - validates release artifacts and checksums
    - refreshes or reuses account/position/order health
    - reads manual overrides
    - evaluates market regime (`NORMAL / CAUTION / PANIC`)
    - enforces fail-closed behavior on stale account truth, unfinished orders, unknown order states, release problems, and recent high execution-failure ratios
  - `trade_clock_service.py` remains the only resident daemon-like process, but its heartbeat is now explicitly separated from the safety truth:
    - `clock_state.json` is the lightweight heartbeat
    - `system_safety_state.json` is the execution-safety truth
    - `incident_log.jsonl` is the abnormal-event ledger
  - Added `reduce_only` support into the portfolio-control layer so:
    - `manual_reduce_only=true`
    - `market_safety_regime=PANIC`
    - `system_mode=DEGRADED` with degraded-reduce-only enabled
    can force sell-only execution without rewriting the order-planning chain
  - Hardened execution runtime config generation to use per-invocation generated files instead of one shared autogen path, reducing config collisions between execution runs and health probes.
- Impact:
  - The execution side now has a single, inspectable answer for “can the system still trade right now?” instead of relying on scattered gate prints.
  - Operators can freeze all new trades or force sell-only behavior by editing one file: `data\trade_clock\manual_overrides.json`.
  - The clock process stays lightweight and low-process-count while still maintaining broker freshness and incident traces.
  - Existing research, release, and strategy logic were not rewritten; the safety layer is an execution-side control envelope.
- Validation:
  - `python -m py_compile trade_clock_service.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\run_gmtrade_health_probe.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\safety_guard.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\execution_manager.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\clock_supervisor.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\execution_bridge_runner.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\config_builder.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\live_execution_bridge\\health_probe.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\live_execution_bridge\\brokers\\gmtrade_sim_broker.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\live_execution_bridge\\portfolio_control.py`
  - `python trade_clock_service.py --profile quick_test --once --skip-preflight`
  - `python launch_canonical.py --profile quick_test --mode execution_only --gate-only --skip-preflight`
  - Confirmed new artifacts are created and updated:
    - `clock_state.json`
    - `system_safety_state.json`
    - `incident_log.jsonl`
    - `manual_overrides.json`
    - `latest_account_health.json`
  - No long full integrated pipeline or real execution dispatch was run.
- Compatibility:
  - Backward compatible at the operator-entry level.
  - `execution_only` now returns an additional `safety` block in its JSON output.
  - `clock_state.json` is intentionally lighter than before and should no longer be treated as the full safety truth.
  - The initial market-regime and execution-failure thresholds are conservative and may halt trading until the operator acknowledges the condition.
- Rollback:
  - Set `ENABLE_SAFETY_LAYER = False` in `hub_v6/local_settings.py` to bypass the new safety guard while keeping the old gate/time logic.
  - Remove `run_gmtrade_health_probe.py`, `live_execution_bridge\health_probe.py`, and `hub_v6\safety_guard.py` if the safety layer needs to be fully backed out.
  - Revert `portfolio_control.py` `reduce_only` handling if sell-only execution is no longer desired.

### 2026-03-22 19:50
- Type:
  - `feature`
  - `refactor`
  - `runtime`
- Scope:
  - `research`
  - `event`
  - `infra`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\contracts\records.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\contracts\interfaces.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\mechanisms\base.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\mechanisms\trend_capex\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\mechanisms\price_inventory\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\mechanisms\macro_style\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\core\event_pipeline.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\core\runtime_engine.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\industry_router\core\backtest_engine.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Completed the first full deepening pass for the three-mechanism industry-router architecture instead of leaving it as a Phase-1 skeleton.
  - Added mechanism-owned module sets for all three groups:
    - `config.json`
    - `profile_view.py`
    - `source_ingest.py`
    - `state_builder.py`
    - `mapping_rules.py`
    - `signal_builder.py`
    - `risk_rules.py`
    - `policy.py`
  - Replaced the old generic `BaseMechanismPolicy` with a delegating policy shell that only wires shared engine calls to mechanism-specific modules.
  - Expanded `PolicyTuning` and the mechanism interface so unified runtime/backtest code can call mechanism-owned source summarization, mapping, risk, and attribution hooks.
  - Rewired `event_pipeline.build_event_stock_mapping(...)` so event-to-stock routing is now produced by mechanism mapping rules rather than one shared spillover heuristic.
  - Rewrote `core\runtime_engine.py` so the formal chain is now:
    - `stock_profile -> mechanism_state_daily -> core_variable_daily -> stock_signal_daily`
    - plus per-mechanism sidecars:
      - `<mechanism>_profile.csv`
      - `<mechanism>_state.csv`
      - `<mechanism>_core_variable.csv`
      - `<mechanism>_signal.csv`
  - Rewrote `core\backtest_engine.py` around the policy hooks and the newer calendar/signal helper modules.
  - Added attribution persistence for zero-trade candidate buckets so all three mechanisms remain visible in attribution outputs even when the latest signal date sits at the end of the available price history.
  - Fixed a real runtime bug during rollout where `signal_builder.py` used `if not event` on pandas `Series`, which raised the ambiguous truth-value error during live `industry_router_only` runs.
  - Re-tuned `price_inventory` and `macro_style` entry thresholds so both mechanisms now produce valid backtest candidates under the current seed pool instead of always stopping at `no_entry_candidates`.
- Impact:
  - The industry router is no longer a thin skeleton around one generic scoring path; each mechanism now owns its own profile/state/signal/risk/policy logic while still feeding one shared runtime and one shared backtest engine.
  - `context_pack` and `research_brief` continue to consume the same `industry_router` payload shape, so upstream planning stays compatible while the internal mechanism logic is now much richer.
  - Operators now get mechanism-sliced runtime CSVs and fuller attribution outputs, which makes debugging and later parameter refinement materially easier.
- Validation:
  - `python -m py_compile` over the full `hub_v6\industry_router` tree
  - `python tools\preflight_check.py --profile quick_test --mode industry_router_only`
  - `python launch_canonical.py --profile quick_test --mode industry_router_only --skip-preflight`
  - `python launch_canonical.py --profile quick_test --mode plan_only --skip-preflight`
  - Re-ran `industry_router_only` after threshold and attribution fixes to confirm:
    - `industry_router_summary.json.backtest.mechanism_status` is now `ok` for all three mechanisms
    - `backtest_attribution_summary.json` contains rows for `trend_capex`, `price_inventory`, and `macro_style`
  - No full integrated supervisor run or execution run was performed.
- Compatibility:
  - Backward compatible at the `context_pack` / `research_brief` consumption layer because the high-level `industry_router` payload keys remain stable.
  - Output schema is intentionally richer inside the `industry_router` artifact directory; old readers that assumed only the earlier minimal files exist may be stale.
  - The router still uses a seed pool and curated source contracts, so structural completion here should not be mistaken for full-market production readiness.
- Rollback:
  - Revert the `hub_v6\industry_router` subtree to the previous skeleton if the deeper module split needs to be backed out.
  - If a partial rollback is needed, first restore `core\runtime_engine.py`, `core\backtest_engine.py`, and `core\event_pipeline.py`, then remove the new mechanism-owned files.

### 2026-03-22 21:09
- Type:
  - `feature`
  - `runtime`
  - `ops`
- Scope:
  - `research`
  - `portfolio`
  - `execution`
  - `infra`
- Files:
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\context_pack.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\orchestrator_v6.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_recommendation.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_release.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_manager.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\supervisor.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\market_state\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\technical_confirmation\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\market_state\default.json`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\technical_confirmation\default.json`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a formal `market_state` layer with contract, feature builder, scorer, policy, and runtime entrypoint.
  - Added a formal `technical_confirmation` layer with contract, feature builder, scorer, policy, and runtime entrypoint.
  - Inserted market-state generation into the V6 chain after `industry_router` and before downstream planning consumers.
  - Extended `research_context_pack.json` so research planning now sees market regime, style bias, mechanism bias, and sizing posture.
  - Reworked portfolio recommendation so it now:
    - loads latest market-state truth
    - applies market-aware exposure and name-cap tightening
    - runs candidate-level technical confirmation
    - gates or downweights weak entries before writing `target_positions.csv`
  - Enriched release publishing so `release_manifest.json` now snapshots:
    - market-state truth
    - technical-confirmation summary
    - copied sidecar artifacts for later inspection
  - Enriched `execution_only` so it now reads market-state posture from the release first, then:
    - scales `portfolio_control.max_daily_turnover_ratio`
    - forces `reduce_only` when the release posture forbids new positions
    - returns market-state truth in gate-only and skipped outputs
  - Added new config surface for:
    - `ENABLE_MARKET_STATE_ENGINE`
    - `ENABLE_TECHNICAL_CONFIRMATION`
    - `MARKET_STATE_USE_ROUTER_BIAS`
    - `PORTFOLIO_MARKET_STATE_AWARE_SIZING`
    - `PORTFOLIO_TECHNICAL_CONFIRMATION_GATE`
    - `MARKET_STATE_CONFIG_PATH`
    - `TECHNICAL_CONFIRMATION_CONFIG_PATH`
  - Fixed a real config bug during rollout where `configs\market_state\default.json` was missing its outer closing brace and silently forced policy defaults.
- Impact:
  - The system now has a formal “market-state / capital-flow total gate” and a formal “technical confirmation gate” instead of routing everything through one stock signal table.
  - Research planning, portfolio sizing, release truth, and execution posture now share one consistent market-state contract.
  - The release layer is now richer and closer to a real producer-consumer boundary between research and execution.
  - Current local posture is conservative and can materially shrink the book on weak tape; this is expected.
- Validation:
  - `python -m py_compile main_research_runner.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\config_builder.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\context_pack.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\orchestrator_v6.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\portfolio_recommendation.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\portfolio_release.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\execution_manager.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\supervisor.py`
  - `python -m py_compile` over the new `hub_v6\market_state` and `hub_v6\technical_confirmation` trees
  - `python tools\preflight_check.py --profile quick_test --mode plan_only`
  - targeted market-state build probe using the canonical research Python
  - targeted portfolio-recommendation build probe using the canonical research Python
  - `main_research_runner.py --mode release_only --profile quick_test --config ...\hub_config.v6.runtime.quick_test.json`
  - `main_research_runner.py --mode execution_only --profile quick_test --gate-only --config ...\hub_config.v6.runtime.quick_test.json`
  - Confirmed current validated truth:
    - `market_regime = risk_off`
    - `style_bias = defensive`
    - `mechanism_bias = macro_style`
    - `risk_budget_multiplier = 0.58`
    - `turnover_multiplier = 0.55`
    - `entry_strictness = 0.68`
    - `technical_confirmation.allow_count = 1`
    - latest release id = `release_20260322_210622_0dea6fd4`
  - No full integrated pipeline or real execution dispatch was run.
- Compatibility:
  - Backward compatible at the operator-entry level.
  - Output schemas are richer:
    - `research_context_pack.json` now carries market-state truth
    - `target_positions.csv` now carries technical-confirmation and market-aware fields
    - `release_manifest.json` now carries market-state and technical-confirmation blocks
  - Existing modes still exist; the main change is that more downstream consumers now read the new shared truth layers.
- Rollback:
  - Set these to `False` in `hub_v6/local_settings.py` to neutralize most of this rollout without deleting code:
    - `ENABLE_MARKET_STATE_ENGINE`
    - `ENABLE_TECHNICAL_CONFIRMATION`
    - `PORTFOLIO_MARKET_STATE_AWARE_SIZING`
    - `PORTFOLIO_TECHNICAL_CONFIRMATION_GATE`
  - Restore `configs\market_state\default.json` and `configs\technical_confirmation\default.json` if policy thresholds need a hard reset.
  - Revert the `hub_v6\market_state\*`, `hub_v6\technical_confirmation\*`, and touched consumer modules if a full rollback is needed.

### 2026-03-22 21:16
- Type:
  - `ops`
  - `governance`
- Scope:
  - `dev_log`
  - `engineering_policy`
- Files:
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a new stable `Engineering Bias` section near the top of the handoff log.
  - Recorded the current operator preference that future Codex sessions should be somewhat more aggressive and more willing to innovate on research-side architecture instead of piling up transitional compatibility sludge.
  - Explicitly preserved the guardrails that formal entry, release contracts, safety boundaries, and the dedicated gmtrade environment should still be treated conservatively unless the user says otherwise.
- Impact:
  - Future sessions now have a written mandate to prefer cleaner unified research-side redesigns when the user is explicitly pushing the system forward.
  - This should reduce half-unified internal layering and unnecessary historical baggage during the next strategy-side build-out.
- Validation:
  - Markdown-only dev-log update; no code-path validation needed.
- Compatibility:
  - No code or runtime behavior changed.
  - This is a handoff-policy change for future Codex sessions.
- Rollback:
  - Remove or revise the `Engineering Bias` section if the operator later wants a more conservative change policy again.

### 2026-03-22 21:40
- Type:
  - `feature`
  - `runtime`
- Scope:
  - `portfolio`
  - `market_state`
  - `technical_confirmation`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\market_state\default.json`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\configs\technical_confirmation\default.json`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\technical_confirmation\core\scorer.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_recommendation.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Softened the first market-state rollout so `risk_off` is no longer treated like near-freeze posture:
    - `risk_budget_multiplier` raised from `0.58` to `0.72`
    - `turnover_multiplier` raised from `0.55` to `0.70`
    - `entry_strictness` lowered from `0.68` to `0.54`
    - `new_position_policy` changed from `tight` to `selective`
  - Softened technical-confirmation thresholds and added two new behaviors:
    - `pilot_entry_thresholds` for borderline new entries
    - `existing_position_policy` for graded `reduce_watch` instead of a flat weak-hold penalty
  - `technical_confirmation/core/scorer.py` now distinguishes:
    - hard reject
    - pilot small entry
    - wait
    - graded weak-hold reduction
  - Added post-filter reweighting in `portfolio_recommendation.py` so when filtering leaves the book far below the current regime exposure budget, remaining names are re-expanded up to a configurable exposure floor instead of staying trapped in accidental low exposure.
  - Added new local settings:
    - `PORTFOLIO_ENABLE_POST_FILTER_REWEIGHT`
    - `PORTFOLIO_MIN_EXPOSURE_FILL_RATIO`
- Impact:
  - The portfolio layer is now materially more aggressive than the first market-state/tech rollout.
  - In the latest targeted validation, final target exposure increased from `0.1415` to `0.5400` while preserving:
    - `12` names
    - single-name cap `<= 0.10`
    - market-state total-exposure cap `<= 0.72`
  - Existing weak holdings are no longer all crushed to the same `0.45` multiplier; several now retain materially larger weights when their hold-health is weak but not catastrophic.
  - The system still blocks obviously poor fresh entries; this pass was meant to remove over-suppression, not to remove discipline entirely.
- Validation:
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\config_builder.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\portfolio_recommendation.py`
  - `python -m py_compile quant_research_hub_v6_repacked_clean\\quant_research_hub_v6_repacked_clean\\hub_v6\\technical_confirmation\\core\\scorer.py`
  - Targeted module-level rebuild using the canonical research Python:
    - rebuilt market-state artifacts
    - rebuilt portfolio recommendation
    - published a fresh release
    - checked execution gate
  - Confirmed latest targeted outputs:
    - `risk_budget_multiplier = 0.72`
    - `turnover_multiplier = 0.70`
    - `entry_strictness = 0.54`
    - `new_position_policy = selective`
    - `portfolio_weight_totals.reweight_before = 0.141524`
    - `portfolio_weight_totals.reweight_after = 0.54`
    - latest release id = `release_20260322_213813_c410597a`
  - No full integrated run or real execution dispatch was performed.
- Compatibility:
  - Operator entrypoints and mode semantics are unchanged.
  - Release manifests are richer because portfolio summary now also carries post-filter reweight totals.
  - This change is intentionally behavior-changing at the portfolio-construction layer.
- Rollback:
  - Revert `configs\market_state\default.json` and `configs\technical_confirmation\default.json` to the prior stricter values if the new posture is deemed too loose.
  - Set `PORTFOLIO_ENABLE_POST_FILTER_REWEIGHT = False` to keep the softer market/tech posture but disable exposure re-expansion.
  - Revert `technical_confirmation\core\scorer.py` and `portfolio_recommendation.py` if you want to return to the first hard-gating implementation.

### 2026-03-22 22:18
- Type:
  - `feature`
  - `runtime`
  - `refactor`
- Scope:
  - `portfolio`
  - `release`
  - `execution_audit`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_v2a\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_recommendation.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_release.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\portfolio_control.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\runtime.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a formal `hub_v6\portfolio_v2a` submodule with deterministic:
    - portfolio posture engine
    - lifecycle state machine
    - admission/replacement engine
    - sidecar writer
  - Portfolio recommendation now runs V2A after market-state/technical-confirmation enrichment and before final target-book persistence.
  - V2A now emits formal sidecars under `data\portfolio_recommendation_v6\portfolio_v2a`:
    - `latest_portfolio_posture.json`
    - `latest_position_lifecycle.csv`
    - `position_lifecycle_daily.csv`
    - `admission_replacement_audit.json`
    - `portfolio_control_summary.json`
  - `target_positions.csv` is now enriched with V2A fields such as:
    - `previous_state`
    - `current_state`
    - `recommended_action`
    - `position_action_intent`
    - `size_confidence`
    - `target_weight_cap_v2a`
    - `proposal_target_weight`
  - Release publishing now snapshots the new V2A sidecars into the release directory and manifest.
  - Execution-side portfolio-control audit now carries through V2A metadata from `target_positions.csv` into `position_state_before/after_*`.
  - Added new V2A config surface:
    - `ENABLE_PORTFOLIO_V2A`
    - `PORTFOLIO_ENABLE_LIFECYCLE_STATE_MACHINE`
    - `PORTFOLIO_ENABLE_ADMISSION_REPLACEMENT`
    - `PORTFOLIO_ENABLE_SOFT_CROWDING_PENALTY`
    - `PORTFOLIO_ENABLE_RICH_PORTFOLIO_AUDIT`
    - `PORTFOLIO_V2A_PILOT_MAX_WEIGHT`
    - `PORTFOLIO_V2A_BUILD_SPEED`
    - `PORTFOLIO_V2A_TRIM_SPEED`
    - `PORTFOLIO_V2A_REPLACEMENT_IMPROVEMENT_THRESHOLD`
    - `PORTFOLIO_V2A_SOFT_CROWDING_PENALTY_STRENGTH`
  - Corrected an important boundary issue during rollout:
    - research-side V2A no longer inherits execution-side `HALT` as unconditional `reduce_only`
    - only manual halt/reduce-only or `panic` regime force hard research-side no-new-entry posture
- Impact:
  - The portfolio layer is no longer a flat rank-and-cap allocator. It now has explicit posture, per-name state, staged sizing intent, and replacement audit.
  - Research/release/execution remain on the same canonical path; V2A is an upgrade of the existing chain, not a parallel system.
  - Current validated V2A posture on the latest quick-test book is:
    - `rebalance_mode = defend`
    - `new_entry_budget = 0.05616`
    - `add_budget = 0.1584`
    - `state_counts = trim:9, hold:3, watch:3`
  - The current candidate universe is still weak, so V2A legitimately produced `new_entry_count = 0` under this posture; that is current-truth behavior, not a silent failure.
- Validation:
  - `python -m py_compile` over:
    - full `hub_v6\portfolio_v2a` tree
    - `hub_v6\portfolio_recommendation.py`
    - `hub_v6\portfolio_release.py`
    - `hub_v6\config_builder.py`
    - `live_execution_bridge\portfolio_control.py`
    - `live_execution_bridge\runtime.py`
  - Targeted module-level rebuild using the canonical research Python:
    - rebuilt market-state artifacts
    - rebuilt portfolio recommendation with V2A
    - published fresh releases
    - ran `assess_execution_gate(...)`
  - Targeted execution-side compatibility probe:
    - `load_target_positions(...)` from `target_positions.csv`
    - `plan_portfolio_control(...)` against `latest_account_state.json`
    - confirmed V2A metadata is present in execution-side position-state rows
  - Confirmed latest formal compatibility release:
    - `release_20260322_221756_f9b45604`
  - No full integrated pipeline and no real broker execution dispatch were run.
- Compatibility:
  - Operator entrypoints and split modes are unchanged.
  - `execution_only` remains release-driven and does not need to understand the full V2A internals to keep working.
  - `target_positions.csv` and release manifests are intentionally richer; downstream readers that ignore unknown columns remain compatible.
- Rollback:
  - Set `ENABLE_PORTFOLIO_V2A = False` to bypass the new research-side state/admission layer while keeping the softer market-state / technical-confirmation posture.
  - Set `PORTFOLIO_ENABLE_POST_FILTER_REWEIGHT = False` if exposure re-expansion needs to be disabled independently.
  - Revert the `hub_v6\portfolio_v2a` subtree plus the touched portfolio/release/execution-audit files for a full rollback.

### 2026-03-22 23:32
- Type:
  - `architecture`
  - `execution`
  - `oms`
- Scope:
  - `authority_refactor`
  - `broker_truth`
  - `feedback_governance`
- Files:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\oms\*`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\execution_bridge_runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\context_pack.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\orchestrator_v6.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_v2a\contracts.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_v2a\exposure_engine.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_v2a\lifecycle_engine.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\portfolio_v2a\runtime.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\brokers\gmtrade_sim_broker.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\portfolio_control.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\runtime.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\config_builder.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.example.py`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Introduced a formal narrow-but-complete OMS package under `hub_v6\oms` with:
    - contract schemas
    - path/config ownership
    - broker/account snapshot loading
    - desired-vs-actual reconciliation
    - deterministic actual-state derivation
    - intent/order/fill ledgers
    - feedback-bucket artifact emission
  - Reassigned execution-side truth ownership:
    - research/V2A now define `desired_state`
    - release freezes the contract
    - OMS defines `actual_state`
    - execution bridge is now an OMS-governed dispatch arm rather than the hidden source of lifecycle truth
  - `live_execution_bridge\runtime.py` now delegates into OMS runtime instead of maintaining a competing execution truth path.
  - `gmtrade_sim_broker.py` now exposes standardized fill rows so OMS can keep a fill ledger from broker truth.
  - Added authoritative OMS artifacts under `data\live_execution_bridge\oms_v1`:
    - `latest_actual_portfolio_state.json`
    - `desired_vs_actual_gap.csv`
    - `oms_summary.json`
    - `intent_ledger_latest.csv`
    - `order_ledger_latest.csv`
    - `fill_ledger_latest.csv`
    - `actual_state_daily.csv`
    - `control_feedback_latest.json`
    - `research_meta_feedback_latest.json`
    - `narrative_feedback_latest.json`
  - V2A continuity now prefers OMS broker-truth artifacts over previous target/lifecycle sidecars.
  - Bucket B control feedback now feeds back into V2A posture pacing.
  - Bucket C research meta feedback is now injected into `research_context_pack.json` as aggregated execution-realism context.
  - Added manual OMS overrides scaffold for:
    - frozen symbols
    - force-close intents
    - expire orders
    - force resync
- Impact:
  - The system now has a formal constitutional separation between:
    - desired research state
    - frozen release contract
    - safety/gate permission
    - OMS actual-state truth
    - broker dispatch
  - Execution-side truth is no longer implicitly buried in `portfolio_control_runs` and ad hoc account snapshots.
  - V2A can now converge toward real holdings/open intents instead of only replaying the last target book.
  - `latest_account_state.json` remains available for compatibility, but OMS artifacts are now authoritative.
- Validation:
  - `python -m py_compile` over:
    - full `hub_v6\oms` tree
    - touched `portfolio_v2a` files
    - `execution_bridge_runner.py`
    - `context_pack.py`
    - `orchestrator_v6.py`
    - `live_execution_bridge\brokers\gmtrade_sim_broker.py`
    - `live_execution_bridge\portfolio_control.py`
    - `live_execution_bridge\runtime.py`
  - Research-Python import probe confirmed:
    - `hub_v6.oms` lightweight reader path no longer pulls in `gmtrade`
    - `hub_v6.portfolio_v2a.runtime` can import with OMS reader active
  - GMTrade39 stubbed OMS probe confirmed artifact emission without placing live orders:
    - `latest_actual_portfolio_state.json`
    - `desired_vs_actual_gap.csv`
    - `oms_summary.json`
    - `intent_ledger_latest.csv`
  - `python launch_canonical.py --profile quick_test --mode execution_only --gate-only`
    - still works after the OMS refactor
    - confirmed split-path compatibility remained intact
  - No full integrated pipeline and no real broker execution dispatch were run in this session.
- Compatibility:
  - Operator entrypoints remain unchanged:
    - `launch_canonical.py`
    - `main_research_runner.py`
    - `trade_clock_service.py`
  - Split modes remain unchanged:
    - `research_only`
    - `release_only`
    - `execution_only`
  - Existing `portfolio_control_runs` artifacts still exist as compatibility audit outputs, but OMS artifacts now win on truth ownership.
- Rollback:
  - Revert the new `hub_v6\oms` package and restore `live_execution_bridge\runtime.py` to the prior direct bridge flow if the OMS layer is not wanted.
  - Set `ENABLE_OMS = False` and remove the `oms` runtime-config section if a partial rollback is needed while keeping other execution features.
  - If V2A continuity must stop using OMS truth, set `OMS_USE_BROKER_TRUTH_FOR_V2A_CONTINUITY = False`.

### 2026-03-23 00:24
- Type:
  - `feature`
  - `bugfix`
- Scope:
  - `execution`
  - `infra`
- Files:
  - `F:\quant_data\Ashare\main_research_runner.py`
  - `F:\quant_data\Ashare\RUN_PROFILES.yaml`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\oms\runtime.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\oms\audit.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\oms\core\continuity_engine.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\oms\core\exception_policy.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\oms\core\intent_manager.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\oms\validation\runner.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\brokers\gmtrade_sim_broker.py`
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\live_execution_bridge\dev_log_snapshot.py`
- Change:
  - Hardened OMS Stage2/3 around three deterministic layers: cross-session intent continuity, formal OMS manual intervention, and bounded synthetic/replay validation.
  - Added OMS operator artifacts for `latest_open_intents.json`, `latest_intent_continuity_report.json`, `session_resume_audit.json`, `cancel_replace_audit.json`, `latest_manual_intervention_state.json`, `manual_override_history.jsonl`, `gap_control_metrics_daily.csv`, `mechanism_realism_rollup.csv`, and validation outputs.
  - Added broker-aware cancel requests through gmtrade `order_cancel` plumbing, first-pass replace lineage (`old_intent_id -> new_intent_id`), and richer Bucket B / Bucket C aggregation.
  - Added `oms_validate` mode to `main_research_runner.py` and `RUN_PROFILES.yaml` so OMS hardening checks can run without touching the long research chain.
  - Fixed two execution-adjacent bugs exposed during hardening: Windows path backslashes breaking dev-log snapshot regex replacement, and OMS validation importing the gmtrade stack through the old eager `snapshot_loader` path.
- Impact:
  - OMS can now resume unresolved intents across sessions, keep operator interventions auditable, emit cancel/replace lineage, and expose more actionable aggregated control/research feedback without giving up truth ownership.
  - V2A posture now consumes richer OMS control metrics, but OMS still remains the only owner of broker/account/order/fill truth.
  - Operators now have two distinct override planes: trade-clock safety overrides and OMS-local continuity/repair overrides.
- Validation:
  - `python -m py_compile` on touched OMS/runtime/broker/dev-log files.
  - `python main_research_runner.py --mode oms_validate --profile quick_test` -> `9/9` synthetic scenarios passed.
  - `python launch_canonical.py --mode oms_validate --profile quick_test --preflight-only` passed.
  - Bounded fake-broker OMS runtime probe executed under `gmtrade39` and wrote latest OMS artifacts without placing live orders.
  - No long integrated pipeline or full-cycle run was executed.
- Compatibility:
  - Backward compatible with existing `execution_only` / release split.
  - `OMS` artifact schema expanded; downstream readers that only consume prior OMS summary/actual-state files remain usable, but new fields/sidecars now exist.
  - `data\trade_clock\manual_overrides.json` remains the global safety override file; the new OMS operator file is `data\live_execution_bridge\oms_v1\manual_overrides.json`.
- Rollback:
  - Revert the touched OMS/runtime/broker/dev-log files and remove `oms_validate` from `main_research_runner.py` / `RUN_PROFILES.yaml`.
  - To disable broker-side cancel requests without rollback, set `OMS_ENABLE_BROKER_CANCEL = False`.

### 2026-03-23 00:58
- Type:
  - `docs`
- Scope:
  - `operator`
  - `handoff`
- Files:
  - `F:\quant_data\Ashare\SYSTEM_DAILY_USAGE_GUIDE_CN.txt`
  - `F:\quant_data\Ashare\CODEX_DEV_LOG.md`
- Change:
  - Added a root-level plain-text operator guide that explains the system in everyday language instead of code-first language.
  - The guide now explains:
    - what the four-layer system is
    - which entrypoints matter
    - which mode/profile to use for daily work
    - where to edit defaults and account settings
    - where to inspect release / clock / safety / OMS truth
    - how to distinguish trade-clock overrides from OMS overrides
  - Added a stable pointer in `Latest Stable Snapshot` so future operators and Codex sessions can find the guide quickly.
- Impact:
  - Daily operation no longer depends on reverse-reading code or reconstructing mode meaning from scattered logs.
  - New collaborators can now start from one plain-text guide before reading deeper governance or runtime files.
- Validation:
  - Cross-checked guide content against:
    - `launch_canonical.py`
    - `main_research_runner.py`
    - `hub_v6\local_settings.py`
    - current `CODEX_DEV_LOG.md` stable truth
  - No runtime code path changed.
- Compatibility:
  - Documentation-only change.
  - No effect on runtime behavior, release contracts, OMS truth ownership, or execution safety.
- Rollback:
  - Remove `SYSTEM_DAILY_USAGE_GUIDE_CN.txt` and the new pointer/log entry if the guide is no longer wanted.

