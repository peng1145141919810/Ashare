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

## Latest Stable Snapshot
- Snapshot date: `2026-03-21`
- Formal operator entry: `F:\quant_data\Ashare\launch_canonical.py`
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
- Current recommended commands:
  - `python F:\quant_data\Ashare\launch_canonical.py`
  - `python F:\quant_data\Ashare\launch_canonical.py --profile overnight`
  - `python F:\quant_data\Ashare\launch_canonical.py --profile quick_test`
  - `python F:\quant_data\Ashare\launch_canonical.py --mode resume_downstream --profile quick_test`
- Runtime transparency:
  - `main_research_runner.py` now prints a stage preview before dispatch.
  - `supervisor_state.json` is updated incrementally during integrated runs instead of only at the end.
  - `supervisor_state.json` now carries `current_stage`, `stages`, and `stage_history` for operator inspection.
- Latest confirmed milestone:
  - V6 research plan generation confirmed on `2026-03-21 14:09:34`
  - quick_test V5 cycle observed generating new candidates on `2026-03-21 14:39:15`
  - downstream recovery plus fresh portfolio output confirmed on `2026-03-21 15:32`
  - execution bridge connectivity confirmed on `2026-03-21 15:39:49`
- Current truth:
  - old V6 readmes pointing to `run_v6_full_cycle_real.py` are stale
  - this log is the current source of truth

## Session Start Checklist
- Read `Latest Stable Snapshot`, `Known Dangerous Operations`, and `Known Issues` before touching code.
- Confirm whether the user has explicitly allowed any long-running integrated run in the current session.
- Use `launch_canonical.py` plus the documented profile for formal operator runs.
- Use `main_research_runner.py` when you need to inspect or reason about the wrapped business chain directly.
- If a change alters current runtime truth, update the stable sections first, then append a new change-log entry.

## Run Profile Quick Reference
| Profile | Intent | V6 Plan Reuse | V5 Cycles | Use Case | Notes |
| --- | --- | --- | --- | --- | --- |
| `overnight` | full nightly research | `24h` | `8` | sleep-time full research pass | heavy nightly mode; highest runtime cost |
| `quick_test` | minimal full-chain debug | `24h` | `1` | faster debugging of the integrated chain | current code default; not a smoke test |

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

## Project Overview
- This is a complex medium-sized A-share research and execution system, not a small script project.
- The production chain is:
  1. Market data update and train-table append
  2. Raw event ingest from announcements plus Tushare news
  3. Structured event extraction
  4. Data-gap analysis
  5. V6 research brief generation
  6. V5.1 GPU iterative research
  7. Portfolio recommendation generation
  8. Gmtrade simulation execution
  9. Daily performance feedback back into the next run

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
  - `hub_v6/data_gap.py` identifies missing derived features or refresh actions.
  - `hub_v6/context_pack.py` merges extracted events, gap findings, and bridge context into `research_context_pack.json`.
- V6 planning layer:
  - `hub_v6/research_brief_engine.py` reads the context pack and produces `research_brief.json`.
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
  - `hub_v6/portfolio_recommendation.py` reads the latest valid V5 result, resolves the corresponding run directory, loads `latest_portfolio_v1.csv`, attaches price context, and writes:
    - `portfolio_recommendation.json`
    - `target_positions.csv`
    - `rebalance_orders.csv`
  - It consumes both V5 outputs and the latest market snapshot files, and also reads `performance_feedback.json` for posture overrides.
- Execution layer:
  - `live_execution_bridge/runtime.py` is the execution runtime entry used by the supervisor.
  - `live_execution_bridge/rebalance.py` converts target holdings and current account state into order intents.
  - `live_execution_bridge/brokers/gmtrade_sim_broker.py` is the real gmtrade simulation adapter currently in use.
  - The execution layer consumes `target_positions.csv` and price snapshots, then writes:
    - `execution_report_*.json`
    - `orders_*.csv`
    - `fills_*.csv`
    - `latest_account_state.json`
    - `equity_curve.csv`
- Feedback loop:
  - After execution, `hub_v6/supervisor.py` reads `equity_curve.csv` and writes `performance_feedback.json`.
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
| `market_pipeline_report.json` | market pipeline | operator / debugging | `F:\quant_data\Ashare\data\daily_cache_v6\market_pipeline_report.json` | JSON | shows data sync and train append status |
| `research_context_pack.json` | context pack builder | research brief engine / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\context_pack\research_context_pack.json` | JSON | full evidence pack |
| `research_brief.json` | V6 research planner | V5 bridge / operator | `F:\quant_data\Ashare\data\event_lake_v6\research\briefs\research_brief.json` | JSON | core planning artifact |
| `run_manifest.json` | formal governance wrapper | operator / debugging | `F:\quant_data\Ashare\outputs\canonical_runs\<run_id>\run_manifest.json` | JSON | run id, operator entry, runtime root, mode/profile, and trace metadata |
| `candidate_override.json` | V5 bridge | V5.1 runtime | `F:\quant_data\Ashare\data\event_lake_v6\bridge\candidate_override.json` | JSON | tells V5 what routes, models, labels to favor |
| `portfolio_recommendation.json` | portfolio recommendation layer | operator / execution bridge | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\portfolio_recommendation.json` | JSON | summary of selected strategy and portfolio state |
| `target_positions.csv` | portfolio recommendation layer | Gmtrade execution bridge | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\target_positions.csv` | CSV | target holdings with price fields |
| `rebalance_orders.csv` | portfolio recommendation layer | operator / execution bridge | `F:\quant_data\Ashare\data\portfolio_recommendation_v6\rebalance_orders.csv` | CSV | delta orders relative to prior holdings |
| `execution_report_*.json` | Gmtrade execution bridge | operator / supervisor feedback | `F:\quant_data\Ashare\data\live_execution_bridge\execution_report_*.json` | JSON | execution summary per run |
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
| `ENABLE_DAILY_STRATEGY_FEEDBACK` | `hub_v6/local_settings.py` | `True` | determines whether prior-day performance changes route and portfolio posture |
| `ENABLE_TUSHARE_NEWS` / `ENABLE_TUSHARE_MAJOR_NEWS` | `hub_v6/local_settings.py` | `True / True` | enables Tushare message-layer inputs |
| `TUSHARE_NEWS_MAX_SOURCES_PER_RUN` | `hub_v6/local_settings.py` | `1` | affects short-news breadth vs quota safety |
| `TUSHARE_MAJOR_NEWS_MAX_SOURCES_PER_RUN` | `hub_v6/local_settings.py` | `3` | affects major-news breadth vs quota safety |
| `V5_PROJECT_ROOT` | `hub_v6/local_settings.py` | `F:\quant_data\Ashare\quant_research_hub_v5_1_gpu_integrated` | legacy-named metadata field passed into V5 config; not the actual script launch path |
| `V5_HUB_OUTPUT_ROOT` | `hub_v6/local_settings.py` | `F:\quant_data\Ashare\data\research_hub_v5_1_gpu_integrated` | current V5 output root consumed by registry, cycle summaries, and portfolio recommendation |

## Known Issues
- OpenAI upstream network resets can still happen occasionally; the client now retries transient failures and auto-drops unsupported `reasoning.effort`.
- Tushare news can still return zero rows when upstream quota is exhausted even after local quota guarding.
- V5.1 runtime exposes sparse heartbeat artifacts while a cycle is running; operators often need to infer progress from candidate file timestamps.
- `hub_v6/local_settings.py` still contains legacy V5 naming such as `V5_PROJECT_ROOT`, which can mislead readers into thinking a root-level package is launched directly.
- The actual V5 launcher path is package-local `...\v5_gpu_runtime\run_research_hub_v5_1_local.py`; treat `project_root` inside V5 JSON as required config metadata, not launch-path truth.

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
