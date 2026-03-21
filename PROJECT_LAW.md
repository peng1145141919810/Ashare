# Project Law

## Purpose
This file defines the low-risk governance layer for the live A-share system.
It does not replace business logic. It defines which paths and wrappers are
formal, live, archived, experimental, or deprecated.

## Binding Rules
1. Do not replace the live business root entry by refactoring imports.
2. Do not move live runtime packages just to improve layout.
3. Do not treat archived or experimental paths as live runtime by default.
4. All governance additions must strengthen one operator path, one live runtime,
   and one run-trace chain.

## Canonical Runtime Law
- Formal operator entry:
  - `F:\quant_data\Ashare\launch_canonical.py`
- Formal trade-clock service entry:
  - `F:\quant_data\Ashare\trade_clock_service.py`
- Canonical wrapped business root:
  - `F:\quant_data\Ashare\main_research_runner.py`
- Unique live runtime root:
  - `F:\quant_data\Ashare\quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean`
- Default mode:
  - `integrated_supervisor`
- Default profile:
  - `quick_test`
- Allowed run profiles:
  - Defined only in `F:\quant_data\Ashare\RUN_PROFILES.yaml`
- Precision-style split:
  - `research_only` produces research artifacts and portfolio releases.
  - `execution_only` consumes a published release and applies trading-clock gates.
  - `trade_clock_service.py` is the lightweight always-on trigger process for execution.

## Formal Output Law
- Governance run-trace root:
  - `F:\quant_data\Ashare\outputs\canonical_runs`
- Live data/output root used by the business chain:
  - `F:\quant_data\Ashare\data`
- Formal run trace artifact:
  - `run_manifest.json`
- Rule:
  - Every formal run started from `launch_canonical.py` must have one run
    directory under `outputs\canonical_runs\<run_id>\`.

## Directory Status Definitions
- `live`
  - The current runtime code and current business outputs used by the official
    chain.
- `archive`
  - Historical code or assets kept for rollback or reference only. Not a live
    edit target.
- `experiment`
  - Samples, scratch outputs, generated candidate workspaces, or ad hoc
    research artifacts. Not a source of canonical runtime truth.
- `deprecated`
  - Paths or entrypoints kept only for transition. They must point operators to
    the current replacement.

## Current Classification
- `live`
  - `main_research_runner.py`
  - `quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean`
  - `data\`
  - `outputs\canonical_runs`
- `archive`
  - `F:\quant_data\早期实验数据\Ashare_legacy_code_20260321`
- `experiment`
  - `outputs\` except `outputs\canonical_runs`
  - `quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\examples`
- `deprecated`
  - Historical direct operator habits that bypass the governance wrapper once
    `launch_canonical.py` is in use

## Non-Goals
- No business-logic rewrite
- No import-tree rewrite
- No path cleanup that changes the live call chain
- No large directory migration
