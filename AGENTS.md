# AGENTS.md

Before making changes in this repository, read `F:\quant_data\Ashare\CODEX_DEV_LOG.md`.

## Required First Steps
1. Read `CODEX_DEV_LOG.md` before touching code.
2. Treat `CODEX_DEV_LOG.md` as the current source of truth when older `README` files disagree.
3. Read these sections first: `Latest Stable Snapshot`, `Session Start Checklist`, `Known Dangerous Operations`, and `Known Issues`.
4. Check whether the user has explicitly allowed a long-running end-to-end run in the current session.
5. If you make code, config, runtime, data-path, or operational-rule changes, update `CODEX_DEV_LOG.md` before ending the turn.

## Hard Operational Rules
- Do not run the full integrated pipeline or any full-cycle validation by default.
- The user has explicitly said full validation can run for hours and freeze Codex.
- Default to lightweight checks only:
  - file inspection
  - targeted `Select-String`
  - targeted small commands
  - `python -m py_compile` on touched files
- Do not switch the Gmtrade bridge to the main Python environment. It must keep using the dedicated `gmtrade39` Python.
- Do not echo API tokens or duplicate secrets into normal user-facing output unless explicitly asked.

## Dev Log Maintenance
- `F:\quant_data\Ashare\CODEX_DEV_LOG.md` is a living handoff file, not a one-time snapshot.
- Future Codex sessions must append or revise the log when they materially change:
  - entrypoints
  - configs
  - runtime profiles
  - data sources
  - execution behavior
  - operational warnings
  - validation policy
- Future Codex sessions should also refresh the relevant stable sections when needed:
  - `Latest Stable Snapshot`
  - `Run Profile Quick Reference`
  - `Artifact Registry`
  - `Config Surface`
  - `Known Issues`
  - `Deferred Work`
  - `Decision Log`
- If current truth changes, update the relevant stable sections before appending the new historical change-log entry.
- Use the `Change Log Entry Template` in `CODEX_DEV_LOG.md` unless there is a strong reason not to.
- Each new log entry should include:
  - local date and time
  - type
  - scope
  - file path or module location
  - what changed
  - impact
  - validation
  - compatibility
  - rollback guidance when practical
- Do not leave undocumented behavioral changes in code.

## Current Canonical Entry
- Use `F:\quant_data\Ashare\main_research_runner.py` as the primary root entry unless the user asks for an older path on purpose.

## Runtime Notes
- Default mode is `integrated_supervisor`.
- Default profile is `overnight`.
- `quick_test` exists for minimal full-chain debugging.
- Tushare quotas and the execution-bridge dual-Python setup are documented in `CODEX_DEV_LOG.md`.
