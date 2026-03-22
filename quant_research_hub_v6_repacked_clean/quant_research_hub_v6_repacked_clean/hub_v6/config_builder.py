# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict
from . import local_settings as LS


def _dedupe_nonempty(items: list[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding=encoding)
    os.replace(tmp_path, path)
    return path

def build_runtime_config() -> Dict[str, Any]:
    enabled_sources = []
    if LS.ENABLE_CNINFO or LS.ENABLE_SSE or LS.ENABLE_SZSE:
        enabled_sources.append("announcements")
    if LS.ENABLE_TUSHARE_NEWS or LS.ENABLE_TUSHARE_MAJOR_NEWS:
        enabled_sources.append("news")
    ollama_event_extract_model = str(getattr(LS, "OLLAMA_EVENT_EXTRACT_MODEL", getattr(LS, "OLLAMA_MODEL", "qwen2.5:7b")) or "qwen2.5:7b")
    ollama_event_extract_timeout_seconds = int(getattr(LS, "OLLAMA_EVENT_EXTRACT_TIMEOUT_SECONDS", getattr(LS, "OLLAMA_TIMEOUT_SECONDS", 120)) or 120)
    ollama_research_model = str(getattr(LS, "OLLAMA_RESEARCH_MODEL", ollama_event_extract_model) or ollama_event_extract_model)
    ollama_research_models = _dedupe_nonempty(
        [ollama_research_model] + list(getattr(LS, "OLLAMA_RESEARCH_FALLBACK_MODELS", []) or [])
    ) or [ollama_research_model]
    ollama_research_timeout_seconds = int(getattr(LS, "OLLAMA_RESEARCH_TIMEOUT_SECONDS", getattr(LS, "OLLAMA_TIMEOUT_SECONDS", 120)) or 120)
    ollama_evidence_card_model = str(getattr(LS, "OLLAMA_EVIDENCE_CARD_MODEL", ollama_research_model) or ollama_research_model)
    ollama_review_router_model = str(getattr(LS, "OLLAMA_REVIEW_ROUTER_MODEL", ollama_event_extract_model) or ollama_event_extract_model)
    ollama_runtime_explainer_model = str(getattr(LS, "OLLAMA_RUNTIME_EXPLAINER_MODEL", ollama_event_extract_model) or ollama_event_extract_model)
    ollama_v5_review_model = str(getattr(LS, "OLLAMA_V5_REVIEW_MODEL", ollama_research_model) or ollama_research_model)
    return {
        "project_name": "quant_research_hub_v6_lean_portfolio_integrated",
        "project_root": str(LS.PROJECT_ROOT),
        "train_table_dir": LS.TRAIN_TABLE_DIR,
        "hub_output_root": LS.HUB_OUTPUT_ROOT,
        "execution": {"python_executable": LS.PYTHON_EXECUTABLE, "mode": LS.RUN_MODE, "max_cycles": 1},
        "paths": {
            "raw_event_root": LS.RAW_EVENT_ROOT,
            "event_store_root": LS.EVENT_STORE_ROOT,
            "inventory_root": LS.INVENTORY_ROOT,
            "research_root": LS.RESEARCH_ROOT,
            "bridge_root": LS.BRIDGE_ROOT,
            "market_state_root": LS.MARKET_STATE_ROOT,
            "daily_cache_root": LS.DAILY_CACHE_ROOT,
            "log_root": LS.LOG_ROOT,
            "portfolio_output_root": LS.PORTFOLIO_OUTPUT_ROOT,
            "live_execution_root": LS.LIVE_EXECUTION_ROOT,
            "trade_release_root": str(getattr(LS, "TRADE_RELEASE_ROOT", Path(LS.LIVE_EXECUTION_ROOT).parents[0] / "trade_release_v1")),
            "trade_clock_root": str(getattr(LS, "TRADE_CLOCK_ROOT", Path(LS.LIVE_EXECUTION_ROOT).parents[0] / "trade_clock")),
            "automation_runs_root": str(getattr(LS, "AUTOMATION_RUNS_ROOT", Path(LS.PROJECT_ROOT).parents[1] / "outputs" / "automation_runs")),
            "trading_calendar_cache_path": str(getattr(LS, "TRADING_CALENDAR_CACHE_PATH", Path(LS.MARKET_STATE_ROOT) / "trading_calendar_a_share.csv")),
            "industry_router_output_root": str(getattr(LS, "INDUSTRY_ROUTER_OUTPUT_ROOT", Path(LS.RESEARCH_ROOT) / "industry_router")),
            "technical_confirmation_root": str(getattr(LS, "TECHNICAL_CONFIRMATION_ROOT", Path(LS.RESEARCH_ROOT) / "technical_confirmation")),
            "oms_output_root": str(getattr(LS, "OMS_OUTPUT_ROOT", Path(LS.LIVE_EXECUTION_ROOT) / "oms_v1")),
        },
        "providers": {
            "tushare": {
                "enabled": True,
                "token_env": LS.TUSHARE_TOKEN_ENV,
                "token": LS.TUSHARE_TOKEN,
                "rate_limit_sleep_seconds": 0.8,
                "max_retry": 3,
                "retry_sleep_seconds": 2.0,
                "rate_limit_backoff_seconds": 12.0,
            },
            "deepseek_worker": {
                "enabled": LS.ENABLE_DEEPSEEK_WORKER,
                "api_key_env": LS.DEEPSEEK_API_KEY_ENV,
                "base_url": LS.DEEPSEEK_BASE_URL,
                "model": LS.DEEPSEEK_MODEL,
                "research_models": LS.DEEPSEEK_RESEARCH_FALLBACK_MODELS,
                "temperature": 0.1,
                "timeout_seconds": 90,
            },
            "openai_research": {
                "enabled": LS.ENABLE_OPENAI_RESEARCH,
                "api_key_env": LS.OPENAI_API_KEY_ENV,
                "base_url": LS.OPENAI_BASE_URL,
                "model": LS.OPENAI_MODEL,
                "fallback_models": LS.OPENAI_RESEARCH_FALLBACK_MODELS,
                "timeout_seconds": 180,
                "reasoning_effort": "medium",
                "store": False,
            },
        },
        "local_ollama": {
            "research_enabled": LS.ENABLE_LOCAL_OLLAMA_RESEARCH,
            "base_url": LS.OLLAMA_BASE_URL,
            "model": ollama_research_model,
            "research_models": ollama_research_models,
            "timeout_seconds": ollama_event_extract_timeout_seconds,
            "research_timeout_seconds": ollama_research_timeout_seconds,
            "event_extract_model": ollama_event_extract_model,
            "event_extract_timeout_seconds": ollama_event_extract_timeout_seconds,
            "evidence_card_enabled": bool(getattr(LS, "ENABLE_LOCAL_OLLAMA_EVIDENCE_CARD", True)),
            "evidence_card_model": ollama_evidence_card_model,
            "evidence_card_timeout_seconds": int(getattr(LS, "OLLAMA_EVIDENCE_CARD_TIMEOUT_SECONDS", ollama_research_timeout_seconds) or ollama_research_timeout_seconds),
            "evidence_card_max_items": int(getattr(LS, "OLLAMA_EVIDENCE_CARD_MAX_ITEMS", 2) or 2),
            "review_router_enabled": bool(getattr(LS, "ENABLE_LOCAL_OLLAMA_REVIEW_ROUTER", True)),
            "review_router_model": ollama_review_router_model,
            "review_router_timeout_seconds": int(getattr(LS, "OLLAMA_REVIEW_ROUTER_TIMEOUT_SECONDS", ollama_event_extract_timeout_seconds) or ollama_event_extract_timeout_seconds),
            "review_router_max_items": int(getattr(LS, "OLLAMA_REVIEW_ROUTER_MAX_ITEMS", 6) or 6),
            "runtime_explainer_enabled": bool(getattr(LS, "ENABLE_LOCAL_OLLAMA_RUNTIME_EXPLAINER", True)),
            "runtime_explainer_model": ollama_runtime_explainer_model,
            "runtime_explainer_timeout_seconds": int(getattr(LS, "OLLAMA_RUNTIME_EXPLAINER_TIMEOUT_SECONDS", 45) or 45),
            "runtime_explainer_stages": list(getattr(LS, "OLLAMA_RUNTIME_EXPLAINER_STAGES", ["v6_planning", "v5_gpu", "portfolio_recommendation", "execution_bridge"]) or ["v6_planning", "v5_gpu", "portfolio_recommendation", "execution_bridge"]),
            "v5_review_enabled": bool(getattr(LS, "ENABLE_LOCAL_OLLAMA_V5_REVIEW", True)),
            "v5_review_model": ollama_v5_review_model,
            "v5_review_timeout_seconds": int(getattr(LS, "OLLAMA_V5_REVIEW_TIMEOUT_SECONDS", ollama_research_timeout_seconds) or ollama_research_timeout_seconds),
        },
        "event_ingest": {
            "enabled_sources": enabled_sources,
            "lookback_hours": LS.LOOKBACK_HOURS,
            "save_raw_text": True,
            "max_single_text_chars": 40000,
            "max_pdf_fetch_per_run": LS.MAX_PDF_FETCH_PER_RUN,
            "download_pdf_for_high_value_announcements": LS.DOWNLOAD_HIGH_VALUE_PDF,
            "max_cninfo_pages_per_market": LS.MAX_CNINFO_PAGES_PER_MARKET,
            "enable_cninfo": LS.ENABLE_CNINFO,
            "enable_sse": LS.ENABLE_SSE,
            "enable_szse": LS.ENABLE_SZSE,
            "enable_tushare_news": LS.ENABLE_TUSHARE_NEWS,
            "enable_tushare_major_news": LS.ENABLE_TUSHARE_MAJOR_NEWS,
            "news_sources": LS.TUSHARE_NEWS_SOURCES,
            "major_news_sources": LS.TUSHARE_MAJOR_NEWS_SOURCES,
            "max_tushare_news_sources_per_run": LS.TUSHARE_NEWS_MAX_SOURCES_PER_RUN,
            "max_tushare_major_news_sources_per_run": LS.TUSHARE_MAJOR_NEWS_MAX_SOURCES_PER_RUN,
            "tushare_news_rate_window_seconds": LS.TUSHARE_NEWS_RATE_WINDOW_SECONDS,
            "tushare_news_rate_max_calls": LS.TUSHARE_NEWS_RATE_MAX_CALLS,
            "tushare_major_news_rate_window_seconds": LS.TUSHARE_MAJOR_NEWS_RATE_WINDOW_SECONDS,
            "tushare_major_news_rate_max_calls": LS.TUSHARE_MAJOR_NEWS_RATE_MAX_CALLS,
            "high_value_title_keywords": LS.HIGH_VALUE_TITLE_KEYWORDS,
        },
        "event_extract": {
            "enabled": True,
            "low_confidence_threshold": 0.55,
            "manual_review_threshold": 0.45,
            "max_events_per_run": LS.MAX_EVENTS_PER_RUN,
            "batch_size": LS.DEEPSEEK_BATCH_SIZE,
            "llm_min_score": LS.LLM_MIN_EVENT_SCORE,
            "max_content_chars_per_event": LS.MAX_CONTENT_CHARS_PER_EVENT,
            "save_extract_summary": LS.SAVE_EXTRACT_SUMMARY,
        },
        "data_gap_engine": {"enabled": True, "stale_hours_hard_refresh": 36, "missing_ratio_warn": 0.05, "missing_ratio_hard": 0.15, "event_trigger_recompute": True, "max_new_feature_candidates_per_day": 8},
        "research_context_pack": {"recent_event_days": 7, "max_priority_events": LS.MAX_PRIORITY_EVENTS, "include_market_state": True, "include_family_state": True, "include_data_gap_report": True},
        "market_state": {
            "enabled": bool(getattr(LS, "ENABLE_MARKET_STATE_ENGINE", True)),
            "config_path": str(getattr(LS, "MARKET_STATE_CONFIG_PATH", Path(LS.PROJECT_ROOT) / "configs" / "market_state" / "default.json")),
            "use_router_bias": bool(getattr(LS, "MARKET_STATE_USE_ROUTER_BIAS", True)),
        },
        "industry_router": {
            "enabled": bool(getattr(LS, "ENABLE_INDUSTRY_ROUTER", True)),
            "contract_root": str(getattr(LS, "INDUSTRY_ROUTER_CONTRACT_ROOT", Path(LS.PROJECT_ROOT) / "configs" / "industry_router")),
            "output_root": str(getattr(LS, "INDUSTRY_ROUTER_OUTPUT_ROOT", Path(LS.RESEARCH_ROOT) / "industry_router")),
            "history_lookback_days": int(getattr(LS, "INDUSTRY_ROUTER_HISTORY_LOOKBACK_DAYS", 14) or 14),
            "enable_context_pack": bool(getattr(LS, "INDUSTRY_ROUTER_ENABLE_CONTEXT_PACK", True)),
            "enable_backtest": bool(getattr(LS, "INDUSTRY_ROUTER_ENABLE_BACKTEST", True)),
            "source_fetch": {
                "enabled": bool(getattr(LS, "INDUSTRY_ROUTER_ENABLE_SOURCE_FETCH", True)),
                "timeout_seconds": int(getattr(LS, "INDUSTRY_ROUTER_SOURCE_FETCH_TIMEOUT_SECONDS", 8) or 8),
                "cache_hours": int(getattr(LS, "INDUSTRY_ROUTER_SOURCE_FETCH_CACHE_HOURS", 12) or 12),
                "max_sources_per_run": int(getattr(LS, "INDUSTRY_ROUTER_SOURCE_FETCH_MAX_SOURCES_PER_RUN", 9) or 9),
            },
            "backtest": {
                "horizons": list(getattr(LS, "INDUSTRY_ROUTER_BACKTEST_HORIZONS", [1, 2]) or [1, 2]),
                "top_k": int(getattr(LS, "INDUSTRY_ROUTER_BACKTEST_TOP_K", 3) or 3),
            },
        },
        "research_brain": {"enabled": True, "planning_model": "openai_research", "worker_model": "deepseek_worker"},
        "supervisor": {"token_plan_min_interval_hours": LS.TOKEN_PLAN_MIN_INTERVAL_HOURS, "run_forever": LS.SUPERVISOR_RUN_FOREVER, "max_ticks": LS.SUPERVISOR_MAX_TICKS, "sleep_seconds": LS.SUPERVISOR_SLEEP_SECONDS, "v5_gpu_max_cycles_per_tick": LS.V5_GPU_MAX_CYCLES_PER_TICK, "v5_gpu_dry_run": LS.V5_GPU_DRY_RUN, "require_gpu": LS.REQUIRE_GPU},
        "dynamic_strategy": {
            "enabled": LS.ENABLE_DAILY_STRATEGY_FEEDBACK,
            "lookback_days": LS.STRATEGY_FEEDBACK_LOOKBACK_DAYS,
            "defensive_daily_return_threshold": LS.DEFENSIVE_DAILY_RETURN_THRESHOLD,
            "defensive_three_day_return_threshold": LS.DEFENSIVE_THREE_DAY_RETURN_THRESHOLD,
            "aggressive_daily_return_threshold": LS.AGGRESSIVE_DAILY_RETURN_THRESHOLD,
            "aggressive_three_day_return_threshold": LS.AGGRESSIVE_THREE_DAY_RETURN_THRESHOLD,
        },
        "v5_gpu_runtime": {"project_root": LS.V5_PROJECT_ROOT, "hub_output_root": LS.V5_HUB_OUTPUT_ROOT, "train_table_dir": LS.TRAIN_TABLE_DIR, "bridge_input_root": LS.V5_BRIDGE_INPUT_ROOT, "python_executable": LS.PYTHON_EXECUTABLE},
        "market_pipeline": {
            "enabled": LS.ENABLE_MARKET_PIPELINE,
            "enriched_dir": LS.ENRICHED_DAILY_DIR,
            "flags_path": LS.TRADABILITY_FLAGS_PATH,
            "hs300_path": LS.HS300_DAILY_PATH,
            "hs300_membership_history_path": LS.HS300_MEMBERSHIP_HISTORY_PATH,
            "listing_master_path": LS.LISTING_MASTER_PATH,
            "stock_universe_path": LS.STOCK_UNIVERSE_CLEAN_PATH,
            "price_snapshot_path": LS.LIVE_PRICE_SNAPSHOT_PATH,
            "train_append_lookback_rows": LS.TRAIN_APPEND_LOOKBACK_ROWS,
            "train_append_prefix": LS.TRAIN_APPEND_PREFIX,
            "sync_tushare_missing_days": LS.SYNC_TUSHARE_MISSING_DAYS,
        },
        "portfolio_recommendation": {
            "enabled": LS.ENABLE_PORTFOLIO_RECOMMENDATION,
            "max_names": LS.PORTFOLIO_MAX_NAMES,
            "single_name_cap": LS.PORTFOLIO_SINGLE_NAME_CAP,
            "total_exposure_cap": LS.PORTFOLIO_TOTAL_EXPOSURE_CAP,
            "simulation_ready_need_gate": LS.PORTFOLIO_SIMULATION_READY_NEED_GATE,
            "market_state_aware_sizing": bool(getattr(LS, "PORTFOLIO_MARKET_STATE_AWARE_SIZING", True)),
            "technical_confirmation_gate": bool(getattr(LS, "PORTFOLIO_TECHNICAL_CONFIRMATION_GATE", True)),
            "enable_post_filter_reweight": bool(getattr(LS, "PORTFOLIO_ENABLE_POST_FILTER_REWEIGHT", True)),
            "min_exposure_fill_ratio": float(getattr(LS, "PORTFOLIO_MIN_EXPOSURE_FILL_RATIO", 0.75) or 0.75),
        },
        "portfolio_v2a": {
            "enabled": bool(getattr(LS, "ENABLE_PORTFOLIO_V2A", True)),
            "lifecycle_state_machine_enabled": bool(getattr(LS, "PORTFOLIO_ENABLE_LIFECYCLE_STATE_MACHINE", True)),
            "admission_replacement_enabled": bool(getattr(LS, "PORTFOLIO_ENABLE_ADMISSION_REPLACEMENT", True)),
            "soft_crowding_penalty_enabled": bool(getattr(LS, "PORTFOLIO_ENABLE_SOFT_CROWDING_PENALTY", True)),
            "rich_audit_enabled": bool(getattr(LS, "PORTFOLIO_ENABLE_RICH_PORTFOLIO_AUDIT", True)),
            "output_root": str(Path(LS.PORTFOLIO_OUTPUT_ROOT) / "portfolio_v2a"),
            "pilot_max_weight": float(getattr(LS, "PORTFOLIO_V2A_PILOT_MAX_WEIGHT", 0.04) or 0.04),
            "build_speed": float(getattr(LS, "PORTFOLIO_V2A_BUILD_SPEED", 1.25) or 1.25),
            "trim_speed": float(getattr(LS, "PORTFOLIO_V2A_TRIM_SPEED", 0.72) or 0.72),
            "replacement_improvement_threshold": float(getattr(LS, "PORTFOLIO_V2A_REPLACEMENT_IMPROVEMENT_THRESHOLD", 0.08) or 0.08),
            "soft_crowding_penalty_strength": float(getattr(LS, "PORTFOLIO_V2A_SOFT_CROWDING_PENALTY_STRENGTH", 0.08) or 0.08),
        },
        "technical_confirmation": {
            "enabled": bool(getattr(LS, "ENABLE_TECHNICAL_CONFIRMATION", True)),
            "config_path": str(getattr(LS, "TECHNICAL_CONFIRMATION_CONFIG_PATH", Path(LS.PROJECT_ROOT) / "configs" / "technical_confirmation" / "default.json")),
        },
        "portfolio_control": {
            "enabled": bool(getattr(LS, "ENABLE_PORTFOLIO_CONTROL", True)),
            "drift_threshold": float(getattr(LS, "PORTFOLIO_CONTROL_DRIFT_THRESHOLD", 0.005) or 0.005),
            "max_daily_turnover_ratio": float(getattr(LS, "PORTFOLIO_CONTROL_MAX_DAILY_TURNOVER_RATIO", 0.25) or 0.25),
            "enable_execution_feedback": bool(getattr(LS, "PORTFOLIO_CONTROL_ENABLE_EXECUTION_FEEDBACK", True)),
            "enable_dev_log_snapshot": bool(getattr(LS, "PORTFOLIO_CONTROL_ENABLE_DEV_LOG_SNAPSHOT", True)),
            "dev_log_top_holdings": int(getattr(LS, "PORTFOLIO_CONTROL_DEV_LOG_TOP_HOLDINGS", 8) or 8),
            "allow_odd_lot_exit": bool(getattr(LS, "PORTFOLIO_CONTROL_ALLOW_ODD_LOT_EXIT", True)),
            "reduce_only": False,
        },
        "oms": {
            "enabled": bool(getattr(LS, "ENABLE_OMS", True)),
            "output_root": str(getattr(LS, "OMS_OUTPUT_ROOT", Path(LS.LIVE_EXECUTION_ROOT) / "oms_v1")),
            "use_broker_truth_for_v2a_continuity": bool(getattr(LS, "OMS_USE_BROKER_TRUTH_FOR_V2A_CONTINUITY", True)),
            "intent_expiry_days": int(getattr(LS, "OMS_INTENT_EXPIRY_DAYS", 3) or 3),
            "control_feedback_lookback_runs": int(getattr(LS, "OMS_CONTROL_FEEDBACK_LOOKBACK_RUNS", 20) or 20),
            "research_meta_lookback_runs": int(getattr(LS, "OMS_RESEARCH_META_LOOKBACK_RUNS", 60) or 60),
            "compat_write_latest_account_state": bool(getattr(LS, "OMS_COMPAT_WRITE_LATEST_ACCOUNT_STATE", True)),
            "enable_broker_cancel": bool(getattr(LS, "OMS_ENABLE_BROKER_CANCEL", True)),
        },
        "safety": {
            "enabled": bool(getattr(LS, "ENABLE_SAFETY_LAYER", True)),
            "health_probe_interval_seconds": int(getattr(LS, "SAFETY_HEALTH_PROBE_INTERVAL_SECONDS", 300) or 300),
            "account_state_max_age_seconds": int(getattr(LS, "SAFETY_ACCOUNT_STATE_MAX_AGE_SECONDS", 900) or 900),
            "position_sync_max_age_seconds": int(getattr(LS, "SAFETY_POSITION_SYNC_MAX_AGE_SECONDS", 900) or 900),
            "release_max_age_seconds": int(getattr(LS, "SAFETY_RELEASE_MAX_AGE_SECONDS", 172800) or 172800),
            "fail_on_unfinished_orders": bool(getattr(LS, "SAFETY_FAIL_ON_UNFINISHED_ORDERS", True)),
            "fail_on_unknown_order_status": bool(getattr(LS, "SAFETY_FAIL_ON_UNKNOWN_ORDER_STATUS", True)),
            "degraded_reduce_only": bool(getattr(LS, "SAFETY_DEGRADED_REDUCE_ONLY", True)),
            "caution_turnover_multiplier": float(getattr(LS, "SAFETY_CAUTION_TURNOVER_MULTIPLIER", 0.5) or 0.5),
            "market_caution_mean_pct_chg": float(getattr(LS, "SAFETY_CAUTION_MARKET_MEAN_PCT_CHG", -1.0) or -1.0),
            "market_panic_mean_pct_chg": float(getattr(LS, "SAFETY_PANIC_MARKET_MEAN_PCT_CHG", -2.2) or -2.2),
            "market_caution_hs300_return_pct": float(getattr(LS, "SAFETY_CAUTION_HS300_RETURN_PCT", -1.5) or -1.5),
            "market_panic_hs300_return_pct": float(getattr(LS, "SAFETY_PANIC_HS300_RETURN_PCT", -3.0) or -3.0),
            "market_caution_limit_down_ratio": float(getattr(LS, "SAFETY_CAUTION_LIMIT_DOWN_RATIO", 0.05) or 0.05),
            "market_panic_limit_down_ratio": float(getattr(LS, "SAFETY_PANIC_LIMIT_DOWN_RATIO", 0.12) or 0.12),
            "execution_fail_ratio_degraded": float(getattr(LS, "SAFETY_EXECUTION_FAIL_RATIO_DEGRADED", 0.35) or 0.35),
            "execution_fail_ratio_halt": float(getattr(LS, "SAFETY_EXECUTION_FAIL_RATIO_HALT", 0.75) or 0.75),
            "execution_fail_min_orders": int(getattr(LS, "SAFETY_EXECUTION_FAIL_MIN_ORDERS", 3) or 3),
        },
        "execution_policy": {
            "account_mode": str(getattr(LS, "EXECUTION_ACCOUNT_MODE", "simulation") or "simulation").strip().lower(),
            "precision_trade_enabled": bool(getattr(LS, "PRECISION_TRADE_ENABLED", False)),
            "allow_integrated_precision_execution": bool(getattr(LS, "ALLOW_INTEGRATED_PRECISION_EXECUTION", False)),
            "namespace": "main",
            "shadow_run": False,
        },
        "trade_release": {
            "enabled": bool(getattr(LS, "ENABLE_TRADE_RELEASE", True)),
            "valid_after_time": str(getattr(LS, "TRADE_RELEASE_VALID_AFTER_TIME", "09:30:30") or "09:30:30"),
            "expires_at_time": str(getattr(LS, "TRADE_RELEASE_EXPIRES_AT_TIME", "15:00:00") or "15:00:00"),
            "calendar_lookback_days": int(getattr(LS, "TRADE_RELEASE_CALENDAR_LOOKBACK_DAYS", 7) or 7),
            "calendar_forward_days": int(getattr(LS, "TRADE_RELEASE_CALENDAR_FORWARD_DAYS", 45) or 45),
        },
        "trade_clock": {
            "enabled": bool(getattr(LS, "ENABLE_TRADE_CLOCK", True)),
            "timezone": str(getattr(LS, "TRADE_CLOCK_TIMEZONE", "Asia/Shanghai") or "Asia/Shanghai"),
            "poll_seconds": int(getattr(LS, "TRADE_CLOCK_POLL_SECONDS", 30) or 30),
            "execution_windows": list(
                getattr(
                    LS,
                    "TRADE_CLOCK_EXECUTION_WINDOWS",
                    [{"label": "morning_primary", "start": "09:30:30", "end": "10:00:00"}],
                )
                or [{"label": "morning_primary", "start": "09:30:30", "end": "10:00:00"}]
            ),
            "scheduler": {
                "enabled": bool(getattr(LS, "TRADE_CLOCK_SCHEDULER_ENABLED", True)),
                "profile": str(getattr(LS, "TRADE_CLOCK_SCHEDULER_PROFILE", "daily_production") or "daily_production"),
                "log_tail_lines": int(getattr(LS, "TRADE_CLOCK_SCHEDULER_LOG_TAIL_LINES", 30) or 30),
                "fallback_max_portfolio_age_hours": int(getattr(LS, "TRADE_CLOCK_FALLBACK_MAX_PORTFOLIO_AGE_HOURS", 96) or 96),
                "fallback_require_release": bool(getattr(LS, "TRADE_CLOCK_FALLBACK_REQUIRE_RELEASE", False)),
                "simulation_namespace": str(getattr(LS, "TRADE_CLOCK_SIMULATION_NAMESPACE", "simulation") or "simulation"),
                "shadow_namespace": str(getattr(LS, "TRADE_CLOCK_SHADOW_NAMESPACE", "shadow") or "shadow"),
                "simulation_execution_mode": str(getattr(LS, "TRADE_CLOCK_SIMULATION_EXECUTION_MODE", "precision") or "precision"),
                "shadow_execution_mode": str(getattr(LS, "TRADE_CLOCK_SHADOW_EXECUTION_MODE", "precision") or "precision"),
                "simulation_precision_trade": bool(getattr(LS, "TRADE_CLOCK_SIMULATION_PRECISION_TRADE", True)),
                "shadow_precision_trade": bool(getattr(LS, "TRADE_CLOCK_SHADOW_PRECISION_TRADE", True)),
                "phases": {
                    "research": {
                        "time": str(getattr(LS, "TRADE_CLOCK_PHASE_RESEARCH_TIME", "15:05:00") or "15:05:00"),
                        "timeout_minutes": int(getattr(LS, "TRADE_CLOCK_RESEARCH_TIMEOUT_MINUTES", 420) or 420),
                    },
                    "release": {
                        "time": str(getattr(LS, "TRADE_CLOCK_PHASE_RELEASE_TIME", "15:10:00") or "15:10:00"),
                        "timeout_minutes": int(getattr(LS, "TRADE_CLOCK_RELEASE_TIMEOUT_MINUTES", 30) or 30),
                    },
                    "preopen_gate": {
                        "time": str(getattr(LS, "TRADE_CLOCK_PHASE_PREOPEN_GATE_TIME", "09:20:00") or "09:20:00"),
                        "timeout_minutes": int(getattr(LS, "TRADE_CLOCK_PREOPEN_GATE_TIMEOUT_MINUTES", 15) or 15),
                    },
                    "simulation": {
                        "time": str(getattr(LS, "TRADE_CLOCK_PHASE_SIMULATION_TIME", "09:28:00") or "09:28:00"),
                        "timeout_minutes": int(getattr(LS, "TRADE_CLOCK_SIMULATION_TIMEOUT_MINUTES", 45) or 45),
                    },
                    "shadow": {
                        "time": str(getattr(LS, "TRADE_CLOCK_PHASE_SHADOW_TIME", "09:35:00") or "09:35:00"),
                        "timeout_minutes": int(getattr(LS, "TRADE_CLOCK_SHADOW_TIMEOUT_MINUTES", 30) or 30),
                    },
                    "summary": {
                        "time": str(getattr(LS, "TRADE_CLOCK_PHASE_SUMMARY_TIME", "15:20:00") or "15:20:00"),
                        "timeout_minutes": int(getattr(LS, "TRADE_CLOCK_SUMMARY_TIMEOUT_MINUTES", 20) or 20),
                    },
                },
            },
        },
        "execution_bridge": {
            "enabled": LS.ENABLE_EXECUTION_BRIDGE,
            "mode": "gmtrade_sim",
            "python_executable": LS.GMTRADE_PYTHON_EXECUTABLE,
            "config_template_path": LS.GMTRADE_RUNTIME_CONFIG_TEMPLATE,
            "autogen_config_path": LS.GMTRADE_RUNTIME_AUTOGEN_PATH,
            "script_path": LS.GMTRADE_BRIDGE_SCRIPT_PATH,
            "health_probe_script_path": LS.GMTRADE_HEALTH_PROBE_SCRIPT_PATH,
        },
    }

def save_runtime_config(config_path: Path) -> Path:
    return _atomic_write_text(
        config_path,
        json.dumps(build_runtime_config(), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
