# -*- coding: utf-8 -*-
from __future__ import annotations
import json
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
            "trading_calendar_cache_path": str(getattr(LS, "TRADING_CALENDAR_CACHE_PATH", Path(LS.MARKET_STATE_ROOT) / "trading_calendar_a_share.csv")),
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
        },
        "portfolio_control": {
            "enabled": bool(getattr(LS, "ENABLE_PORTFOLIO_CONTROL", True)),
            "drift_threshold": float(getattr(LS, "PORTFOLIO_CONTROL_DRIFT_THRESHOLD", 0.005) or 0.005),
            "max_daily_turnover_ratio": float(getattr(LS, "PORTFOLIO_CONTROL_MAX_DAILY_TURNOVER_RATIO", 0.25) or 0.25),
            "enable_execution_feedback": bool(getattr(LS, "PORTFOLIO_CONTROL_ENABLE_EXECUTION_FEEDBACK", True)),
            "enable_dev_log_snapshot": bool(getattr(LS, "PORTFOLIO_CONTROL_ENABLE_DEV_LOG_SNAPSHOT", True)),
            "dev_log_top_holdings": int(getattr(LS, "PORTFOLIO_CONTROL_DEV_LOG_TOP_HOLDINGS", 8) or 8),
            "allow_odd_lot_exit": bool(getattr(LS, "PORTFOLIO_CONTROL_ALLOW_ODD_LOT_EXIT", True)),
        },
        "execution_policy": {
            "account_mode": str(getattr(LS, "EXECUTION_ACCOUNT_MODE", "simulation") or "simulation").strip().lower(),
            "precision_trade_enabled": bool(getattr(LS, "PRECISION_TRADE_ENABLED", False)),
            "allow_integrated_precision_execution": bool(getattr(LS, "ALLOW_INTEGRATED_PRECISION_EXECUTION", False)),
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
        },
        "execution_bridge": {
            "enabled": LS.ENABLE_EXECUTION_BRIDGE,
            "mode": "gmtrade_sim",
            "python_executable": LS.GMTRADE_PYTHON_EXECUTABLE,
            "config_template_path": LS.GMTRADE_RUNTIME_CONFIG_TEMPLATE,
            "autogen_config_path": LS.GMTRADE_RUNTIME_AUTOGEN_PATH,
            "script_path": LS.GMTRADE_BRIDGE_SCRIPT_PATH,
        },
    }

def save_runtime_config(config_path: Path) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(build_runtime_config(), ensure_ascii=False, indent=2), encoding='utf-8')
    return config_path
