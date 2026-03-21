# -*- coding: utf-8 -*-
"""GitHub 上传用的本地配置模板。

复制为 `local_settings.py` 后，再按本机环境填写真实路径和密钥。
"""
from __future__ import annotations

from pathlib import Path


RUN_MODE = "integrated_supervisor"  # full_cycle / integrated_supervisor
DEFAULT_RUN_PROFILE = "quick_test"  # overnight / quick_test
LOOKBACK_HOURS = 24
DOWNLOAD_HIGH_VALUE_PDF = True
MAX_PDF_FETCH_PER_RUN = 12
MAX_PRIORITY_EVENTS = 12
MAX_CNINFO_PAGES_PER_MARKET = 3
ENABLE_CNINFO = True
ENABLE_SSE = True
ENABLE_SZSE = True
ENABLE_TUSHARE_NEWS = True
ENABLE_TUSHARE_MAJOR_NEWS = True
TUSHARE_NEWS_SOURCES = ["sina", "cls", "wallstreetcn"]
TUSHARE_MAJOR_NEWS_SOURCES = ["新浪财经", "华尔街见闻", "财联社"]
TUSHARE_NEWS_MAX_SOURCES_PER_RUN = 1
TUSHARE_MAJOR_NEWS_MAX_SOURCES_PER_RUN = 3
TUSHARE_NEWS_RATE_WINDOW_SECONDS = 60
TUSHARE_NEWS_RATE_MAX_CALLS = 1
TUSHARE_MAJOR_NEWS_RATE_WINDOW_SECONDS = 3600
TUSHARE_MAJOR_NEWS_RATE_MAX_CALLS = 4
ENABLE_OPENAI_RESEARCH = True
ENABLE_DEEPSEEK_WORKER = True
ENABLE_LOCAL_OLLAMA_RESEARCH = True
ENABLE_MARKET_PIPELINE = True
ENABLE_EXECUTION_BRIDGE = False
ENABLE_DAILY_STRATEGY_FEEDBACK = True

MAX_EVENTS_PER_RUN = 18
DEEPSEEK_BATCH_SIZE = 6
LLM_MIN_EVENT_SCORE = 1.35
MAX_CONTENT_CHARS_PER_EVENT = 1800
SAVE_EXTRACT_SUMMARY = True

TOKEN_PLAN_MIN_INTERVAL_HOURS = 24
SUPERVISOR_RUN_FOREVER = False
SUPERVISOR_MAX_TICKS = 1
SUPERVISOR_SLEEP_SECONDS = 300
V5_GPU_MAX_CYCLES_PER_TICK = 8
V5_GPU_DRY_RUN = False
REQUIRE_GPU = True
STRATEGY_FEEDBACK_LOOKBACK_DAYS = 5
DEFENSIVE_DAILY_RETURN_THRESHOLD = -0.02
DEFENSIVE_THREE_DAY_RETURN_THRESHOLD = -0.03
AGGRESSIVE_DAILY_RETURN_THRESHOLD = 0.015
AGGRESSIVE_THREE_DAY_RETURN_THRESHOLD = 0.02

OVERNIGHT_V5_GPU_MAX_CYCLES_PER_TICK = 8
OVERNIGHT_TOKEN_PLAN_MIN_INTERVAL_HOURS = 24
OVERNIGHT_MAX_PDF_FETCH_PER_RUN = 12
OVERNIGHT_MAX_EVENTS_PER_RUN = 18
OVERNIGHT_DEEPSEEK_BATCH_SIZE = 6
OVERNIGHT_MAX_PRIORITY_EVENTS = 12

QUICK_TEST_V5_GPU_MAX_CYCLES_PER_TICK = 1
QUICK_TEST_TOKEN_PLAN_MIN_INTERVAL_HOURS = 24
QUICK_TEST_MAX_PDF_FETCH_PER_RUN = 2
QUICK_TEST_MAX_EVENTS_PER_RUN = 6
QUICK_TEST_DEEPSEEK_BATCH_SIZE = 2
QUICK_TEST_MAX_PRIORITY_EVENTS = 4

ENABLE_PORTFOLIO_RECOMMENDATION = True
PORTFOLIO_MAX_NAMES = 20
PORTFOLIO_SINGLE_NAME_CAP = 0.10
PORTFOLIO_TOTAL_EXPOSURE_CAP = 1.00
PORTFOLIO_SIMULATION_READY_NEED_GATE = False

HIGH_VALUE_TITLE_KEYWORDS = [
    "业绩预告", "业绩快报", "年度报告", "半年度报告", "一季度报告", "三季度报告",
    "回购", "分红", "权益分派", "增持", "减持", "定向增发", "发行股份", "可转债",
    "并购", "重组", "重大合同", "中标", "停牌", "复牌", "风险提示", "问询函", "处罚",
    "诉讼", "仲裁", "要约收购", "股份解除限售",
]

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
DATA_ROOT = REPO_ROOT / "data"

PORTFOLIO_OUTPUT_ROOT = str(DATA_ROOT / "portfolio_recommendation_v6")
TRAIN_TABLE_DIR = str(DATA_ROOT / "ml_datasets" / "train_table_v1")
HUB_OUTPUT_ROOT = str(DATA_ROOT / "ml_runs" / "research_hub_v6")
RAW_EVENT_ROOT = str(DATA_ROOT / "event_lake_v6" / "raw")
EVENT_STORE_ROOT = str(DATA_ROOT / "event_lake_v6" / "curated")
INVENTORY_ROOT = str(DATA_ROOT / "event_lake_v6" / "inventory")
RESEARCH_ROOT = str(DATA_ROOT / "event_lake_v6" / "research")
BRIDGE_ROOT = str(DATA_ROOT / "event_lake_v6" / "bridge")
MARKET_STATE_ROOT = str(DATA_ROOT / "market_state_v6")
DAILY_CACHE_ROOT = str(DATA_ROOT / "daily_cache_v6")
LOG_ROOT = str(DATA_ROOT / "event_lake_v6" / "logs")
ENRICHED_DAILY_DIR = str(DATA_ROOT / "enriched_daily_csv_qfq")
TRADABILITY_FLAGS_PATH = str(DATA_ROOT / "auxiliary_data" / "tradability_flags.csv")
HS300_DAILY_PATH = str(DATA_ROOT / "index_daily_csv" / "000300_HS300.csv")
HS300_MEMBERSHIP_HISTORY_PATH = str(DATA_ROOT / "auxiliary_data" / "hs300_membership_history.csv")
LISTING_MASTER_PATH = str(DATA_ROOT / "auxiliary_data" / "listing_date_master.csv")
STOCK_UNIVERSE_CLEAN_PATH = str(DATA_ROOT / "auxiliary_data" / "stock_universe_clean.csv")
LIVE_EXECUTION_ROOT = str(DATA_ROOT / "live_execution_bridge")
LIVE_PRICE_SNAPSHOT_PATH = str(DATA_ROOT / "live_execution_bridge" / "daily_price_snapshot.csv")
TRAIN_APPEND_LOOKBACK_ROWS = 160
TRAIN_APPEND_PREFIX = "part_live_append"
SYNC_TUSHARE_MISSING_DAYS = True

PYTHON_EXECUTABLE = r"C:\path\to\your\python.exe"
V5_PROJECT_ROOT = str(REPO_ROOT / "quant_research_hub_v5_1_gpu_integrated")
V5_HUB_OUTPUT_ROOT = str(DATA_ROOT / "research_hub_v5_1_gpu_integrated")
V5_BRIDGE_INPUT_ROOT = BRIDGE_ROOT
GMTRADE_PYTHON_EXECUTABLE = r"C:\path\to\gmtrade39\Scripts\python.exe"
GMTRADE_RUNTIME_CONFIG_TEMPLATE = str(PACKAGE_ROOT / "configs" / "gmtrade_runtime_config.example.json")
GMTRADE_RUNTIME_AUTOGEN_PATH = str(PACKAGE_ROOT / "configs" / "gmtrade_runtime_config.autogen.json")
GMTRADE_BRIDGE_SCRIPT_PATH = str(PACKAGE_ROOT / "run_gmtrade_portfolio_bridge.py")

OPENAI_MODEL = "gpt-5.4"
OPENAI_RESEARCH_FALLBACK_MODELS = ["gpt-4.1", "gpt-4o"]
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_RESEARCH_FALLBACK_MODELS = ["deepseek-chat", "deepseek-reasoner"]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TUSHARE_TOKEN_ENV = "TUSHARE_TOKEN"
TUSHARE_TOKEN = ""
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_TIMEOUT_SECONDS = 120
OLLAMA_RESEARCH_FALLBACK_MODELS = ["qwen2.5:7b"]
OLLAMA_RESEARCH_TIMEOUT_SECONDS = 90
