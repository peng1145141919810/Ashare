from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _package_root() -> Path:
    return Path(__file__).resolve().parent / "quant_research_hub_v6_repacked_clean" / "quant_research_hub_v6_repacked_clean"


PACKAGE_ROOT = _package_root()
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hub_v6 import local_settings as LS
from hub_v6.config_builder import build_runtime_config
from hub_v6.execution_manager import run_execution_only
from hub_v6.orchestrator_v6 import run_v6_cycle
from hub_v6.runtime_profiles import normalize_profile, profile_overrides
from hub_v6.supervisor import run_integrated_supervisor, run_release_only, run_research_only, run_resume_downstream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A 股一键研究/更新/执行总入口")
    parser.add_argument(
        "--mode",
        default=str(LS.RUN_MODE or "integrated_supervisor"),
        choices=[
            "integrated_supervisor",
            "research_only",
            "release_only",
            "execution_only",
            "resume_downstream",
            "oms_validate",
            "full_cycle",
            "ingest_only",
            "extract_only",
            "gap_only",
            "industry_router_only",
            "plan_only",
            "bridge_only",
        ],
        help="运行模式，默认走 integrated_supervisor",
    )
    parser.add_argument(
        "--profile",
        default=str(LS.DEFAULT_RUN_PROFILE or "overnight"),
        choices=["overnight", "daily_production", "quick_test"],
        help="overnight 为重研究，daily_production 为日常自动主链，quick_test 为轻量联调",
    )
    parser.add_argument(
        "--config",
        default="",
        help="显式指定配置文件；留空时自动生成对应 profile 的运行配置",
    )
    parser.add_argument(
        "--resume-execution",
        action="store_true",
        help="仅在 resume_downstream 模式下生效；为 True 时，持仓建议生成后继续重跑执行桥",
    )
    parser.add_argument("--release-id", default="", help="仅 execution_only 模式使用；显式指定 release_id")
    parser.add_argument("--ignore-window", action="store_true", help="仅 execution_only 模式使用；忽略交易时间窗口门禁")
    parser.add_argument("--gate-only", action="store_true", help="仅 execution_only 模式使用；只做门禁判断，不触发执行桥")
    parser.add_argument("--execution-mode", default="", choices=["", "simulation", "precision"], help="执行账户模式；留空时使用 local_settings 默认值")
    parser.add_argument("--precision-trade", default="default", choices=["default", "on", "off"], help="精准撮合模式下是否允许真实下发执行桥；default 使用 local_settings")
    parser.add_argument("--execution-namespace", default="", help="execution_only 命名空间；用于隔离 simulation / shadow 账本")
    parser.add_argument("--shadow-run", action="store_true", help="execution_only 使用 shadow-run；进入 OMS 留痕但不提交 broker 动作")
    parser.add_argument("--source-summary-path", default="", help="仅 release_only 模式使用；显式指定 portfolio_recommendation.json")
    parser.add_argument("--source-target-positions-path", default="", help="仅 release_only 模式使用；显式指定 target_positions.csv")
    parser.add_argument("--release-note", default="", help="仅 release_only 模式使用；写入 release manifest 的说明")
    parser.add_argument("--release-source-mode", default="", help="仅 release_only 模式使用；覆盖 release source_mode")
    return parser.parse_args()


def _deep_update(config: Dict[str, Any], overrides: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for section, values in overrides.items():
        bucket = dict(config.get(section, {}) or {})
        bucket.update(values)
        config[section] = bucket
    return config


def _apply_execution_runtime_overrides(
    config: Dict[str, Any],
    execution_mode: str = "",
    precision_trade: str = "default",
    execution_namespace: str = "",
    shadow_run: bool = False,
) -> Dict[str, Any]:
    policy = dict(config.get("execution_policy", {}) or {})
    if str(execution_mode).strip():
        policy["account_mode"] = str(execution_mode).strip().lower()
    if str(precision_trade).strip().lower() == "on":
        policy["precision_trade_enabled"] = True
    elif str(precision_trade).strip().lower() == "off":
        policy["precision_trade_enabled"] = False
    if str(execution_namespace).strip():
        policy["namespace"] = str(execution_namespace).strip()
    policy["shadow_run"] = bool(shadow_run)
    config["execution_policy"] = policy
    return config


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding=encoding)
    os.replace(tmp_path, path)
    return path


def _write_runtime_config(
    profile: str,
    execution_mode: str = "",
    precision_trade: str = "default",
    execution_namespace: str = "",
    shadow_run: bool = False,
) -> Path:
    resolved_profile = normalize_profile(profile)
    config = build_runtime_config()
    config = _deep_update(config, profile_overrides(resolved_profile))
    config = _apply_execution_runtime_overrides(
        config=config,
        execution_mode=execution_mode,
        precision_trade=precision_trade,
        execution_namespace=execution_namespace,
        shadow_run=shadow_run,
    )
    config["runtime_selection"] = {
        "profile": str(resolved_profile),
        "default_mode": str(getattr(LS, "RUN_MODE", "integrated_supervisor") or "integrated_supervisor"),
        "execution_mode": str(config.get("execution_policy", {}).get("account_mode", "") or ""),
        "precision_trade_enabled": bool(config.get("execution_policy", {}).get("precision_trade_enabled", False)),
        "execution_namespace": str(config.get("execution_policy", {}).get("namespace", "") or ""),
        "shadow_run": bool(config.get("execution_policy", {}).get("shadow_run", False)),
    }
    config_path = PACKAGE_ROOT / "configs" / f"hub_config.v6.runtime.{resolved_profile}.json"
    return _atomic_write_text(
        config_path,
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _effective_config_path(
    explicit_path: str,
    profile: str,
    execution_mode: str = "",
    precision_trade: str = "default",
    execution_namespace: str = "",
    shadow_run: bool = False,
) -> Path:
    if str(explicit_path).strip():
        return Path(explicit_path).resolve()
    return _write_runtime_config(
        profile,
        execution_mode=execution_mode,
        precision_trade=precision_trade,
        execution_namespace=execution_namespace,
        shadow_run=shadow_run,
    )


def _mode_stage_preview(mode: str, config: Dict[str, Any]) -> list[str]:
    if mode == "integrated_supervisor":
        stages = []
        if bool(config.get("market_pipeline", {}).get("enabled", False)):
            stages.append("市场数据流水线")
        stages.extend(["策略反馈刷新", "V6 研究计划"])
        stages.append("V5.1 GPU 研究")
        if bool(config.get("portfolio_recommendation", {}).get("enabled", False)):
            stages.append("持仓建议生成")
        if bool(config.get("execution_bridge", {}).get("enabled", False)):
            stages.append("执行桥")
        return stages
    if mode == "resume_downstream":
        stages = ["断点续跑持仓建议"]
        if bool(config.get("execution_bridge", {}).get("enabled", False)):
            stages.append("可选执行桥")
        return stages
    mapping = {
        "research_only": ["市场数据流水线", "策略反馈刷新", "V6 研究计划", "V5.1 GPU 研究", "持仓建议生成", "组合 release 发布"],
        "release_only": ["发布最新持仓建议为组合 release"],
        "execution_only": ["读取最新 release", "交易时钟门禁检查", "执行桥"],
        "oms_validate": ["OMS synthetic/replay validation harness", "状态迁移与连续性探针", "validation artifact 输出"],
        "full_cycle": ["基础表刷新与事件抓取", "事件抽取", "分行业/分机制路由", "市场状态/资金面总阀门", "数据缺口分析", "研究计划生成", "桥接产物生成"],
        "ingest_only": ["基础表刷新与事件抓取"],
        "extract_only": ["基础表刷新与事件抓取", "事件抽取"],
        "gap_only": ["基础表刷新与事件抓取", "事件抽取", "数据缺口分析"],
        "industry_router_only": ["分行业/分机制路由", "stock_signal_daily 生成", "分机制回测骨架"],
        "plan_only": ["基础表刷新与事件抓取", "事件抽取", "分行业/分机制路由", "市场状态/资金面总阀门", "数据缺口分析", "研究计划生成"],
        "bridge_only": ["基础表刷新与事件抓取", "事件抽取", "分行业/分机制路由", "市场状态/资金面总阀门", "数据缺口分析", "研究计划生成", "桥接产物生成"],
    }
    return mapping.get(mode, ["未知阶段"])


def _print_stage_preview(mode: str, config: Dict[str, Any]) -> None:
    stages = _mode_stage_preview(mode=mode, config=config)
    print("阶段预览:")
    for idx, stage in enumerate(stages, start=1):
        print(f"  {idx}. {stage}")


def _emit_result_json(payload: Dict[str, Any]) -> None:
    print("===== ASHARE RESULT JSON START =====")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("===== ASHARE RESULT JSON END =====")


def main() -> None:
    args = parse_args()
    config_path = _effective_config_path(
        args.config,
        args.profile,
        execution_mode=str(args.execution_mode).strip(),
        precision_trade=str(args.precision_trade).strip(),
        execution_namespace=str(args.execution_namespace).strip(),
        shadow_run=bool(args.shadow_run),
    )
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    print("===== ASHARE START =====")
    print("配置文件:", config_path)
    print("运行模式:", args.mode)
    print("运行档位:", args.profile)
    print("V5 cycles:", config.get("supervisor", {}).get("v5_gpu_max_cycles_per_tick"))
    print("研究计划最小间隔(小时):", config.get("supervisor", {}).get("token_plan_min_interval_hours"))
    print("执行账户模式:", config.get("execution_policy", {}).get("account_mode"))
    print("精准交易开关:", config.get("execution_policy", {}).get("precision_trade_enabled"))
    print("执行命名空间:", config.get("execution_policy", {}).get("namespace"))
    print("Shadow Run:", config.get("execution_policy", {}).get("shadow_run"))
    print("日志根目录:", config.get("paths", {}).get("log_root"))
    print("Supervisor 状态文件:", Path(str(config.get("paths", {}).get("research_root", ""))) / "supervisor" / "supervisor_state.json")
    _print_stage_preview(mode=args.mode, config=config)
    if args.mode == "integrated_supervisor":
        run_integrated_supervisor(config_path)
    elif args.mode == "research_only":
        run_research_only(config_path)
    elif args.mode == "release_only":
        release = run_release_only(
            config_path,
            source_mode=str(args.release_source_mode).strip() or "release_only",
            summary_path=str(args.source_summary_path).strip(),
            target_positions_path=str(args.source_target_positions_path).strip(),
            note=str(args.release_note).strip(),
        )
        print("最新 release:", release.get("release_id"))
        print("Trade Date:", release.get("trade_date"))
        print("Manifest:", release.get("artifacts", {}).get("manifest_path"))
        _emit_result_json(release)
    elif args.mode == "execution_only":
        result = run_execution_only(
            config_path=config_path,
            release_id=str(args.release_id).strip(),
            ignore_window=bool(args.ignore_window),
            gate_only=bool(args.gate_only),
            trigger_label="manual",
            trigger_source="main_research_runner",
        )
        _emit_result_json(result)
    elif args.mode == "oms_validate":
        from hub_v6.oms.validation import run_oms_validation_suite

        result = run_oms_validation_suite(config=config)
        _emit_result_json(result)
    elif args.mode == "resume_downstream":
        run_resume_downstream(config_path, include_execution=bool(args.resume_execution))
    else:
        run_v6_cycle(config_path=config_path, mode=args.mode)
    print("===== ASHARE DONE =====")


if __name__ == "__main__":
    main()
