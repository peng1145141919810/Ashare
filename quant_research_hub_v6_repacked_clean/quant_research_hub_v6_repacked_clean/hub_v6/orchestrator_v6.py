# -*- coding: utf-8 -*-
"""V6 主控编排器。"""

from __future__ import annotations

from pathlib import Path

from .config_utils import load_config
from .context_pack import build_research_context_pack, save_research_context_pack
from .data_gap_engine import build_data_gap_report, save_data_gap_report
from .data_inventory import build_inventory_real, save_inventory
from .event_extract import extract_events_with_worker, save_event_store
from .event_ingest import ingest_events_real, refresh_market_basics
from .logging_utils import log_line
from .research_brief_engine import build_research_brief, save_research_brief
from .v5_bridge import build_research_actions, save_bridge_outputs


def run_v6_cycle(config_path: Path, mode: str = "full_cycle") -> None:
    """运行 V6 单轮编排。"""
    config = load_config(config_path=config_path)
    project_root = config_path.resolve().parent.parent
    prompt_root = project_root / "prompts"

    raw_items = []
    structured_events = []
    inventory = {}
    data_gap_report = {}
    context_pack = {}
    research_brief = {}

    if mode in {"ingest_only", "extract_only", "gap_only", "plan_only", "bridge_only", "full_cycle"}:
        refresh_market_basics(config=config)
        raw_items = ingest_events_real(config=config)

    if mode in {"extract_only", "gap_only", "plan_only", "bridge_only", "full_cycle"}:
        structured_events = extract_events_with_worker(
            config=config,
            raw_items=raw_items,
            prompt_root=prompt_root,
        )
        save_event_store(config=config, events=structured_events)

    if mode in {"gap_only", "plan_only", "bridge_only", "full_cycle"}:
        inventory = build_inventory_real(config=config)
        save_inventory(config=config, inventory=inventory)
        data_gap_report = build_data_gap_report(
            config=config,
            inventory=inventory,
            structured_events=structured_events,
        )
        save_data_gap_report(config=config, report=data_gap_report)

    if mode in {"plan_only", "bridge_only", "full_cycle"}:
        context_pack = build_research_context_pack(
            config=config,
            structured_events=structured_events,
            data_gap_report=data_gap_report,
        )
        save_research_context_pack(config=config, pack=context_pack)
        research_brief = build_research_brief(
            config=config,
            context_pack=context_pack,
            prompt_root=prompt_root,
        )
        save_research_brief(config=config, brief=research_brief)
        log_line(
            config,
            "研究计划生成完成：mode=%s provider=%s model=%s theses=%s candidate_experiments=%s"
            % (
                research_brief.get("generation_mode", "unknown"),
                research_brief.get("llm_provider", ""),
                research_brief.get("llm_model", ""),
                len(list(research_brief.get("core_theses", []) or [])),
                len(list(research_brief.get("candidate_experiments", []) or [])),
            ),
        )

    if mode in {"bridge_only", "full_cycle"}:
        actions = build_research_actions(brief=research_brief)
        save_bridge_outputs(config=config, actions=actions)
