# -*- coding: utf-8 -*-
"""本地 Ollama 事件预处理器。"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import requests


VALID_EVENT_TYPES = {
    "财务业绩", "分红回购", "增减持", "并购重组",
    "重大合同", "监管处罚", "停复牌", "诉讼仲裁", "其他",
}


RULE_MAP = [
    ("业绩预告", "财务业绩"),
    ("业绩快报", "财务业绩"),
    ("年度报告", "财务业绩"),
    ("半年度报告", "财务业绩"),
    ("一季度报告", "财务业绩"),
    ("三季度报告", "财务业绩"),
    ("回购", "分红回购"),
    ("分红", "分红回购"),
    ("增持", "增减持"),
    ("减持", "增减持"),
    ("收购", "并购重组"),
    ("重组", "并购重组"),
    ("中标", "重大合同"),
    ("合同", "重大合同"),
    ("处罚", "监管处罚"),
    ("问询", "监管处罚"),
    ("停牌", "停复牌"),
    ("复牌", "停复牌"),
    ("诉讼", "诉讼仲裁"),
    ("仲裁", "诉讼仲裁"),
]


def _provider_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """读取本地模型配置。

    Args:
        config: 运行配置。

    Returns:
        Dict[str, Any]: 本地模型配置。
    """
    return dict(config.get("local_ollama", {}) or {})


def build_prompt(title: str) -> str:
    """构造标题级事件抽取提示词。

    Args:
        title: 标题文本。

    Returns:
        str: 提示词。
    """
    return f"""
你是量化研究系统里的事件预处理器。
请只输出 JSON，不要输出解释，不要输出 markdown，不要输出代码块。

对下面这条标题做结构化提取：

标题：{title}

输出格式必须严格为：
{{
  "event_type": "从以下类别中选一个：财务业绩/分红回购/增减持/并购重组/重大合同/监管处罚/停复牌/诉讼仲裁/其他",
  "entity": "主体名称，未知就填空字符串",
  "importance": 0到10之间的整数，
  "summary": "一句中文摘要"
}}
""".strip()


def rule_fallback_for_title(title: str) -> Dict[str, Any]:
    """规则兜底。

    Args:
        title: 标题文本。

    Returns:
        Dict[str, Any]: 兜底结构化结果。
    """
    event_type = "其他"
    for keyword, mapped in RULE_MAP:
        if keyword in title:
            event_type = mapped
            break
    importance = 3
    if event_type in {"财务业绩", "并购重组", "重大合同", "监管处罚", "停复牌", "诉讼仲裁"}:
        importance = 7
    elif event_type in {"分红回购", "增减持"}:
        importance = 5
    return {
        "event_type": event_type,
        "entity": "",
        "importance": importance,
        "summary": title[:60],
        "extract_backend": "rule_fallback",
        "extract_ok": False,
    }


def parse_single_title(title: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """调用本地 Ollama 处理单条标题。

    Args:
        title: 标题文本。
        config: 运行配置。

    Returns:
        Dict[str, Any]: 结构化结果。
    """
    provider = _provider_cfg(config)
    base_url = str(provider.get("base_url", "http://localhost:11434")).rstrip("/")
    model = str(provider.get("model", "qwen2.5:7b") or "qwen2.5:7b")
    timeout_seconds = int(provider.get("timeout_seconds", 120) or 120)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(title)}],
        "stream": False,
        "options": {"temperature": 0},
    }

    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    content = str(data.get("message", {}).get("content", "") or "").strip()
    parsed = json.loads(content)

    event_type = str(parsed.get("event_type", "其他") or "其他")
    if event_type not in VALID_EVENT_TYPES:
        event_type = "其他"

    importance_raw = parsed.get("importance", 0)
    try:
        importance = int(importance_raw)
    except Exception:
        importance = 0
    importance = max(0, min(10, importance))

    return {
        "event_type": event_type,
        "entity": str(parsed.get("entity", "") or ""),
        "importance": importance,
        "summary": str(parsed.get("summary", title) or title)[:80],
        "extract_backend": f"ollama_{model.replace(':', '_')}",
        "extract_ok": True,
    }


def batch_parse_titles(items: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """批量处理事件标题。

    Args:
        items: 原始事件列表。
        config: 运行配置。

    Returns:
        List[Dict[str, Any]]: 补充结构化结果后的事件列表。
    """
    outputs: List[Dict[str, Any]] = []
    for item in items:
        title = str(item.get("title") or item.get("raw_title") or "").strip()
        if not title:
            enriched = dict(item)
            enriched.update(rule_fallback_for_title(title=""))
            outputs.append(enriched)
            continue
        try:
            parsed = parse_single_title(title=title, config=config)
            enriched = dict(item)
            enriched.update(parsed)
            outputs.append(enriched)
        except Exception as exc:
            enriched = dict(item)
            enriched.update(rule_fallback_for_title(title=title))
            enriched["extract_error"] = str(exc)
            outputs.append(enriched)
    return outputs
