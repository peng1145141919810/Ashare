# quant_research_hub_v6_announcement_runtime_full

这是重打后的 V6 公告驱动研究版。

## 先看哪里
1. `README_PYCHARM.md`
2. `hub_v6/local_settings.py`
3. 直接运行 `run_v6_full_cycle_real.py`

## 这版改了什么
- 不再依赖 Tushare 的 `anns_d` 单独权限。
- 公告入口改成：巨潮为主，上交所/深交所兜底。
- Tushare 继续负责基础数据、`news`、`major_news`。
- OpenAI 研究脑走 Responses API。
- 模式切换改成手改 `hub_v6/local_settings.py`，不走终端。
