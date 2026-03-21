# Ashare Quant Research System

当前 GitHub 仓库只承载可运行代码和治理文档，不包含大体量训练表、运行产物、密钥、本机配置。

## 当前正式入口

- 正式操作入口: `launch_canonical.py`
- 被包装的业务根入口: `main_research_runner.py`
- 默认模式: `integrated_supervisor`
- 默认 profile: `quick_test`
- 精准撮合建议工作流:
  - `research_only` -> 发布 `portfolio release`
  - `trade_clock_service.py` -> 常驻读时间并触发 `execution_only`

不要把旧文档里提到的历史入口当成现行真相。当前运行法统以 `CODEX_DEV_LOG.md` 为准。

## 仓库里有什么

- 当前 live runtime 代码
- V6 + 内嵌 V5.1 研究链
- 执行桥接代码
- 治理层文件:
  - `PROJECT_LAW.md`
  - `SYSTEM_MANIFEST.yaml`
  - `RUN_PROFILES.yaml`
  - `CHANGELOG_CANONICAL.md`
  - `CODEX_DEV_LOG.md`

## 仓库里没有什么

- `data/` 下的大体量数据
- 训练表文件
- 运行输出
- API token
- 本机 `local_settings.py`

这些内容需要按实际环境单独补齐。

## 协作者最小上手步骤

1. 克隆仓库。
2. 准备两套 Python 环境:
   - 研究主环境
   - `gmtrade39` 专用环境
3. 从 `quant_research_hub_v6_repacked_clean/quant_research_hub_v6_repacked_clean/hub_v6/local_settings.example.py` 复制出本机 `local_settings.py`。
4. 在 `local_settings.py` 里至少填写:
   - `PYTHON_EXECUTABLE`
   - `GMTRADE_PYTHON_EXECUTABLE`
   - `TUSHARE_TOKEN`
   - 需要时再填 OpenAI / DeepSeek 等密钥
5. 通过云盘补齐 `data/` 下所需训练表和运行依赖数据。
6. 先跑轻量预检:

```powershell
python launch_canonical.py --preflight-only --profile quick_test --mode integrated_supervisor
```

7. 预检通过后，再跑正式入口:

```powershell
python launch_canonical.py --profile quick_test
```

## 精准撮合分层

- 研究层:
  - `python launch_canonical.py --profile quick_test --mode research_only`
- 发布层:
  - `python launch_canonical.py --profile quick_test --mode release_only`
- 执行层:
  - `python launch_canonical.py --profile quick_test --mode execution_only --gate-only`
- 执行账户模式切换:
  - 模拟调试: `python launch_canonical.py --profile quick_test --mode execution_only --gate-only --execution-mode simulation`
  - 精准撮合巡检: `python launch_canonical.py --profile quick_test --mode execution_only --gate-only --execution-mode precision --precision-trade off`
  - 精准撮合允许交易: `python launch_canonical.py --profile quick_test --mode execution_only --execution-mode precision --precision-trade on`
- 时钟服务:
  - `python trade_clock_service.py --profile quick_test --once`
  - 开机自启动安装脚本: `scripts/install_trade_clock_autostart.ps1`
  - 当前默认建议: `precision + trade off`

## 运行和排查

- 运行总状态:
  - `data/event_lake_v6/research/supervisor/supervisor_state.json`
- 正式包装层追溯:
  - `outputs/canonical_runs/<run_id>/run_manifest.json`
- 正式输出法统:
  - `outputs/canonical_runs` 是 formal trace root
  - 其余 `outputs/` 默认不视为正式法统输出

## 给协作者的注意事项

- 本仓库已经把历史根目录 `quant_research_hub_v5*` 退役，现行 V5.1 在内嵌目录里。
- `.githooks/` 和 `scripts/` 里有一部分是维护者本地运维脚本，不是运行主链必需项。
- 不要把 `quick_test` 当成秒级 smoke test，它仍然会走真实集成链路。
