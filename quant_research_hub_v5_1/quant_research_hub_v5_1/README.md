# quant_research_hub_v5_1

V5.1 的目标不是继续做“参数农民工 for-loop”，而是把研究中枢升级成一套**带算力意识的自动研究控制器**：

- 先诊断：Alpha 弱、过拟合、回撤过大、模型家族过窄、缺 GPU 路线。
- 再分配预算：feature / model / training / portfolio / risk / data / hybrid。
- 再生成候选：每个候选都有自己的代码工作区，不污染主库。
- 再执行：训练、预测、组合、回测、打分、入账、策略池升级。
- 再控制资源：根据模型家族和数据规模决定全量训练、限流抽样、GPU 优先、或直接跳过。

## V5.1 相比 V5 的关键变化

1. 新增正式 GPU 家族：
   - `lightgbm_gpu`
   - `xgboost_gpu`

2. 引入资源护栏：
   - `random_forest` 在大样本下不再硬跑全量。
   - `extra_trees` / `generated_family` / `hist_gbdt` 也有自己的样本预算。
   - 训练前先做预算判断，不合适就抽样或跳过。

3. 把算力成本写进评分：
   - 训练耗时越长，分数会被扣。
   - 计算成本越高，分数会被扣。
   - GPU 请求失败退回 CPU，也会扣分。

4. 保留 V5 的进度心跳：
   - 每个长阶段持续输出“仍在运行中”。
   - 写 `live_progress.json`。

## 目录

- `run_research_hub_v5_1_local.py`：PyCharm 一键运行入口。
- `hub/local_settings.py`：本地模式切换。
- `configs/hub_config.v5_1.local.json`：本地配置。
- `hub/model_families.py`：模型家族工厂（含 GPU 家族）。
- `hub/training_engine.py`：训练引擎（含资源护栏）。
- `hub/evaluator.py`：评分器（含算力成本惩罚）。
- `hub/validate.py`：本地环境校验（含 `nvidia-smi` / LightGBM / XGBoost 探测）。

## 最短启动顺序

1. 先改 `configs/hub_config.v5_1.local.json`。
2. 再改 `hub/local_settings.py`。
3. PyCharm 直接运行 `run_research_hub_v5_1_local.py`。

## local_settings 建议

### 先校验环境
```python
CONFIG_PATH = r"configs/hub_config.v5_1.local.json"
MODE = "validate_only"
DRY_RUN = True
MAX_CYCLES = 1
SLEEP_SECONDS = 0
```

### 再看计划
```python
MODE = "plan"
DRY_RUN = True
MAX_CYCLES = 1
```

### 再跑一轮真实 batch
```python
MODE = "batch"
DRY_RUN = False
MAX_CYCLES = 1
```

### 最后再进持续闭环
```python
MODE = "adaptive_research_brain"
DRY_RUN = False
MAX_CYCLES = None
SLEEP_SECONDS = 0
```

## 本地最先看的文件

- `hub_output_root/validation.json`
- `hub_output_root/controller_state.json`
- `hub_output_root/live_progress.json`
- `hub_output_root/registry/experiment_registry.csv`
- `hub_output_root/cycles/<cycle_id>/cycle_plan.json`
- `hub_output_root/cycles/<cycle_id>/cycle_summary.json`
- `hub_output_root/strategy_family/strategy_family_state.csv`
- `hub_output_root/strategy_family/deployment_gate.json`

## RTX 4000 Ada 默认思路

V5.1 已经按 Windows + NVIDIA + RTX 4000 Ada 的思路预置了：

- LightGBM 走 `device_type=gpu`
- XGBoost 走 `device=cuda`
- `random_forest` 不再允许千万级全量训练
- 大样本树模型优先走 GPU 家族

## API

OpenAI / DeepSeek 仍沿用 openai-compatible 配置，关键点：

- `api_key_env` 写环境变量名，不写明文 key。
- OpenAI：`OPENAI_API_KEY`
- DeepSeek：`DEEPSEEK_API_KEY`

## 这一版的边界

V5.1 已经能：
- 自动决定研究方向
- 自动生成候选代码
- 自动切换 GPU 家族
- 自动做资源限流和成本惩罚

但它还没做：
- 自动把候选代码合并回主库
- 自动改你旧脚本本体
- 自动实盘下单
