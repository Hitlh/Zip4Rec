# CD / LightGCN 运行配置

本目录配置 Amazon Reviews 2023 CDs & Vinyl、LightGCN 自然 Top-10 与 Qwen3-VL-8B-Instruct 的完整流程。

## 文件作用

| 文件 | 用途 |
| --- | --- |
| `pipeline_agent.json` | 从原始数据开始，先运行 agent 选出排版代码，再批量渲染、训练和评估 |
| `pipeline.json` | 从原始数据开始，使用已选中的 `optical_compression/selected/compact.py` 渲染、训练和评估 |
| `layout_optimizer.json` | 初始排版、视觉 agent、固定字体字号、搜索预算、token 与阅读门槛 |
| `grpo.json` | GRPO 学习率、更新规模、采样排列数、KL、SFT 回放和融合系数 |

## 总入口参数

路径均相对于执行命令时的仓库根目录，也可使用绝对路径。

| 字段 | 含义 |
| --- | --- |
| `reviews`, `metadata` | 原始交互与物品元数据，JSONL 或 JSONL.GZ |
| `base_model` | Qwen3-VL-8B-Instruct 权重、tokenizer、processor 目录 |
| `output` | 本次运行的输出目录 |
| `asset_model`, `asset_workers` | 物品描述和用户画像生成模型、请求并发数 |
| `filtering.user_k`, `filtering.item_k` | 正反馈 k-core 下限，默认 10 / 15 |
| `retrieval.device`, `retrieval.epochs` | LightGCN 设备（`cpu` / `cuda`）、训练轮数 |
| `retrieval.validation_users`, `retrieval.test_users` | 互斥验证/测试用户数，默认各 1000；小数据集须下调 |
| `layout_optimizer_config` | 一次性 agent 搜索配置路径 |
| `layout_strategy` | 已选中代码目录，可填 `optical_compression/selected` 或搜索产物 `frozen/` |
| `sft_learning_rate` | SFT 学习率，默认 `1e-5` |
| `grpo_config` | GRPO 配置路径 |

`layout_optimizer_config` 与 `layout_strategy` 只能设置一个。切换配置或输入时使用新的 `output`。

## 运行

按[根 README](../../README.md) 安装依赖、准备数据及本地模型，修改上述路径。配置 API 后，从仓库根目录选择一条命令运行：

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENAI_BASE_URL='https://api.openai.com/v1'
export OPENAI_API_KEY='<你的 API key>'

# 运行 agent 选择排版，再固定代码完成后续流程。
python run_pipeline.py --config configs/cd_lightgcn/pipeline_agent.json

# 或使用仓库中已选中的 compact。
python run_pipeline.py --config configs/cd_lightgcn/pipeline.json
```

总入口依次输出 `semantic/`、`layout_search/`（搜索模式）、`densepage/` 和 `ranking/`。最终结果为 `<output>/ranking/results.json`。

分模块命令、输入输出及各代码作用见[模块一](../../semantic_context/README.md)、[模块二](../../optical_compression/README.md)、[模块三](../../ranking/README.md)。更换数据集的调整位置见[根 README](../../README.md#更换数据集)。
