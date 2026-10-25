# Zip4Rec

Zip4Rec 将用户画像、历史交互语义和候选物品描述压缩为 DensePage 图片，通过视觉语言模型的 SFT 和 GRPO 训练完成推荐重排序。

仓库以 **Amazon Reviews 2023 CDs & Vinyl + LightGCN + Qwen3-VL-8B-Instruct** 为完整运行示例，从原始数据生成训练集、验证集和测试集，再训练与评估模型。数据、模型权重和运行结果由使用者在本地准备或生成。

## 三个模块

| 模块 | 工作流程与代码说明 |
| --- | --- |
| [semantic_context](semantic_context/README.md) | 原始数据筛选、LightGCN 候选召回、物品描述与用户画像提取、训练/验证/测试集生成 |
| [optical_compression](optical_compression/README.md) | agent 搜索排版代码，固定选中代码，批量生成 DensePage 和图文训练输入 |
| [ranking](ranking/README.md) | SFT、GRPO、验证集选择检查点、测试集评分与重排序评估 |

```text
reviews + metadata
  → 筛选与时间划分 → LightGCN 自然 Top-10
  → 物品描述与用户画像 → 训练集 / 验证集 / 测试集
  → agent 选择排版代码 → 固定代码批量生成 DensePage
  → SFT → 验证选择 → GRPO → 验证选择 → 测试评估
```

## 环境与输入

使用 Python 3.10，准备一张支持 BF16 的 CUDA GPU。以下以 CUDA 12.4 为例安装依赖：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

准备以下内容：

1. [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) 的 CDs & Vinyl reviews、metadata，均为 JSONL 或 JSONL.GZ。字段要求见[模块一](semantic_context/README.md)。
2. 本地 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 权重、tokenizer 和 processor。
3. 用于生成物品描述、用户画像以及运行视觉 agent 的 Chat Completions 兼容服务。agent 的模型需同时支持图片输入和工具调用。

## 从原始数据运行

所有命令从仓库根目录执行。修改 [pipeline_agent.json](configs/cd_lightgcn/pipeline_agent.json) 中的数据、模型和输出路径，并在 [layout_optimizer.json](configs/cd_lightgcn/layout_optimizer.json) 中设置视觉 agent 的模型。配置字段见[配置说明](configs/cd_lightgcn/README.md)。

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENAI_BASE_URL='https://api.openai.com/v1'
export OPENAI_API_KEY='<你的 API key>'
python run_pipeline.py --config configs/cd_lightgcn/pipeline_agent.json
```

总入口依次调用三个模块：用训练样本搜索一次排版，导出 `layout_search/frozen/`，再只调用该目录内的固定代码生成三个划分的图片。

仓库中 [optical_compression/selected/compact.py](optical_compression/selected/compact.py) 已标注为选中的 CD 排版代码。直接复用时，修改 [pipeline.json](configs/cd_lightgcn/pipeline.json) 的本地路径后运行：

```bash
python run_pipeline.py --config configs/cd_lightgcn/pipeline.json
```

主要输出位于配置的 `output` 目录：

| 目录或文件 | 内容 |
| --- | --- |
| `semantic/datasets/` | 语义训练/验证/测试集、SFT 配对、十候选组和完整评估用户 |
| `layout_search/frozen/` | agent 选中的排版代码、字体与配置 |
| `densepage/` | 三个划分的图片、图文输入和实际 token 统计 |
| `ranking/` | SFT/GRPO 权重、验证选择、测试分数与排序 |
| `ranking/results.json` | LightGCN、SFT 和 GRPO 的评估结果 |

相同配置重跑会复用已完成且校验通过的阶段；改变数据或配置时使用新的输出目录。阶段执行记录在 `workflow_state.json` 中。

## 更换数据集

CD 示例要求交互表和物品表；只有交互 ID 而没有物品文本不能运行语义提取。其他数据集需先转换成[约定字段](semantic_context/README.md#输入格式)，再调整以下位置：

| 调整内容 | 位置 |
| --- | --- |
| 数据路径、k-core、召回训练轮数、验证/测试用户数 | `configs/cd_lightgcn/pipeline*.json` |
| 评分条件、正反馈定义 | `semantic_context/raw_filters.py`、`filter_data.py` |
| 物品文本清洗、领域提示词、语义字段及校验 | `extract_metadata.py`、`llm_profile_common.py`、`assets.py`（均在模块一） |
| 用户/候选文字组织与领域用语 | `semantic_context/prepare.py`、`samples.py` |
| 页面字段与推荐提示词 | `optical_compression/document.py`、`build.py`，以及 `ranking/sft_support.py`；复用 compact 时还需调整 `optical_compression/selected/compact.py` |
| 初始字体、字号、搜索预算和阅读门槛 | `configs/cd_lightgcn/layout_optimizer.json` |

当前评分筛选和语义字段以音乐为例，跨领域需要修改对应代码；修改 JSON 配置即可运行的范围是相同字段与语义约定的数据。训练入口使用 Qwen3-VL-8B 和固定十候选组。

## 辅助文件与检查

`run_pipeline.py` 是总入口，`workflow.py` 管理阶段记录和产物校验，`configs/` 保存运行参数，`tests/` 检查数据边界、渲染和排序逻辑，`optical_compression/fonts/` 提供固定字体及许可。

```bash
python -m unittest discover -s tests -v
```
