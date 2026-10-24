# CD / LightGCN 复现指南

本配置运行 Amazon Reviews 2023 CDs & Vinyl、LightGCN 自然 Top-10 与 Qwen3-VL-8B-Instruct 的 Zip4Rec 流程。

## 输入和配置

从 [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) 获取该类别的 reviews 与 metadata，准备本地 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，按根目录 README 安装依赖。

复现当前已选 compact 使用 [`pipeline.json`](pipeline.json)；从原始数据运行 agent 选择再渲染使用 [`pipeline_agent.json`](pipeline_agent.json)。按本地环境修改：

| 字段 | 含义 |
| --- | --- |
| `reviews`, `metadata` | 原始 JSONL 或 JSONL.GZ 文件 |
| `base_model` | 本地基座、tokenizer、processor 目录 |
| `output` | 本次完整运行的独立输出目录 |
| `asset_model`, `asset_workers` | 物品语义/画像生成模型与请求并发数 |
| `filtering` | 用户和物品 k-core 下限 |
| `retrieval` | LightGCN 设备、训练轮数、验证与测试用户数 |
| `grpo_config` | [`grpo.json`](grpo.json) 的路径 |
| `sft_learning_rate` | SFT 学习率 |
| `layout_strategy` | 已选定代码目录；复现配置为 `optical_compression/selected`，也可填 agent 导出的 `frozen/` |
| `layout_optimizer_config` | 新数据集的一次性搜索配置；语义训练集生成后执行，与 `layout_strategy` 互斥 |

默认核心参数：user-10/item-15；LightGCN 64 维、2 层、35 epoch、学习率 1e-3、DNS=32、模型种子 7；用户组种子 603759240；最近历史长度 3。SFT 为 LoRA r=8、alpha=16、dropout=0.05、学习率 1e-5、3 epoch、梯度累积 16、种子 6062。

GRPO 参数见 `grpo.json`：学习率 1e-6、单 epoch、每次更新 4 个用户组、16 用户组旧策略窗口、每用户 8 个采样排列、温度 1、裁剪 0.2、KL=0.02、SFT 回放=0.1、固定融合系数 1.5。

## 完整入口

配置语义生成服务的 `OPENAI_BASE_URL` 与 `OPENAI_API_KEY` 环境变量。从仓库根目录运行：

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENAI_BASE_URL='https://api.openai.com/v1'
# 新数据集：先选择，再固定调用。
python run_pipeline.py --config configs/cd_lightgcn/pipeline_agent.json

# 复现已选的 compact：不重复搜索。
python run_pipeline.py --config configs/cd_lightgcn/pipeline.json
```

该入口执行数据筛选和五次上游 LightGCN 训练、物品描述和用户画像请求、返回校验、三类语义集导出、新数据集的一次性排版搜索或复用已选代码、固定代码的 DensePage 渲染和 processor 检查、SFT/GRPO 训练与验证选择、锁定模型的测试评分。请求会使用所配置的服务，训练使用选定 GPU。

SFT 保存步数按配对数量计算，GRPO 按训练组数和四分位比例保存。参考规模对应 SFT 158/316/474 和 GRPO 79/158/237/315；其他数据规模会得到自己的检查点。

## 分模块运行

### 1. 数据与语义

详见 [模块一](../../semantic_context/README.md)。三个阶段分别准备原始候选和物品请求、校验物品返回并生成画像请求、校验画像返回并导出语义训练/验证/测试集：

```bash
python -m semantic_context.pipeline prepare --reviews data/raw/CDs_and_Vinyl.jsonl.gz --metadata data/raw/meta_CDs_and_Vinyl.jsonl.gz --output artifacts/cd_prepare --device cuda
python -m semantic_context.run_requests sync --requests artifacts/cd_prepare/item_requests/requests.jsonl --output artifacts/cd_prepare/item_results.jsonl
python -m semantic_context.pipeline finish-items --work artifacts/cd_prepare --results artifacts/cd_prepare/item_results.jsonl
python -m semantic_context.run_requests sync --requests artifacts/cd_prepare/profile_requests/requests.jsonl --output artifacts/cd_prepare/profile_results.jsonl
python -m semantic_context.pipeline finish-profiles --work artifacts/cd_prepare --results artifacts/cd_prepare/profile_results.jsonl
```

默认使用 gpt-4o，`prepare --asset-model` 可指定生成模型。请求、返回、schema、版本和内容哈希会保存到运行目录。校验失败时检查相应报告，使用请求执行器的 `retry-input` 构造重试请求；不要将失败或缺失的画像当作有效样本。

### 2. DensePage

先用 [`layout_optimizer.json`](layout_optimizer.json) 控制字体、视觉模型和搜索预算，运行一次 agent：

```bash
python -m optical_compression.optimize --config configs/cd_lightgcn/layout_optimizer.json --samples artifacts/cd_prepare/datasets/train_candidates.jsonl --model data/models/Qwen3-VL-8B-Instruct --output artifacts/layout_cd_search
```

完成后只加载选中代码，不再调用 agent：

```bash
python -m optical_compression.pipeline --datasets artifacts/cd_prepare/datasets --layout artifacts/layout_cd_search/frozen --model data/models/Qwen3-VL-8B-Instruct --output artifacts/cd
```

复现本项目已选的 compact 时，跳过上面的选择步骤，把 `--layout` 换为 `optical_compression/selected`。代码位于 [`selected/compact.py`](../../optical_compression/selected/compact.py)，清单标注其为当前选中的排版代码。

渲染阶段对三个划分的全部候选生成页面、组装输入、检查实际 token 和 2048 长度上限，不加载 VLM 权重、不请求 API。冻结的 agent 策略还会逐条检查严格少于纯文本。参见 [模块二](../../optical_compression/README.md)。

### 3. SFT、GRPO 和评估

```bash
python -m ranking.pipeline --data artifacts/cd --datasets artifacts/cd_prepare/datasets --model data/models/Qwen3-VL-8B-Instruct --grpo-config configs/cd_lightgcn/grpo.json --output runs/cd_complete
```

SFT 从基座新建语言 LoRA。验证集按 NDCG@10、MRR@10、H@1 降序及更早步数选择检查点，GRPO 从选中 SFT 初始化并始终以其为冻结参考。两个阶段都完成验证选择后写入 `test_plan.json`，才对锁定模型进行测试。参见 [模块三](../../ranking/README.md)。

## 输出与结果

复现配置输出在 `artifacts/zip4rec_cd_full/`，搜索配置输出在 `artifacts/zip4rec_cd_agent/`；模块级入口使用上述各自的输出目录。

```text
semantic/
  filtered/                 # 筛选交互与清洗元数据
  sources/                  # 冻结目标、候选、历史和画像上下文
  items/, profiles/         # 生成且通过校验的物品语义与用户画像
  datasets/                 # 语义组、SFT 配对、GRPO 候选、全量评估用户
layout_search/             # 搜索模式产生；记录及 frozen/ 选中代码
densepage/                 # 固定代码生成的 PNG、图文数据与实际 token 统计
ranking/
   sft/, grpo/              # 本次训练的模型及参考缓存
   validation/              # 候选检查点的验证分数
   sft_selection.json, grpo_selection.json
   test_plan.json
   test/                    # 分数、逐用户排序、条件/整体指标
   results.json             # SFT、GRPO、LightGCN 原序参考结果
```

验证与测试 cohort 互斥，训练可使用评估用户更早的事件。训练候选严格屏蔽所有未来正反馈及最终外部目标。评估只改变基础模型自然 Top-10 内的顺序，不注入目标；整体指标使用完整用户数，命中子集指标单独报告。

`workflow_state.json` 保存阶段状态、命令和产物哈希；相同配置重跑可复用已完成且未变化的阶段。SFT 使用 `ranking.train_sft --resume-from-checkpoint`，GRPO 使用 `ranking.train_grpo --resume` 恢复已保存检查点。未保存检查点的失败阶段不能据此声称完整恢复。

## 参考结果与验证范围

原 CD 配置：1,258 个训练组、2,516 条 SFT 配对、101 名验证命中用户、108 名测试命中用户、测试全集 1,000 用户。GRPO checkpoint-237、alpha=1.5 的融合条件 NDCG@10 为 0.573137954，整体 NDCG@10 为 0.061898899。

这是单种子参考结果，原测试集曾用于先前实验。公开入口尚未完整重跑全量语义生成与 GPU 训练；重新生成语义资产、API 模型版本变化、数据版本和数值环境变化都可能改变检查点选择及分数。使用本次运行的验证选择和实际用户数，不强行选用参考步数或填入参考指标。

```bash
python -m unittest discover -s tests -v
```

测试覆盖数据边界、页面构建、排序与策略数值、恢复状态及指标分母；微型输入在临时目录动态生成。
