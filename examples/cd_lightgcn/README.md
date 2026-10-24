# CD / LightGCN 示例

示例对应 Amazon Reviews 2023 CDs & Vinyl、LightGCN 自然 Top-10 候选、Qwen3-VL-8B-Instruct、语言 LoRA、SFT 后 GRPO 的实验分支。

## 1. 安装与无模型演示

从仓库根目录运行。CPU 演示只需要 Pillow；模型阶段另需安装兼容显卡的 PyTorch 与 `requirements.txt` 中其余依赖。

```bash
python -m pip install Pillow
python -m semantic_context.prepare --groups examples/cd_lightgcn/train_groups.jsonl --item-summaries examples/cd_lightgcn/item_summaries.jsonl --user-profiles examples/cd_lightgcn/user_profiles.jsonl --output artifacts/demo/train_context.jsonl
python -m optical_compression.build --groups artifacts/demo/train_context.jsonl --item-summaries examples/cd_lightgcn/item_summaries.jsonl --output artifacts/demo/train
python -m unittest discover -s tests -v
```

随仓库提供的三组 `*_groups.jsonl`、画像及物品语义全部是合成数据，只用于检验接口，不能得到论文指标。将 `train` 替换为 `valid` 或 `test` 可分别构建其他集合。每个输出目录必须是新目录。

## 2. 准备真实输入

真实数据按模块一的 schema 准备，冻结用户互斥的 train / valid / test，元数据与画像仅使用可用历史。每个划分分别运行语义整理和渲染，将结果放在 `artifacts/cd/train/`、`artifacts/cd/valid/`、`artifacts/cd/test/`。进入渲染的是自然命中用户；保存筛选前全部用户的数量。上游自然候选不能注入目标。

若已有原实验准备好的逐候选 JSONL 与 PNG，应直接使用这些冻结输入，并重映射图像路径；精确复现历史输入不需要重新生成画像。`semantic_context.prepare` 为可移植字段组织入口，重新格式化画像后不保证与原始自然语言画像字符串完全一致。

基础模型放在本地 `data/models/Qwen3-VL-8B-Instruct`。模型及真实数据不随代码发布。

## 3. SFT

```bash
export CUDA_VISIBLE_DEVICES=0
python -m ranking.train_sft --model data/models/Qwen3-VL-8B-Instruct --train-data artifacts/cd/train/sft_pairs.jsonl --output-dir runs/cd_sft --epochs 3 --learning-rate 1e-5 --batch-size 1 --grad-acc-steps 16 --save-steps 158 --save-total-limit 4 --seed 6062
```

参考实验共 2,516 条配对样本、1,258 个训练用户组；3 个 epoch 共 474 更新。验证集固定 alpha=1.5 选择 checkpoint-316。对 checkpoint-158/316/474 分别运行下文 `ranking.score`，再用 `ranking.select`；不得按测试集重选。`--preflight-only --preflight-scan-all` 可提前检查 processor 输入长度，但仍需安装 GPU 运行依赖。

## 4. GRPO

```bash
python -m ranking.make_run --config examples/cd_lightgcn/grpo.json --base-model data/models/Qwen3-VL-8B-Instruct --initial-adapter runs/cd_sft/checkpoint-316 --train artifacts/cd/train/candidates.jsonl --valid artifacts/cd/valid/candidates.jsonl --output runs/cd_grpo
python -m ranking.prepare_reference --run runs/cd_grpo
python -m ranking.train_grpo --run runs/cd_grpo
```

参考配置为学习率 1e-6、单 epoch、每次更新 4 个用户组、16 组旧策略窗口、每个用户采样 **8 个排列**、KL=0.02、回放=0.1、温度 1、裁剪 0.2。候选数量始终为 **10**。1,258 组得到 315 次更新，保存 79/158/237/315；不同组数按相同比例生成检查点。SFT 初始化始终为冻结参考。

`make_run` 可加 `--historical-valid-scores`，在训练前检查迁移环境是否复现已有 SFT 验证分数；没有历史分数时，会重新构建本机参考，但不声称通过历史数值复现。原始代码哈希与输入哈希在训练前检查。`python -m ranking.train_grpo --run runs/cd_grpo --resume` 用于中断恢复。

## 5. 验证集选择与测试

为每个训练检查点运行评分，以下先展示 step79。其余检查点同样运行，改动 adapter 与 output 文件名中的步数。

```bash
python -m ranking.score --model data/models/Qwen3-VL-8B-Instruct --adapter runs/cd_grpo/model/checkpoint-79 --data artifacts/cd/valid/candidates.jsonl --output artifacts/cd/valid_step79.jsonl
python -m ranking.select --scores artifacts/cd/valid_step79.jsonl artifacts/cd/valid_step158.jsonl artifacts/cd/valid_step237.jsonl artifacts/cd/valid_step315.jsonl --steps 79 158 237 315 --alpha 1.5 --output runs/cd_grpo/selection.json
```

参考分支在验证集选中 step237。新运行须使用自己 `selection.json` 中锁定的步数；下面 237 是参考示例。

```bash
python -m ranking.score --model data/models/Qwen3-VL-8B-Instruct --adapter runs/cd_grpo/model/checkpoint-237 --data artifacts/cd/test/candidates.jsonl --output artifacts/cd/test_scores.jsonl
python -m ranking.rerank --scores artifacts/cd/test_scores.jsonl --alpha 1.5 --output artifacts/cd/test_rankings.jsonl
python -m ranking.evaluate --scores artifacts/cd/test_scores.jsonl --alpha 1.5 --all-users 1000 --output artifacts/cd/test_metrics.json
```

`--all-users 1000` 仅适用于原实验完整用户数，使用自己的数据时应填真实值。整体指标 = 自然召回率 × 命中用户条件指标。

## 历史参考

`reference_results.json` 来自原实验存档，未在本次代码整理中重训。101 名验证命中用户、108 名测试命中用户；测试全集 1,000 用户。GRPO step237、alpha=1.5 的条件 NDCG@10 为 0.573137954，整体 NDCG@10 为 0.061898899。属于单种子、已有测试暴露的实验记录。

策略数值测试：

```bash
python -m unittest ranking.test_policy -v
```
