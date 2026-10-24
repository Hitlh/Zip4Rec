# CD / LightGCN 示例

示例对应 Amazon Reviews 2023 CDs & Vinyl、LightGCN 自然 Top-10 候选、Qwen3-VL-8B-Instruct、语言 LoRA、SFT 后 GRPO 的实验分支。

## 推荐：从原始数据完成整个项目

准备原始 Amazon Reviews 2023 CD reviews / metadata 文件，以及本地 Qwen3-VL-8B-Instruct。填写 [`pipeline.json`](pipeline.json) 中的路径和语义生成模型。安装 CUDA 匹配的 PyTorch 及 `requirements.txt`，在当前终端配置 `OPENAI_BASE_URL` 与 `OPENAI_API_KEY`，然后执行：

```bash
export CUDA_VISIBLE_DEVICES=0
python run_pipeline.py --config examples/cd_lightgcn/pipeline.json
```

该命令会实际运行数据筛选、五次上游 LightGCN 训练、物品描述与用户画像请求、返回校验、三类语义集导出、DensePage 渲染、全量 processor 检查、SFT 训练和验证、GRPO 参考构建与训练、验证集选择、最后的测试推理和整体指标计算。语义生成会使用配置的服务，模型训练使用选定 GPU；历史 `reference_results.json` 不参与训练或选择。

输出在 `artifacts/zip4rec_cd_full/`：

```text
semantic/        # 原始筛选、候选、生成请求/响应、语义资产、训练/验证/测试集
  datasets/      # 含原始语义组、SFT 正负配对、GRPO 完整候选、全量评估用户
densepage/      # 三类 PNG、图文输入与实际 token 检查
ranking/
   sft/          # 本次从基座新建的语言 LoRA 和各 epoch 检查点
   validation/   # SFT/GRPO 全部待选检查点的验证分数
   grpo/         # 固定 SFT 参考、GRPO 协议、参考缓存及新训练权重
   sft_selection.json, grpo_selection.json
   test_plan.json  # 两个验证选择完成后锁定，测试不参与选择
   test/         # SFT 与 GRPO 测试分数、逐用户排序、条件/整体指标
   results.json  # 本次 SFT、GRPO 与 LightGCN 参考排序的结果汇总
```

SFT 每个 epoch 保存一次，步数根据当前正负配对总数计算；GRPO 检查点按当前用户组数与四分位比例计算。测试分母读取本次完整评估用户数。`workflow_state.json` 记录每个阶段的命令、状态和产物哈希；相同配置重跑可跳过已完成且未变化的阶段。已失败阶段保留报告；SFT / GRPO 中断恢复使用各训练入口的 checkpoint 参数，API 返回校验失败则先处理失败记录，不自动更换模型或用测试结果重选。

以下给出模块级运行和小型接口演示，方便检查某个处理阶段。

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

从原始 reviews / metadata 开始，运行 [模块一完整数据准备流程](../../semantic_context/README.md)。该流程包含筛选、去重、user-10/item-15 k-core、固定验证/测试用户组、严格未来屏蔽的自然候选、物品描述与画像生成、结果校验，以及最终的 train/valid/test 语义样本。

最终输入已完成 SFT 正负抽取和回放标记，直接渲染为三个划分：

```bash
python -m optical_compression.build --samples artifacts/cd_prepare/datasets/train_candidates.jsonl --output artifacts/cd/train
python -m optical_compression.build --samples artifacts/cd_prepare/datasets/valid_candidates.jsonl --output artifacts/cd/valid
python -m optical_compression.build --samples artifacts/cd_prepare/datasets/test_candidates.jsonl --output artifacts/cd/test
```

原实验 validation/test 用户组互斥；训练可以使用相同用户更早的时间截点，必须屏蔽截点之后的全部正反馈及最终外部目标。并非训练/验证/测试用户三者完全互斥。进入页面渲染的是自然命中用户，完整用户数由模块一 manifest 保留。
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
