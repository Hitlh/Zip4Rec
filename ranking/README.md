# 模块三：推荐重排序

从模块二生成的图文输入开始，训练 SFT 与 GRPO 适配器，通过验证集选择检查点，再对测试集评分、融合排序并计算指标。

## 工作流程

1. 从 Qwen3-VL-8B-Instruct 初始化语言 LoRA，以 A/B 正负配对进行 SFT。
2. 对 SFT 检查点在验证集评分，按 NDCG@10、MRR@10、H@1 及更早步数选择。
3. 从选中的 SFT 初始化 GRPO，准备基座和冻结 SFT 的参考分数，再训练排列策略。
4. 在验证集选择 GRPO 检查点，写入固定测试方案。
5. 对选中的 SFT/GRPO 进行测试，输出语义分数、融合排序、LightGCN 原序参考及指标。

语义分数取适配器相对基座的 A/B logits 增量。融合为同用户十候选内的 `z(LightGCN) + 1.5 × z(semantic)`；只重排自然召回候选，不注入目标。指标分别报告召回命中用户的条件结果，以及以完整用户数为分母的整体结果。

## 代码作用

| 文件 | 作用 |
| --- | --- |
| `pipeline.py` | 串联 SFT、GRPO、验证选择与固定模型测试 |
| `train_sft.py`、`sft_support.py` | 加权 A/B 增量交叉熵、语言 LoRA 和 SFT 数据处理 |
| `make_run.py` | 固定 GRPO 输入、初始适配器与训练配置 |
| `prepare_reference.py` | 计算基座与冻结 SFT 的参考 logits |
| `train_grpo.py` | 旧策略采样窗口、训练更新、检查点与恢复 |
| `policy.py` | Plackett–Luce 排列采样、相对优势、裁剪、KL 与 SFT 回放 |
| `exact_policy.py` | 短候选列表的精确期望，供策略诊断和梯度检查 |
| `model_io.py` | 模型与 LoRA 加载、精度及前向计算 |
| `conversation.py` | 图文对话与 processor 输入 |
| `score.py` | 用固定适配器计算候选语义分数 |
| `select.py` | 根据验证集分数选择检查点 |
| `rerank.py`、`evaluate.py` | 分数融合、确定性排序、条件与整体指标 |
| `common.py`、`runtime_fingerprint.py` | 共享数据校验、读写及运行环境记录 |

## 输入与运行

需要支持 BF16 的单张 CUDA GPU，以及模块一和模块二本次运行的完整输出：

- `--data`：模块二根目录，包含 train/valid/test 的图文候选、训练配对和 `complete.json`；必须已通过实际 processor 检查。
- `--datasets`：模块一的 `datasets/`，提供划分标识和完整评估用户数。
- `--model`：本地 Qwen3-VL-8B-Instruct 权重、tokenizer 和 processor。
- `--grpo-config`：GRPO 参数，默认示例见 [grpo.json](../configs/cd_lightgcn/grpo.json)。

从仓库根目录运行：

```bash
export CUDA_VISIBLE_DEVICES=0
python -m ranking.pipeline \
  --data artifacts/cd_densepage \
  --datasets artifacts/cd_prepare/datasets \
  --model data/models/Qwen3-VL-8B-Instruct \
  --grpo-config configs/cd_lightgcn/grpo.json \
  --output artifacts/cd_ranking
```

可用 `--sft-learning-rate` 修改 SFT 学习率，默认 `1e-5`。SFT 训练三轮、梯度累积 16；保存步数由实际配对数计算。GRPO 默认每用户八个排列、每次更新四个用户组；其余参数见配置文件。每组固定十个候选且仅一个正例，需保留模块一/二产生的回放标记和权重。

## 输出

| 路径 | 内容 |
| --- | --- |
| `sft/` | SFT 权重和检查点 |
| `grpo/` | 固定训练协议、参考缓存、训练状态和检查点 |
| `validation/` | 检查点验证分数 |
| `sft_selection.json`、`grpo_selection.json` | 两阶段选择结果 |
| `test_plan.json` | 测试前固定的模型与融合设置 |
| `test/` | 测试分数、逐用户排序及指标 |
| `results.json` | SFT、GRPO 与 LightGCN 原序的汇总结果 |

相同配置重跑可复用完整阶段。中途训练恢复可分别使用 `ranking.train_sft --resume-from-checkpoint` 和 `ranking.train_grpo --resume`，完整参数见各入口的 `--help`。从原始数据自动运行三个模块的命令见[根 README](../README.md)。
