# 模块三：推荐重排序

运行命令见 [CD / LightGCN 示例](../examples/cd_lightgcn/README.md)。从仓库根目录以 `python -m ranking.<入口>` 运行。

| 文件 | 职责 |
| --- | --- |
| `train_sft.py`, `sft_support.py` | 原始加权 A/B 增量交叉熵、语言 LoRA、顺序采样 |
| `make_run.py` | 将输入、SFT 权重和配置冻结为本地 `protocol.json` |
| `prepare_reference.py` | 当前运行的基座 logits 与永久冻结 SFT 参考分数 |
| `train_grpo.py` | 16 用户旧策略窗口、4 用户一次更新、检查点和恢复 |
| `policy.py` | PL/Gumbel 排列采样、组相对优势、逐步裁剪、KL 与回放 |
| `score.py` | 固定适配器评分、输入签名和实际 token 计数 |
| `select.py` | 仅使用相同验证集合选择训练检查点 |
| `rerank.py`, `evaluate.py`, `common.py` | 分数融合、确定性排序、命中子集和整体指标 |

基座使用 BF16，受限输出头与 LoRA 使用 FP32，排序策略使用 FP64。语义分数为 `float32(delta_A - delta_B)`；其中 delta 是适配器与未适配基座在同一输入上的 logits 差。推理不需要标签，评估时才读取目标。推理融合使用每用户候选集合内总体标准差：`z(base) + 1.5 * z(semantic)`，相同融合分数按物品 ID 升序。

`policy.py` 和 `exact_policy.py` 保留原实验共享实现，包括用于原方法对照和精确梯度核验的辅助函数；公开训练入口限定 `objective_kind=grpo`。GRPO 不训练 critic，首选分类 KL 的参考始终为 SFT 初始化。有效选择步骤求和，再对采样排列与用户求均值，不能改为每步平均。此 CD 分支未使用后续其他数据集的 FP32 bridge 或正例终止变体。

每个训练组须有十个候选且仅一个 A 标签。原始 SFT 配对必须保留 `_replay` 与负例 `_replay_weight`。完整训练组、参考缓存和实际输入签名必须一致。恢复用 `--resume`，会保留优化器、随机数状态及旧策略窗口；不要编辑已经冻结的代码或运行配置。

运行 GPU 阶段前设置 `CUDA_VISIBLE_DEVICES` 为单张卡。目录和模型权重均由命令参数指定。精确历史结果还依赖原始数据、processor、初始权重、GPU 与运行库；当前仓库迁移后的入口尚未执行完整 GPU 训练。
