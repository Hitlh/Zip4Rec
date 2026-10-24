# 模块三：推荐重排序

完整入口从模块二本次生成的图文数据开始，执行新建语言 LoRA 的 SFT、逐 epoch 验证和检查点选择，再从选中的 SFT 适配器执行 GRPO，最后完成验证选择、测试协议锁定与测试评估：

```bash
python -m ranking.pipeline --data artifacts/cd --datasets artifacts/cd_prepare/datasets --model data/models/Qwen3-VL-8B-Instruct --grpo-config examples/cd_lightgcn/grpo.json --output runs/cd_complete
```

SFT 会从预训练基座随机初始化 LoRA；GRPO 初始化与永久冻结参考均取本次验证选中的 SFT。基座 logits、SFT 参考 logits 和当前适配器 logits 均在相同图文输入上实际计算。SFT/GRPO 的待选检查点先在验证集评分，两个选择锁定到 `test_plan.json` 后才运行测试评分；输出逐用户排序、SFT/GRPO 语义分数、融合指标和 LightGCN 原序参考。模型选择不读取历史参考结果或测试评分。

SFT 保存步数由实际配对数量计算，GRPO 保存步数由实际用户组数计算，整体指标分母从模块一 manifest 读取。参考输入规模会得到 SFT 158/316/474 与 GRPO 79/158/237/315，但其他数据规模使用自己的步数。

运行命令见 [CD / LightGCN 示例](../examples/cd_lightgcn/README.md)。从仓库根目录以 `python -m ranking.<入口>` 运行。

| 文件 | 职责 |
| --- | --- |
| `pipeline.py` | 从新 SFT 到 GRPO、验证选择、锁定测试与结果汇总 |
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

数据源由模块一从原始数据完整生成，见 [数据准备](../semantic_context/README.md)。CD 的 validation/test 用户互斥，训练可复用相同用户更早的目标事件；运行冻结会检查请求 ID 不重叠及留出验证正例不进入该用户的训练历史或候选。
