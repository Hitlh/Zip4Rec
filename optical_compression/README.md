# 模块二：历史信息光学压缩

`renderer.py` 提取原 CD `compact` 页面：640px 宽、DejaVuSans 14px 正文、动态页面高度和 32px 高度对齐。字体随代码保存并附带许可。页面只含用户语义上下文与单个候选的语义，没有标签、目标标记或推荐分数。

`build.py` 为完整十候选组生成 `candidates.jsonl`，并生成原实验确定性抽取的正负配对 `sft_pairs.jsonl`。负例根据组 ID 哈希在按 LightGCN 分数排列的九个非目标中选取；按原排名使用 0.2/0.5/1.0 权重，并对被抽取负例的权重均值归一化。完整组内的 `_replay` 和 `_replay_weight` 对应这对 SFT 样本。只对训练输出使用样本权重；验证与测试使用完整候选。

```bash
python -m optical_compression.build --groups artifacts/demo/train_context.jsonl --item-summaries examples/cd_lightgcn/item_summaries.jsonl --output artifacts/demo/train
```

图像目录会写入记录中的绝对路径。移动资产后应重新构建输入并重新计算签名与参考缓存。渲染报告中的 `estimated_visual_tokens` 仅是几何估计；实际 token 以 `ranking.score` 调用模型 processor 的结果为准。
