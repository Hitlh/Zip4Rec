# 模块二：历史信息光学压缩

完整批量入口会从模块一刚生成的语义训练、验证、测试集构建三个划分，渲染所有候选、组装图文输入，并调用实际模型 processor 检查每条输入的总 token、视觉 token 和长度上限。超过上限直接报错，不静默截断。

```bash
python -m optical_compression.pipeline --datasets artifacts/cd_prepare/datasets --model data/models/Qwen3-VL-8B-Instruct --output artifacts/cd
```

`--model` 使用基座目录里的 processor；此阶段不加载完整模型权重。每个划分输出 `images/`、`candidates.jsonl`、`sft_pairs.jsonl`、`actual_tokens.json` 和渲染 manifest。该根目录可直接传给模块三。省略 `--model` 仅用于 CPU 渲染演示，模块三完整入口会要求先通过真实 processor 检查。

`renderer.py` 提取原 CD `compact` 页面：640px 宽、DejaVuSans 14px 正文、动态页面高度和 32px 高度对齐。字体随代码保存并附带许可。页面只含用户语义上下文与单个候选的语义，没有标签、目标标记或推荐分数。

`build.py` 可直接读取模块一导出的 `--samples datasets/*_candidates.jsonl`，也兼容原 `--groups` 入口。配对与权重实现统一在 `semantic_context/samples.py`，本模块负责渲染及图文消息组装。它为完整十候选组生成 `candidates.jsonl`，并生成原实验确定性抽取的正负配对 `sft_pairs.jsonl`。负例根据组 ID 哈希在按 LightGCN 分数排列的九个非目标中选取；按原排名使用 0.2/0.5/1.0 权重，并对被抽取负例的权重均值归一化。完整组内的 `_replay` 和 `_replay_weight` 对应这对 SFT 样本。只对训练输出使用样本权重；验证与测试使用完整候选。

```bash
python -m optical_compression.build --groups artifacts/demo/train_context.jsonl --item-summaries examples/cd_lightgcn/item_summaries.jsonl --output artifacts/demo/train
```

图像目录会写入记录中的绝对路径。移动资产后应重新构建输入并重新计算签名与参考缓存。渲染报告中的 `estimated_visual_tokens` 仅是几何估计；实际 token 以 `ranking.score` 调用模型 processor 的结果为准。

模块一完整产物的推荐入口：

```bash
python -m optical_compression.build --samples artifacts/cd_prepare/datasets/train_candidates.jsonl --output artifacts/cd/train
```
