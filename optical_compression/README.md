# 模块二：历史信息光学压缩

支持两种排版来源：原 CD 固定 renderer，以及针对新数据集用十条真实训练样本初始化的视觉 agent 策略。agent 始终保留上方用户、下方物品的单列分区和固定字体字号，通过修改 Python 排版程序优化换行、字段合并、宽度和间距。初始化、真实 token、转写检查、原实验排版测量与冻结命令见 [视觉排版优化指南](optimizer/README.md)。转写阈值默认 90%，可配置或设为 null 关闭。

完整批量入口会从模块一刚生成的语义训练、验证、测试集构建三个划分，渲染所有候选、组装图文输入，并调用实际模型 processor 检查每条输入的总 token、视觉 token 和长度上限。超过上限直接报错，不静默截断。

```bash
python -m optical_compression.pipeline --datasets artifacts/cd_prepare/datasets --model data/models/Qwen3-VL-8B-Instruct --output artifacts/cd
```

使用初始化后的策略时增加 `--layout artifacts/layout_cd_search/frozen` 并使用新的输出目录。此时批量入口还会逐条检查图片总输入 token 严格少于同内容纯文本；策略、字体和 processor 均记录指纹。省略 `--layout` 保持原 CD 行为。

`--model` 使用基座目录里的 processor；此阶段不加载完整模型权重。每个划分输出 `images/`、`candidates.jsonl`、`sft_pairs.jsonl`、`actual_tokens.json` 和渲染 manifest。该根目录可直接传给模块三。省略 `--model` 仅用于 CPU 渲染演示，模块三完整入口会要求先通过真实 processor 检查。

`renderer.py` 实现 CD 的 `compact` 页面：640px 宽、DejaVuSans 14px 正文、动态页面高度和 32px 高度对齐。字体随代码保存并附带许可。页面只含用户语义上下文与单个候选的语义，没有标签、目标标记或推荐分数。

`build.py` 可直接读取模块一导出的 `--samples datasets/*_candidates.jsonl`，也兼容原 `--groups` 入口。配对与权重实现统一在 `semantic_context/samples.py`，本模块负责渲染及图文消息组装。它为完整十候选组生成 `candidates.jsonl`，并生成原实验确定性抽取的正负配对 `sft_pairs.jsonl`。负例根据组 ID 哈希在按 LightGCN 分数排列的九个非目标中选取；按原排名使用 0.2/0.5/1.0 权重，并对被抽取负例的权重均值归一化。完整组内的 `_replay` 和 `_replay_weight` 对应这对 SFT 样本。只对训练输出使用样本权重；验证与测试使用完整候选。

```bash
python -m optical_compression.build --groups artifacts/cd_prepare/datasets/train.jsonl --item-summaries artifacts/cd_prepare/items/item_summaries.jsonl --output artifacts/cd/train
```

图像目录会写入记录中的绝对路径。移动资产后应重新构建输入并重新计算签名与参考缓存。渲染报告中的 `estimated_visual_tokens` 仅是几何估计；实际 token 以 `ranking.score` 调用模型 processor 的结果为准。

模块一完整产物的推荐入口：

```bash
python -m optical_compression.build --samples artifacts/cd_prepare/datasets/train_candidates.jsonl --output artifacts/cd/train
```
