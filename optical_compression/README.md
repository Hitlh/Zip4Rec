# 模块二：光学压缩

先用视觉 agent 为数据集选出排版代码，再固定该代码批量生成 DensePage，作为模块三的训练与评分输入。

## 工作流程与代码

```text
模块一 train_candidates.jsonl
  → 抽取十条真实训练样本 → agent 修改排版、看图、测量与回读
  → frozen/layout.py + 字体 + 配置
  → 固定调用选中代码 → train / valid / test 图片与图文数据
```

页面固定为上方用户、下方物品的单列布局。字体、字号、大标题和蓝绿分区栏保持不变，允许搜索宽度、行高、留白、换行和字段分组。图片用于 SFT/GRPO 训练，搜索同时考虑文字可识别性、字段结构和 token 成本。

| 文件或目录 | 作用 |
| --- | --- |
| `optimize.py` | 一次性排版搜索入口 |
| `optimizer/agent.py` | 视觉模型 HTTP 客户端、提示词与工具调用循环 |
| `optimizer/tools.py` | 四个工具、候选评估、选优和固定代码导出 |
| `layouts/initial.py` | 简单上下分区的初始排版 |
| `document.py` | 整理无标签页面文字、纯文本基线与十条训练样本 |
| `layout_engine.py`、`layout_worker.py` | 执行排版代码、检查内容/几何、绘制固定装饰与页面 |
| `presentation.py` | 字段分组、标签和分隔符处理 |
| `measurement.py` | 实际 processor token 测量与本地 VLM 回读 |
| `selected/compact.py`、`selected/selection.json` | 已选中的 CD compact 排版代码及选择标记 |
| `strategy.py` | 加载选中代码，供批量入口调用 |
| `build.py` | 为一个划分生成图片、候选输入和 SFT 配对 |
| `measure_tokens.py` | 检查实际图文输入 token 和长度上限 |
| `pipeline.py` | 对三个划分执行固定代码渲染与 token 检查 |
| `fonts/` | 固定字体及许可 |

## 1. 搜索一次排版

按[根 README](../README.md) 设置 API，修改 [layout_optimizer.json](../configs/cd_lightgcn/layout_optimizer.json) 中的视觉模型与预算，从仓库根目录运行：

```bash
python -m optical_compression.optimize \
  --config configs/cd_lightgcn/layout_optimizer.json \
  --samples artifacts/cd_prepare/datasets/train_candidates.jsonl \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/layout_cd_search
```

输入至少需要十条不同的真实训练页面。默认要求十条样本的用户/物品区域回读相似度均 ≥90%，图像总输入 token 逐条严格少于对应纯文本，并满足 2048 长度上限。搜索需要视觉 API，本地回读需要 Qwen3-VL 权重和 GPU。

成功后导出 `artifacts/layout_cd_search/frozen/`，含 `layout.py`、`layout.json` 和 `fonts/`；无合格候选则停止。四个工具的实现见 [optimizer/tools.py](optimizer/tools.py)，搜索参数见 [layout_optimizer.json](../configs/cd_lightgcn/layout_optimizer.json)。

## 2. 固定代码批量生成

```bash
python -m optical_compression.pipeline \
  --datasets artifacts/cd_prepare/datasets \
  --layout artifacts/layout_cd_search/frozen \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/cd_densepage
```

本步骤只调用选中代码，使用 processor 检查实际 token；不再搜索或回读。选中代码与字体固定用于训练、验证和测试三个划分，内容不得截断。

仓库已选中的 CD 代码为 [selected/compact.py](selected/compact.py)，直接复用时将上面命令的 `--layout` 改为 `optical_compression/selected`，见[调用说明](selected/README.md)。

## 输出与下一步

每个划分目录生成 `images/`、`candidates.jsonl`、`sft_pairs.jsonl`、`actual_tokens.json` 和 `manifest.json`，根目录生成 `complete.json`。将整个 `artifacts/cd_densepage/` 和模块一的 `datasets/` 交给[模块三](../ranking/README.md)。

图文记录使用图片绝对路径，移动图片后需重新构建输入。后续训练要求本步骤提供 `--model` 并通过实际 processor 检查。
