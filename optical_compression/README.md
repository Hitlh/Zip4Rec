# 模块二：历史信息光学压缩

本模块分为一次性的**排版选择**和后续的**固定代码渲染**：

```text
模块一的真实训练语义样本
  → 视觉 agent 搜索排版代码、看图、测量 token、检查回读
  → 保存选中的 frozen/layout.py、冻结清单和字体
  → 后续只调用选中的代码，生成 train / valid / test 图片与图文输入
  → 模块三 SFT / GRPO
```

图片用于模型训练。agent 需要兼顾可识别性、字段结构与 token 成本，不能为少量 token 收益牺牲阅读清晰度。页面始终为上方用户、下方物品的单列布局；字体、字号、大标题、蓝绿分区栏和原文内容保留。行高、间距、换行、字段分组等规则可以搜索，不强制 16px。选中策略在所有后续划分中固定使用。

## 1. 搜索一次并保存选中代码

配置视觉服务的 `OPENAI_BASE_URL`、`OPENAI_API_KEY`，在 [`layout_optimizer.json`](../configs/cd_lightgcn/layout_optimizer.json) 中填写支持图片和工具调用的模型名称及预算：

```bash
python -m optical_compression.optimize \
  --config configs/cd_lightgcn/layout_optimizer.json \
  --samples artifacts/cd_prepare/datasets/train_candidates.jsonl \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/layout_cd_search
```

只使用十条去标签真实训练样本，不读取验证/测试结果。agent 依次提交完整 Python 排版程序、渲染并查看真实图片、测量目标 processor 的真实 token、检查本地 VLM 回读。默认每条样本两个区域的回读相似度都须达到 90%，且图片总输入 token 严格少于纯文本；合格并经 agent 看图的候选才会导出：

```text
artifacts/layout_cd_search/frozen/
├── layout.py        # 选中的通用排版代码
├── layout.json      # 合格条件、来源、评估与源码/引擎/字体指纹
└── fonts/           # 固定正文与粗体字体、许可
```

没有合格候选时明确失败，不生成选中代码。搜索需要视觉 API；默认回读需要本地模型权重和 GPU。工具接口、排版规则、转写门槛和预算见 [agent 说明](optimizer/README.md)。

## 2. 后续只调用选中代码

```bash
python -m optical_compression.pipeline \
  --datasets artifacts/cd_prepare/datasets \
  --layout artifacts/layout_cd_search/frozen \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/cd_densepage
```

这个入口不调用 agent，不需要 API 凭证。它读取选中代码并对三个数据划分批量渲染，生成图文记录，调用实际 processor 检查 token 与长度；冻结策略还会逐条检查严格少于纯文本。此时不加载 VLM 权重，也不重新执行回读。源代码、字体、引擎或 processor 不匹配时拒绝继续，避免搜索和批量运行混用不同策略。

每个划分输出 `images/`、`candidates.jsonl`、`sft_pairs.jsonl`、`actual_tokens.json` 和 manifest。完整根目录可直接交给模块三。省略 `--model` 仅用于 CPU 渲染检查，模块三完整训练入口会要求实际 processor 验证。

## 当前仓库已选中：compact

[`selected/compact.py`](selected/compact.py) 单独保存项目选定的 CD compact 代码，[`selected/selection.json`](selected/selection.json) 标记其为当前选中策略，后续直接调用。

使用这个已经选好的版本时，直接执行第 2 步，将 `--layout` 指向：

```bash
python -m optical_compression.pipeline \
  --datasets artifacts/cd_prepare/datasets \
  --layout optical_compression/selected \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/cd_densepage
```

省略 `--layout` 也会加载这个选定目录。原 compact 保留其 640px 宽度、14px 正文、16px 行高和 32px 高度对齐。批量运行会检查实际 processor token 和输入长度。详见 [选中代码说明](selected/README.md)。

## 从原始数据自动运行整个项目

```bash
# 语义数据准备完成后先调用 agent，再固定选中代码进行渲染和训练。
python run_pipeline.py --config configs/cd_lightgcn/pipeline_agent.json

# 复用仓库已选好的 compact，直接渲染并复现原 CD 训练流程。
python run_pipeline.py --config configs/cd_lightgcn/pipeline.json
```

`layout_optimizer_config` 指定一次性搜索配置；`layout_strategy` 指定已经选好的目录。两者互斥。完整入口在训练语义集产生后执行搜索，将结果保存在 `layout_search/frozen/`，后续只传递该目录。已完成且指纹未变的选择阶段可复用；批量渲染自身始终不触发搜索。

## 数据组织

`build.py` 读取模块一导出的 `*_candidates.jsonl`，调用选中代码，生成完整十候选组以及确定性的正负 SFT 配对；配对和权重逻辑统一在 `semantic_context/samples.py`。旧的 `--groups` 和 `--item-summaries` 输入也保留。

```bash
python -m optical_compression.build \
  --samples artifacts/cd_prepare/datasets/train_candidates.jsonl \
  --layout optical_compression/selected \
  --output artifacts/cd_densepage/train
```

记录包含图像绝对路径，移动资产后应重新构建模型输入与缓存。`estimated_visual_tokens` 只是原 compact 的几何估计，实际输入 token 以目标 processor 测量结果为准。原始数据、处理数据、搜索记录、生成图片和训练结果均留在本地，不随仓库发布。
