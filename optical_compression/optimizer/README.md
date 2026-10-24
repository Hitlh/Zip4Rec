# 一次性视觉排版优化

此入口为新数据集搜索可复用的 DensePage 排版程序。输入为十条真实训练样本、简单初始布局、固定字体和字号、目标 Qwen3-VL 的本地目录。输出为独立的 `layout.py` 和冻结清单，后续批量渲染无需调用 agent。

页面固定为上方 USER PROFILE、下方 CANDIDATE ITEM，单列顺序阅读。agent 可修改宽度、边距、行距、区间距、区域内字段顺序、换行和短字段合并算法。所有原文字符均须保留；字体、字号、颜色及评分规则由公共代码固定。

## 实现

`agent.py` 使用 OpenAI-compatible Chat Completions 的视觉输入与 function calling，通过已有的 `httpx` 发送请求，不依赖额外 agent 框架。参考接口见 [Function calling](https://developers.openai.com/api/docs/guides/function-calling) 和 [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)。服务和模型必须同时支持图片输入及工具调用，模型名称在配置中指定；默认示例为 `gpt-4o`。

四个工具：

| 工具 | 功能 |
| --- | --- |
| `inspect_inputs` | 查看十条去标签训练样本、字体限制、纯文本基线、初始代码和当前最佳版本 |
| `revise_layout` | 提交完整的候选 `layout()` 源码及修改理由 |
| `render_and_measure` | 渲染、检查内容和几何边界、统计实际 token，并将真实 PNG 返回给视觉 agent |
| `check_readability` | 用本地 Qwen3-VL 转写图片，按配置的阈值检查两个区域；阈值为 null 时仅报告分数 |

Chat Completions 的工具返回值承载 JSON 测量结果；图片紧接着作为 user 多模态消息发送。agent 直接看图，无单独看图工具或候选比较工具。较早的图片会从后续请求中移除，数值反馈和本地图片文件保留。程序自动维护最佳策略，不依赖 agent 自报分数。未实际发送图片给 agent 的候选不能导出。

## 数据和配置

使用模块一的 `train_candidates.jsonl`，或提供十条同结构的真实样本，每条至少包含：

- `split`: 必须为 `train`。
- `user_context`: 用户上下文文字。
- `candidate_payload`: 物品字段对象，可使用新数据集自己的字段。

多于十条时按规范化文本总长度分层选择十条，包含最短和最长样本。同文字的重复输入会去重。提供文件中出现验证或测试记录会被拒绝。标签、推荐分数、源样本编号、目标 ID 等字段不提供给 agent，样本重新编号为 `example_01` 至 `example_10`。用户和物品字段中的空白先规范化；完整内容与纯文本基线共用同一规范化结果。

配置示例：[`layout_optimizer.json`](../../configs/cd_lightgcn/layout_optimizer.json)。`initial_layout` 指定你提供的简单初始排版源码，须遵守下文接口；省略时使用仓库内初始布局。字体默认使用仓库内 DejaVuSans；可在 `constraints` 中添加 `font_path` 指定本地字体，路径相对运行时工作目录。`font_size` 和 `heading_size` 在整个搜索中固定。初始化字体缺少数据字符时直接拒绝，应提供覆盖数据语言的字体。

先检查实际基线，不调用 agent API、不加载 VLM 权重：

```bash
python -m optical_compression.optimize \
  --config configs/cd_lightgcn/layout_optimizer.json \
  --samples artifacts/cd_prepare/datasets/train_candidates.jsonl \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/layout_cd_preview \
  --prepare-only
```

`initialization.json` 保存每条样本实际的纯文本 token 数。也可将对应字典写入配置的 `text_token_limits`；程序会核对这些数与本次 processor 的实测数完全一致，防止不同提示词、样本或模型的基线混用。无需手动提供时由程序自动测量。

## 执行搜索

按现有项目方式设置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。agent 请求会发送选中样本的文字和图片到配置的服务。默认 `reader.enabled=true`、`readability_threshold=0.90`，需本地 Qwen3-VL 权重进行转写。将阈值设为 `null` 可关闭转写门槛，此时可继续保留转写诊断，或再将 `reader.enabled=false` 设为仅用 processor/tokenizer 和 agent 看图，无需本地模型权重或 GPU。

```bash
export CUDA_VISIBLE_DEVICES=0
python -m optical_compression.optimize \
  --config configs/cd_lightgcn/layout_optimizer.json \
  --samples artifacts/cd_prepare/datasets/train_candidates.jsonl \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/layout_cd_search
```

每次使用新的输出目录；当前入口保存搜索记录，但不支持中断后的 agent 会话恢复。默认最多 20 个候选（含初始布局）、40 次模型交互；agent 也可提前结束。API 使用量写入 trace。执行候选 Python 的工作进程限制 CPU、内存、输出和墙钟时间，且不继承 API 凭证；这用于约束生成代码故障，不是面向恶意 Python 的操作系统安全沙箱。该执行方式面向 Linux。

## 硬约束与选择规则

每条初始化样本均须满足：

1. 上下分区，固定字体和字号，原文字段字符不丢失、不重复、字段内部不乱序，无裁切或遮挡。
2. 使用正式评分相同的 system/user 提示及 generation prompt，图片方案的**总输入 token 严格少于同内容纯文本方案**；相等或更多不合格。
3. 总输入长度不超过配置的 `max_length`，默认 2048。
4. `readability_threshold` 非 null 时，所有初始化样本的用户区和物品区分别达到指定相似度，默认 0.90。启用此门槛时必须同时启用 reader。

合格候选按平均真实视觉 token、最大视觉 token、候选 ID 依次升序选择。视觉 token 来自实际 processor 的图像占位 token 数，不使用页面面积估算，也不是图片文件大小。

视觉 agent 同时查看实际图片。本地转写工具只接收图片和固定指令，不接收原文答案或相关性标签，返回用户区、物品区的字符序列相似度。**默认门槛为 90%，可配置为其他数值，或用 null 关闭。** 关闭后分数不影响候选合格判定、排序或导出。这些分数是可读性代理指标，与推荐质量和标准 OCR 准确率分开解释。

没有合格候选时保留搜索记录、以失败状态结束，不导出 `frozen/`。十条样本上的最佳策略不代表全局最优；应在独立训练样本上检查泛化，再固定用于验证和测试。布局搜索不读取验证/测试结果，也不修改 SFT/GRPO 的训练算法。

## 最终优化的代码

`layouts/initial.py` 是简单初始排版。agent 生成的程序保存在每个候选的 `layout.py`，其接口为：

```python
def layout(document, metrics, constraints):
    # document['user'] / document['item']: [{'id': 'u0', 'text': '...'}, ...]
    # metrics.width(text): 固定字体下的实际文字宽度
    # metrics.flow(fields, width, pack=False): 自动换行，长词也可拆分
    return {
        'width': 640,
        'padding': 12,
        'line_gap': 4,
        'section_gap': 16,
        'user_rows': metrics.flow(document['user'], 616, pack=False),
        'item_rows': metrics.flow(document['item'], 616, pack=False),
    }
```

每行由 `[field_id, start, end]` 原文片段组成。agent 可以编写基于长度、宽度的循环、条件分支、字段重排和合并算法；不是只提交一份参数 JSON。支持的 Python 子集、内置函数与接口约束在 agent 系统提示中完整提供。公共引擎验证片段覆盖、绘制上下两区并计算页面高度，排版程序无法修改字体或绘图评分代码。

最终目录：

```text
artifacts/layout_cd_search/
├── initialization.json         # 十条输入、纯文本基线、字体和processor记录
├── candidates/<id>/            # 候选代码、PNG与评估/转写结果
├── agent_trace.json            # 工具调用和API usage；图片以哈希记录
├── search_summary.json
└── frozen/                     # 仅合格搜索产生
    ├── layout.py               # 最终固定的通用排版程序
    ├── layout.json             # 字体/源码/引擎/processor指纹和评估条件
    └── fonts/                  # 固定字体；源字体目录有许可时一并复制
```

冻结包可移动到其他目录使用，加载时核对源代码、字体和引擎哈希。公开最终策略时连同冻结清单、字体及适用许可一并提供；真实数据和生成的运行产物留在本地，不提交仓库。

## 在第二阶段使用

```bash
python -m optical_compression.pipeline \
  --datasets artifacts/cd_prepare/datasets \
  --model data/models/Qwen3-VL-8B-Instruct \
  --layout artifacts/layout_cd_search/frozen \
  --output artifacts/cd_agent_layout
```

也可对全流程配置添加 `"layout_strategy": "artifacts/layout_cd_search/frozen"`，再运行 `run_pipeline.py`。省略该字段或 `--layout` 时保留原 CD renderer，供原实验复现与基线比较。

批量入口核对 processor 与优化时一致，并对 train/valid/test **每条输入**检查长度和严格小于纯文本的要求；短样本等若不满足会明确失败，不按验证或测试表现重新搜索。可选转写只报告实际测过的初始化样本分数。`--layout` 不触发 agent 请求。更换策略需使用新的输出目录，并重新生成下游输入、参考缓存和训练结果。

## 测量当前实验的固定排版

简易初始布局和原 CD 实验排版是两种不同程序。以下入口调用原实验 `render_compact()`，以实际绘制的字段格式作为转写对照；标题、配色、字体、640px 宽度及动态高度均保持原样。

```bash
python -m optical_compression.audit \
  --samples artifacts/cd_prepare/datasets/train_candidates.jsonl \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/current_compact_audit \
  --readback
```

`--readback` 加载本地模型权重进行诊断；省略它只渲染和统计真实 token。输出 `report.json`、PNG、逐样本用户区/物品区相似度及均值/最小值/最大值，不判定转写通过率。使用与优化相同的十条输入，可比较不同排版的实测分数。

## 验证范围

离线测试使用临时合成输入、模拟 API 和模拟测量/转写后端，检查真实 Pillow 渲染、工具协议、图片反馈、代码执行、内容边界、失败拒绝、策略导出和三个划分的接入。它们不替代真实服务与本地 Qwen 模型的端到端运行；仓库不预置或宣称已经搜索得到的最优 CD 策略。
