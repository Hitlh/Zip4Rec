# 一次性视觉排版优化

这是模块二的第一步：为新数据集搜索一次可复用的 DensePage 排版程序。输入为十条真实训练样本、简单初始布局、固定字体和字号、目标 Qwen3-VL 的本地目录。输出为选中的 `frozen/layout.py`、字体和冻结清单，后续批量渲染只调用这个固定代码，无需再次调用 agent。项目当前已选中的原 CD compact 单独保存在 [`selected/compact.py`](../selected/compact.py)，选择状态见 [`selected/selection.json`](../selected/selection.json)。后续可直接加载这个选中目录进行渲染。

agent 的系统提示明确：页面将用于后续推荐相关性与重排序模型的 SFT/GRPO 训练，搜索须兼顾 token 成本、文字可识别性和语义结构清晰度。先形成可靠可读的布局，再压缩；不得为少量 token 收益牺牲字段边界、段落结构或回读稳定性。90% 回读门槛是最低合格要求，不是优化目标。须同时查看两个区域和最差样本，避免过紧行距、过多断词、无区分地合并字段；行高可选固定或自适应，不强制 16px。每次修改理由须说明对识别与结构的影响。自动选优仍按下文的合格检查与 token 排序执行；这些搜索指导和回读分数不代表已经验证了下游训练效果。

页面固定为上方 USER PROFILE、下方 CANDIDATE ITEM，单列顺序阅读。agent 可修改宽度、四边留白、固定行高与文字基线、行距、区间距、页面高度对齐、区域内字段顺序、换行、分组和分隔符。用户原文与物品字段值均须保留；物品字段标签属于展示元数据，可改名或合并为组合标签。原 compact 的大标题 `PROFILE-CONDITIONED RELEVANCE`、粗体分区标题、蓝色 USER PROFILE 栏、绿色 CANDIDATE ITEM 栏、配色和外框由公共引擎强制绘制，不可删除或修改，且计入每次真实 token 测量。字体、字号及评分规则同样固定。

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

多于十条时按规范化文本总长度分层选择十条，包含最短和最长样本。同文字的重复输入会去重。提供文件中出现验证或测试记录会被拒绝。标签、推荐分数、源样本编号、目标 ID 等字段不提供给 agent，样本重新编号为 `example_01` 至 `example_10`。用户和物品字段中的空白先规范化；完整内容与纯文本基线共用同一规范化结果。物品字段额外保留 `key`、`label`、`values` 元数据，其中 `values` 是按原顺序保存的值列表。默认 `text` 和纯文本基线保持原有格式；展示规则不会修改这份基线。

配置示例：[`layout_optimizer.json`](../../configs/cd_lightgcn/layout_optimizer.json)。`initial_layout` 指定你提供的简单初始排版源码，须遵守下文接口；省略时使用仓库内初始布局。正文默认使用仓库内 DejaVuSans，标题使用 DejaVuSans-Bold；可在 `constraints` 中用 `font_path` 和 `bold_font_path` 指定对应本地字体，路径相对运行时工作目录。`font_size`、`heading_size` 和 `title_size` 在整个搜索中固定，默认分别为 14、15、18px。初始化字体缺少数据字符时直接拒绝，应提供覆盖数据语言的字体。

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

1. 上下分区，固定字体和字号，完整保留原版大标题、粗体分区标题、蓝绿彩色栏和外框。页面须容纳完整单行标题，左右边距至少 8px、上下留白至少 4px。用户原文及每个物品字段值不丢失、不重复、字段内部不乱序；固定行高不得造成遮挡，最终向上对齐后的画布也须满足尺寸限制。
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

每行由 `[field_id, start, end]` 原文片段组成。agent 可以编写基于长度、宽度的循环、条件分支、字段重排和合并算法；不是只提交一份参数 JSON。支持的 Python 子集、内置函数与接口约束在 agent 系统提示中完整提供。公共引擎验证片段覆盖、绘制固定大标题和彩色分区栏、上下两区并计算页面高度，排版程序无法修改这些装饰、字体或绘图评分代码。

### 通用排版规则（API v2）

原有六个必填字段继续可用。以下字段均为可选，省略时使用原引擎默认值：

| 字段 | 含义 |
| --- | --- |
| `margins` | `{left, right, top, bottom}`，四项齐全时替代 `padding` 的默认留白；左右至少 8px，上下为 4–512px |
| `line_height` | 固定每行前进像素数（1–256），再叠加 `line_gap`；null 时按该行字形高度计算 |
| `text_anchor` | `lt` 按字形顶部排；`la` 按共同字体基线排 |
| `text_origin` | `ink` 补偿负的左侧字形偏移；`font` 使用字体原点 |
| `height_multiple` | 将最终高度向上对齐到指定倍数（1–256）；32 可表达原 compact 的高度对齐 |
| `min_height` | 页面最小高度，默认 32px |
| `trim_rows` | 仅去掉每行首尾空白后绘制；原文片段覆盖检查仍完整进行 |
| `presentation` | 由原字段构造展示字段的规则，公共引擎从原始输入独立重建并验证 |

`metrics.flow(fields, width, pack=False, wrap='split')` 保留拆分长词的原有行为。`wrap='words'` 在单词边界换行，并须搭配 `trim_rows=True`；完整单词超过可用宽度时明确失败，不允许溢出。也可自行实现换行算法，再提交合法的原文片段。

`presentation` 按 `user` / `item` 指定分组列表。叶子组用 `ids` 引用原字段，`separator` 决定字段间分隔符，`joins={后续字段ID: 分隔符}` 可逐字段覆盖。用户文字完整保留。物品组还支持：

- `label`：展示标签；单字段默认原标签，合并字段时必须显式提供非空组合标签，标签含义须准确对应原字段。
- `list_separator`：列表值的分隔符，默认 `; `，可用 `, ` 等。
- `empty_value`：空列表显示 `Unknown` 或空串。
- `skip_empty`：仅跳过空值；任何非空值均不能省略。

容器组使用 `groups=[子组, ...]` 和 `separator` 连接子组，最多嵌套四层，可表达一个段落里多个带标签字段用 ` • ` 分隔。分隔符限 1–16 个空格或标点（逗号、分号、冒号、斜线、竖线、圆点、括号、破折号），不接受隐藏控制字符或替代原值的文本模板。所有源字段在所属区域须恰好出现一次。

```python
presentation = {
    'item': [
        {'ids': ['i0'], 'label': 'Title'},
        {'groups': [
            {'ids': ['i1'], 'label': 'Artists', 'list_separator': ', '},
            {'ids': ['i2', 'i3'], 'label': 'Era / styles', 'separator': '; ',
             'list_separator': ', '},
        ], 'separator': ' • '},
    ],
}
fields = metrics.prepare(document, presentation)
# 用 fields['user'] / fields['item'] 生成 rows，并在最终 plan 中带上同一个 presentation。
```

上例假定恰有这四个物品字段。实际布局应按 `field['key']` 构造分组，并覆盖新数据集的所有字段。准备后的分组字段 ID 是 `user_g0`、`item_g0` 等；未配置的区域仍使用原字段。引擎先验证所有字段值均保留，再验证准备后字符片段的完整覆盖；回读对照为实际显示的文字。

原 compact 的通用 API 等价程序仅作为测试夹具保留在 `tests/fixtures/compact_equivalent.py`，用于验证搜索空间包含原策略，不会提供给 agent 或自动加入候选池。默认初始源码仍是 `layouts/initial.py`。实际保留的选中代码统一放在 [`selected/`](../selected/README.md)。
因此，agent 从简单布局出发可以生成与 compact 等价的合法策略；有限轮次不保证一定发现该策略。API v2 更改了引擎与输入元数据指纹，旧冻结包需在新引擎下重新评估和冻结，不能直接沿用历史 token/回读报告。

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

冻结包可移动到其他目录使用，加载时核对源代码、正文/粗体字体、固定装饰和引擎哈希。旧版缺少大标题和彩色栏的搜索结果不满足此约束，不能继续作为最终策略；需在新引擎下重新渲染、测量、回读并冻结，旧 token 降幅不可沿用。公开最终策略时连同冻结清单、字体及适用许可一并提供；真实数据和生成的运行产物留在本地，不提交仓库。

## 在第二阶段使用

```bash
python -m optical_compression.pipeline \
  --datasets artifacts/cd_prepare/datasets \
  --model data/models/Qwen3-VL-8B-Instruct \
  --layout artifacts/layout_cd_search/frozen \
  --output artifacts/cd_agent_layout
```

完整的先搜索后渲染入口使用 [`pipeline_agent.json`](../../configs/cd_lightgcn/pipeline_agent.json) 的 `layout_optimizer_config`：训练语义样本生成后自动运行本入口，随后固定调用生成的 `layout_search/frozen/`。已经选好代码时，使用 `layout_strategy` 指定该冻结目录，或指定 `optical_compression/selected` 复用项目选定的 compact；这两种模式都不重新调用 agent。两项配置互斥，省略时默认复用项目选定的 compact。

批量入口核对 processor 与优化时一致，并对 train/valid/test **每条输入**检查长度和严格小于纯文本的要求；短样本等若不满足会明确失败，不按验证或测试表现重新搜索。可选转写只报告实际测过的初始化样本分数。`--layout` 不触发 agent 请求。更换策略需使用新的输出目录，并重新生成下游输入、参考缓存和训练结果。

## 验证范围

离线测试使用临时合成输入、模拟 API 和模拟测量/转写后端，检查真实 Pillow 渲染、工具协议、图片反馈、代码执行、内容边界、失败拒绝、策略导出和三个划分的接入。测试还验证通过通用 API 表达的 compact 与原 renderer 的 PNG 字节一致，以及格式化策略的回读对照、冻结和重放。它们不替代真实服务与本地 Qwen 模型的端到端运行。
