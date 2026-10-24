# 模块一：从原始数据到训练、验证与测试集

本模块完整负责数据准备：原始 Amazon Reviews 2023 CDs & Vinyl reviews / metadata → 数据筛选与去重 → 正反馈 k-core → 冻结用户组和时间目标 → LightGCN 自然候选 → 物品语义提取 → 历史用户画像 → 可用于后续渲染的语义样本与正负训练配对。

## 处理步骤与代码

| 步骤 | 实现 | 产物 |
| --- | --- | --- |
| 原始数据筛选 | `raw_filters.py`, `filter_data.py` | 去重后的正反馈交互 TSV、各级筛选统计 |
| 物品描述清洗 | `extract_metadata.py` | 标题、类别、品牌、去 HTML 描述与原始 ID 映射 |
| 用户组、时间目标与检索器 | `lightgcn.py` | 固定用户组、目标、历史、LightGCN 权重、完整验证和测试自然候选 |
| 严格未来屏蔽训练组 | `build_natural_train.py` | 三个历史截点的自然 Top-10 命中训练组 |
| 语义资产输入准备 | `datasets.py collect` | 经原始事件序列核验的三类 source、仅含长期历史的画像上下文 |
| 物品摘要生成 | `generate_assets.py prepare-items / parse-items` | 原始提示词、请求、原始返回、通过 schema 校验的物品描述 |
| 用户画像生成 | `generate_assets.py prepare-profiles / parse-profiles` | 历史提示词、请求、原始返回、通过证据约束的用户画像 |
| 请求执行 | `run_requests.py` | 支持断点续跑的同步请求、Batch 提交/下载和失败重试输入 |
| 语义训练样本 | `prepare.py`, `samples.py`, `datasets.py export` | train/valid/test、SFT 正负配对、完整候选与用户数统计 |
| 阶段串联 | `pipeline.py` | 三个可检查阶段，将上述实际实现串起来 |

### 数据筛选规则

使用原 CD 实验的顺序规则：元数据连接；标题和描述非空；初始用户交互下限；物品至少重复出现；每用户至少三个五星、两个四星交互；再次检查用户下限；每个 `(user_id, parent_asin)` 保留最新一次交互；保留 `rating >= 4`；迭代 user-10/item-15 k-core 至稳定。先按最新交互去重，再判断正反馈，不能反过来保留一个后来被低分覆盖的旧正例。

过滤后的 TSV 中 train/valid/test 是每个用户的时间位置标记，不是最终重排序样本的角色。`lightgcn.py` 再将用户按稳定 SHA256 分成 validation / test / train_only 三个 cohort；validation 和 test 用户互斥，各自最后一次正反馈作为外部目标。训练样本来自每用户更早的三个截点，所以训练样本用户可能与评估用户重合。对应用户的最终留出目标及训练截点之后的全部已知正反馈，不能作为该训练样本的目标、历史或负候选。

上游 `core/sft_records.jsonl` 是历史流程构造的含目标候选中间件，**不能直接作为最终训练集**。`build_natural_train.py` 重新训练各截点的检索器、屏蔽已见物品和全部未来正例，只保留目标自然进入 Top-10 的训练组；最终导出还会重新按原始事件验证时间前缀和未来屏蔽。

### 物品描述与画像如何提取

物品请求读取清洗后的标题、类别、品牌和描述，输出 `canonical_title / artists / genres / styles_or_moods / era / grounded_summary / knowledge_enrichment / compact_summary / uncertainties` 等字段。原始知识增强提示词随代码保存：输入支持的事实与模型先验补充分别放入 `grounded_summary` 和 `knowledge_enrichment`，摘要上限为 60 个英文空格分词。

画像只读取长期正反馈历史的物品语义，按旧到新顺序组织；最近三次正反馈留在近期历史字段。画像请求不包含用户 ID、目标、候选、LightGCN 分数或划分标签。输出长期偏好摘要、偏好类型、风格、证据数量和不确定性。单条证据只能标为 weak；证据数不得超过输入历史长度；至多两条长期记录只能为 sparse。空长期历史使用确定性空画像。校验不通过、缺少返回或重复返回时会停止阶段，不将失败内容当作有效训练资产。

请求、schema、来源摘要、返回模型、usage、提示词版本和内容哈希都会保留。发布前的导出会核对完整语义资产覆盖、画像历史与物品摘要哈希，防止接错画像或复用旧摘要的画像。

## 从原始文件运行

从仓库根目录执行，先安装 `requirements.txt`；LightGCN 支持 CPU / CUDA。默认参数对应当前 CD 示例的 user-10/item-15、三个截点、35 epoch 和 seed603759240。不同原始文件或数值环境不保证重现历史样本数。

### 1. 筛选、候选、历史上下文与物品请求

```bash
python -m semantic_context.pipeline prepare \
  --reviews data/raw/CDs_and_Vinyl.jsonl.gz \
  --metadata data/raw/meta_CDs_and_Vinyl.jsonl.gz \
  --output artifacts/cd_prepare --device cuda --asset-model gpt-4o
```

此阶段冻结配置并训练上游检索器，输出 `filtered/`、`core/`、`natural_train/`、`sources/` 和 `item_requests/`。检索参数预先固定，测试结果不参与任何参数选择；若要调参，应单独运行下层入口并仅用验证集选择，再从冻结检索器释放测试集。不要用测试召回率调整筛选阈值或重跑挑选数据划分。

### 2. 执行物品语义请求，再生成画像请求

配置服务地址与当前终端的 `OPENAI_API_KEY` 环境变量。凭据不写入仓库、配置文件或命令参数。请求执行是显式步骤，`pipeline` 本身只负责准备与校验数据。

```bash
export OPENAI_BASE_URL='https://api.openai.com/v1'
python -m semantic_context.run_requests sync \
  --requests artifacts/cd_prepare/item_requests/requests.jsonl \
  --output artifacts/cd_prepare/item_results.jsonl
python -m semantic_context.pipeline finish-items \
  --work artifacts/cd_prepare --results artifacts/cd_prepare/item_results.jsonl
```

若使用支持同样 Chat Completions 格式的服务，填写其地址。模型名由 `--asset-model` 设定；历史默认是 gpt-4o，并不要求替换为其他模型。`python -m semantic_context.run_requests --help` 可查看 Batch 和重试命令。中断后可对同一请求和输出重跑；已记录失败的请求先用 `retry-input` 单独生成重试文件。

### 3. 执行画像请求并导出全部数据集

```bash
python -m semantic_context.run_requests sync \
  --requests artifacts/cd_prepare/profile_requests/requests.jsonl \
  --output artifacts/cd_prepare/profile_results.jsonl
python -m semantic_context.pipeline finish-profiles \
  --work artifacts/cd_prepare --results artifacts/cd_prepare/profile_results.jsonl
```

如果所有长期历史都为空，画像请求文件为空，此时直接提供空结果 JSONL 给 `finish-profiles` 即可，确定性画像已经保存。所有阶段的输出路径需是新目录；失败后的校验报告和原始返回保留，使用下层入口向新目录重新校验，不覆盖已经冻结的成功产物。

## 最终产物和模块二衔接

```text
artifacts/cd_prepare/
├── filtered/                       # 交互 TSV、元数据 CSV 与筛选统计
├── core/                           # 用户组、检索器与完整评估候选
├── natural_train/                  # 严格未来屏蔽的自然训练组
├── sources/                        # 冻结的三类请求和数值 ID 画像历史
├── item_requests/, item_results.jsonl
├── items/item_summaries.jsonl       # 有效物品语义与生成来源
├── profile_requests/, profile_results.jsonl
├── profiles/user_profiles.jsonl    # 有效用户画像与生成来源
└── datasets/
    ├── train.jsonl                 # 自然命中训练组及语义上下文
    ├── train_pairs.jsonl           # 一正一负的语义 SFT 配对与权重
    ├── train_candidates.jsonl      # GRPO 完整十候选及 SFT 回放标记
    ├── valid.jsonl, test.jsonl     # 自然命中验证/测试组
    ├── valid_candidates.jsonl, test_candidates.jsonl
    ├── valid_all.jsonl, test_all.jsonl  # 包括召回未命中的完整评估用户
    └── manifest.json              # 全量/命中计数、Recall@10 与输入输出哈希
```

语义逐候选记录包含 `user_context`、`candidate_payload`、A/B 标签、候选原分数和 `_replay` 等字段，不包含 PNG。模块二直接渲染这些已经完成配对的样本，不再重新提取画像或抽取负例：

```bash
python -m optical_compression.build --samples artifacts/cd_prepare/datasets/train_candidates.jsonl --output artifacts/cd/train
python -m optical_compression.build --samples artifacts/cd_prepare/datasets/valid_candidates.jsonl --output artifacts/cd/valid
python -m optical_compression.build --samples artifacts/cd_prepare/datasets/test_candidates.jsonl --output artifacts/cd/test
```

随后将 `artifacts/cd/train/sft_pairs.jsonl` 交给 SFT，将 train 的 `candidates.jsonl` 交给 GRPO，将 valid/test 的完整候选交给固定模型评分。整体指标的用户分母必须取 `datasets/manifest.json` 的 `all_groups`；只统计命中子集不能代表全体用户结果。

已有冻结候选和语义资产时，也可分别调用 `datasets collect / export` 或原 `prepare.py` 入口；完整参数见各入口的 `--help`。
