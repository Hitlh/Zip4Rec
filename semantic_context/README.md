# 模块一：数据与语义准备

从原始 reviews 和 metadata 开始，生成物品描述、用户画像以及可供后续渲染和训练的三个数据划分。

## 工作流程

1. 按物品标题/描述、用户交互数和评分组合筛选数据；同一用户与物品保留最新交互，以评分 ≥4 为正反馈，执行 k-core。
2. 按时间组织历史，固定互斥的验证/测试用户组；训练 LightGCN，生成不注入目标的自然 Top-10。训练使用更早的三个时间截点，并屏蔽未来正反馈。
3. 从清洗后的物品元数据生成结构化物品描述，再仅用长期历史中的物品描述生成用户画像；最近三次正反馈单独保留。
4. 校验语义返回，导出训练/验证/测试集、SFT 正负配对和 GRPO 十候选组。召回未命中的验证/测试用户也保存，用于整体指标。

## 输入格式

两个文件均为 UTF-8 JSONL 或 JSONL.GZ，一行一个对象，物品 ID 通过 `parent_asin` 对齐。

| 文件 | 必需字段 | 可选字段 |
| --- | --- | --- |
| reviews | `user_id`、`parent_asin`、`rating`、`timestamp` | 其他字段不参与当前流程 |
| metadata | `parent_asin`、非空 `title`、非空 `description`（字符串或列表） | `categories`、`store`、`details` |

单条记录的格式示例：

```json
{"user_id":"u1","parent_asin":"CD01","rating":5.0,"timestamp":1609459200000}
```

```json
{"parent_asin":"CD01","title":"Example Album","description":["A jazz recording."],"categories":["CDs & Vinyl","Jazz"],"store":"Example Artist"}
```

`rating` 使用 1–5 分；时间戳必须可转换为整数，并在整个数据集中保持同一单位。当前 CD 筛选要求每用户至少三条五星和两条四星交互，之后还须满足正反馈 k-core，默认 user-10 / item-15。输入必须有足够用户供验证/测试划分、有足够未交互物品供十候选召回；每个划分需要自然命中样本才能继续训练。

## 代码作用

| 文件 | 作用 |
| --- | --- |
| `pipeline.py` | 串联准备、物品返回校验、画像返回校验与数据导出 |
| `raw_filters.py`、`filter_data.py` | CD 原始条件筛选、最新交互去重、正反馈与 k-core |
| `extract_metadata.py` | 提取物品元数据，清洗 HTML 和空白，建立物品 ID 映射 |
| `lightgcn.py` | 时间目标与用户组划分、LightGCN 训练、自然 Top-10 召回 |
| `build_natural_train.py` | 构建更早时间截点的训练候选，屏蔽未来正反馈 |
| `datasets.py` | 收集历史上下文、检查时间边界、导出三个划分及完整评估用户 |
| `generate_assets.py` | 生成物品/画像请求，解析并校验返回 |
| `llm_profile_common.py` | 领域提示词、JSON schema、语义校验和文件读写 |
| `run_requests.py` | 执行兼容 API 请求，支持并发、已完成请求续跑和失败请求重试 |
| `assets.py` | 加载有效物品描述、格式化用户画像 |
| `prepare.py` | 组织用户历史、画像和候选上下文 |
| `samples.py` | 生成逐候选记录、确定性 SFT 正负配对与回放权重 |

## 如何运行

从仓库根目录执行；环境及 API 变量按[根 README](../README.md) 设置。

```bash
# 1. 筛选原始数据、训练检索器、准备物品描述请求。
python -m semantic_context.pipeline prepare \
  --reviews data/raw/CDs_and_Vinyl.jsonl.gz \
  --metadata data/raw/meta_CDs_and_Vinyl.jsonl.gz \
  --output artifacts/cd_prepare --device cuda --asset-model gpt-4o

# 2. 生成物品描述，再准备用户画像请求。
python -m semantic_context.run_requests sync \
  --requests artifacts/cd_prepare/item_requests/requests.jsonl \
  --output artifacts/cd_prepare/item_results.jsonl
python -m semantic_context.pipeline finish-items \
  --work artifacts/cd_prepare --results artifacts/cd_prepare/item_results.jsonl

# 3. 生成用户画像，导出训练集、验证集和测试集。
python -m semantic_context.run_requests sync \
  --requests artifacts/cd_prepare/profile_requests/requests.jsonl \
  --output artifacts/cd_prepare/profile_results.jsonl
python -m semantic_context.pipeline finish-profiles \
  --work artifacts/cd_prepare --results artifacts/cd_prepare/profile_results.jsonl
```

`prepare --help` 可查看筛选下限、种子、召回训练轮数、batch size 和验证/测试用户数。默认验证/测试各 1000 用户，小数据集需降低 `--validation-users` 和 `--test-users`。使用新输出目录开始准备。

如果长期历史均为空，画像请求也为空，此时创建空的 `profile_results.jsonl` 后执行 `finish-profiles`；总入口自动处理这种情况。返回不通过校验时查看 `items/` 或 `profiles/` 的报告；请求工具的完整命令见 `python -m semantic_context.run_requests --help`。

## 输出与下一步

`artifacts/cd_prepare/` 下保存：

| 路径 | 内容 |
| --- | --- |
| `filtered/` | 筛选交互、物品元数据与统计 |
| `core/`、`natural_train/`、`sources/` | 检索器、时间目标、候选、历史与画像上下文 |
| `item_requests/`、`profile_requests/` | 语义请求及来源；API 返回保存为 `*_results.jsonl` |
| `items/`、`profiles/` | 通过校验的物品描述和用户画像 |
| `datasets/train.jsonl`、`valid.jsonl`、`test.jsonl` | 自然命中组及其语义上下文 |
| `datasets/*_candidates.jsonl` | 对应划分的逐候选记录，含 `user_context` 和 `candidate_payload`，交给模块二 |
| `datasets/train_pairs.jsonl` | 一正一负的 SFT 配对与权重 |
| `datasets/valid_all.jsonl`、`test_all.jsonl` | 含召回未命中的完整评估用户 |
| `datasets/manifest.json` | 样本计数、召回率与文件校验信息 |

将整个 `datasets/` 目录交给[模块二](../optical_compression/README.md)。跨领域数据的调整位置见[根 README](../README.md#更换数据集)。
