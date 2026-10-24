# Zip4Rec

Zip4Rec 是面向推荐重排序的视觉语言模型框架：将用户历史语义排版为紧凑的 **DensePage**，通过相关性监督微调（SFT）和面向候选排列的 **GRPO** 训练模型，再融合推荐模型分数与语义分数完成重排序。

当前仓库提供 **Amazon Reviews 2023 CDs & Vinyl + LightGCN + Qwen3-VL-8B-Instruct** 的完整流程。原始数据、处理后的数据、模型权重及运行产物由使用者在本地准备和生成。

## 方法与代码对应

| 模块 | 目录 | 实现内容 |
| --- | --- | --- |
| 数据与历史语义 | [`semantic_context/`](semantic_context/) | 筛选、去重、k-core、时间目标与自然候选、物品描述、用户画像、三类数据集和正负配对 |
| 光学压缩 | [`optical_compression/`](optical_compression/) | 一次性 agent 排版选择、固定代码批量渲染、图文输入和实际 processor token 检查 |
| 推荐重排序 | [`ranking/`](ranking/) | 语言 LoRA 的 SFT、GRPO、验证集选择、固定模型评分、融合排序和指标计算 |

```text
原始 reviews + metadata
  → 筛选、去重与 k-core
  → 冻结验证/测试用户组和时间目标，生成 LightGCN 自然 Top-10
  → 提取物品语义、历史用户画像，构造训练/验证/测试样本
  → agent 选择并冻结排版代码（已有选中代码可直接复用）
  → 单独调用选中代码，批量生成 DensePage 和图文输入
  → SFT → 验证选择 → GRPO → 验证选择
  → 锁定模型和融合系数 → 测试评分、排序与评估
```

每条评分输入包含一个用户的历史上下文及一个候选物品。A/B 输出分别对应交互正例和未交互弱负例。适配器相对未适配基座的增量语义分数为：

```text
s(u, i) = [logit_A - logit_B]_adapter - [logit_A - logit_B]_base
final(u, i) = z_u(LightGCN_score) + α × z_u(s(u, i))
```

`z_u` 在同一用户的十候选内使用总体标准差；推理时常数分数回退为零，同分按物品 ID 升序。CD 配置固定 `α=1.5`。GRPO 从 Plackett–Luce 策略采样排列，以单正例 NDCG@10 为奖励，使用组相对优势、逐步裁剪目标、冻结 SFT 的首选分类 KL 和加权 SFT 回放更新 LoRA。候选数为 **10**，每用户采样排列数为 **8**。

## 环境

原 CD 训练记录的核心环境如下；依赖文件按这些版本设置：

| 依赖 | 参考版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch / CUDA | 2.5.1+cu124 / 12.4 |
| Transformers | 5.8.0 |
| PEFT / Accelerate | 0.19.1 / 1.13.0 |
| Attention | SDPA |
| 训练精度 | BF16 基座、FP32 LoRA 与受限 A/B 输出头、FP64 排序策略 |

在独立 Python 环境中安装与显卡驱动匹配的 PyTorch，再安装其余依赖。例如使用 CUDA 12.4：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

GPU 训练限定单张可见 CUDA 设备，要求支持 BF16。字体和字体许可随 `optical_compression/fonts/` 提供，以固定页面排版。

模块二采用“先选择一次、后固定调用”的逻辑：新数据集从十条真实训练样本运行 [视觉排版 agent](optical_compression/optimizer/README.md)，兼顾训练可识别性与 token 成本，导出选中的 `frozen/layout.py`、字体和清单；后续生成 train/valid/test 图片时只调用选中代码，不再请求 agent。上下分区、大标题、蓝绿分区栏和字体字号固定，回读门槛默认 90%。

当前仓库单独保留项目选定的 [`selected/compact.py`](optical_compression/selected/compact.py)，并通过 [选择清单](optical_compression/selected/selection.json) 标注。后续直接调用这个选中代码进行批量渲染。

## 数据与基座

- 从 [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) 获取 CDs & Vinyl 的 reviews 和 metadata。reviews 使用 `user_id / parent_asin / rating / timestamp`；metadata 使用 `parent_asin / title / description / categories / store` 等字段。
- 将 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 的权重、tokenizer 和 processor 下载到本地模型目录。
- 配置用于物品描述和用户画像生成的服务。参考生成模型为 `gpt-4o`，模型名称可在配置中设置。

已选 compact 的复现参数见 [`pipeline.json`](configs/cd_lightgcn/pipeline.json)，新数据集先搜索再渲染的参数见 [`pipeline_agent.json`](configs/cd_lightgcn/pipeline_agent.json)。数据和权重不随本仓库发布。

## 完整运行

从仓库根目录执行，在配置中填写本地 reviews、metadata、基座目录和输出目录。设置 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 环境变量，语义生成请求使用配置的服务和账户。

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENAI_BASE_URL='https://api.openai.com/v1'
# 新数据集：语义准备 → agent 选择 → 固定代码渲染 → 训练与评估。
python run_pipeline.py --config configs/cd_lightgcn/pipeline_agent.json

# 原 CD 复现：复用仓库已选的 compact，直接渲染和训练。
python run_pipeline.py --config configs/cd_lightgcn/pipeline.json
```

此入口依次运行三个模块，生成自己的语义资产、图文数据、SFT 适配器、GRPO 参考缓存、检查点、验证分数及测试结果。阶段参数、产物路径和评估协议见 [CD / LightGCN 复现指南](configs/cd_lightgcn/README.md)。

主要输出：

```text
artifacts/<本次运行>/
├── semantic/       # 数据处理、语义生成请求/响应、画像和数据划分
├── layout_search/  # 选择模式产生；搜索记录和 frozen/ 选中代码
├── densepage/      # 固定代码生成的三类页面、模型输入、实际 token 统计
└── ranking/
    ├── sft/        # 本次 SFT 权重和检查点
    ├── grpo/       # GRPO 协议、参考缓存和训练权重
    ├── validation/ # 验证分数与选择依据
    ├── test/       # 测试分数、逐用户排序和指标
    └── results.json
```

## 评估与复现范围

验证与测试用户组互斥。训练可使用相同用户更早的时间截点；最终留出事件和该截点之后的已知正反馈必须排除。最终训练及评估使用自然 Top-10，不注入目标。检查点与融合系数在测试评分前锁定。

指标区分目标已召回用户的条件结果与完整评估用户的整体结果。单正例设置下，整体 NDCG/MRR/H 等于自然 Recall@10 乘以相应条件指标。输出保留两类指标及完整用户数。

公开 CD 配置对应的参考规模为 1,258 个训练组、2,516 条 SFT 配对、101 名验证命中用户、108 名测试命中用户；测试全集为 1,000 用户。原实验 GRPO checkpoint-237 的融合条件 NDCG@10 为 **0.573137954**，整体为 **0.061898899**。这些是单种子实验参考值，新运行应根据自己的验证分数选择检查点。

当前公开入口已通过数据、渲染、工作流和策略数值测试，尚未对发布后的完整入口重跑全量语义服务调用与 Qwen GPU 训练。语义模型的版本和生成内容、输入数据版本、processor 与运行环境都会影响结果；应保存本次请求/响应、内容哈希和模型选择记录。Pillow 和 httpx 的版本尚未完全锁定。上述参考分数不代表任意重新生成的语义资产均能逐位复现。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试在临时目录生成微型合成输入，覆盖时间前缀、未来正例屏蔽、资产校验、页面渲染、配对/回放、PL 概率与梯度、GRPO 优势、裁剪目标、工作流与指标分母。缺少 PyTorch 时跳过相应项目；完整检查应使用已安装全部依赖的环境。
