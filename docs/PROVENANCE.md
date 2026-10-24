# 代码来源与整理范围

代码从本地实验项目整理，原始文件相对实验根目录的路径与 SHA256 见 `source_manifest.json`。仓库结构固定为历史语义、光学压缩、重排序三个模块。原项目保留原状。

| 新位置 | 原始实现 |
| --- | --- |
| `semantic_context/lightgcn.py` | `amazon2023_user_disjoint_protocol.py`；原始数值实现保留，固定样本量断言改为按配置核验 |
| `semantic_context/assets.py` | 历史长度实验归档的 `original_renderer.py` 内资产加载和字段格式化函数 |
| `optical_compression/renderer.py` | 同一归档的 compact 渲染及排版函数 |
| `ranking/train_sft.py` | `condition_pointwise_experiment/train_condition_pointwise.py` |
| `ranking/sft_support.py`, `conversation.py` | 原 native10 文件中的共享加载、受限输出头和 conversation 函数，公开接口收敛为 A/B |
| `ranking/train_grpo.py`, `policy.py`, `exact_policy.py`, `model_io.py`, `prepare_reference.py`, `common.py` | `condition_pointwise_experiment/cd_lightgcn_grpo_k_20261022/code/` |
| `ranking/test_policy.py` | 同目录已有策略数值与梯度核验 |

新增可移植连接入口包括 `semantic_context/prepare.py`、`optical_compression/build.py`、`ranking/make_run.py`、`score.py`、`select.py`、`rerank.py` 和 `evaluate.py`，它们复用原始字段与数值规则，去掉服务器调度。

迁移改动：相对包导入替换实验绝对目录；字体随仓库保存；compact 中未定义的 `cfg` 改为局部空字典（与已核验历史长度实验 wrapper 一致）；A/B 口令校验只校验当前使用的两类；GPU 与路径由运行环境及命令提供。GRPO 检查点按原比例计算以支持不同组数；本机 cache 绑定实际模型与运行环境。参考预处理的历史分数验证变为显式可选参数。原 SFT 的顺序采样、负例权重、delta CE 以及 GRPO 损失均保留。

不发布实验服务器调度脚本、登录信息、全量用户记录、PNG 缓存、模型或 LoRA 权重。合成示例与历史数值记录明确分开。原始共享策略中的其他方法辅助函数仅为保留数值核验，公开训练入口只接受 GRPO。

此仓库是整理后的可移植研究代码，精确重现实验还需原始冻结输入、权重与相同数值环境；本次未重跑完整 GPU SFT/GRPO。验证情况记录于 `VALIDATION.md`。

## 模块一完整数据链路补充

新增原始顺序筛选、去重/k-core、元数据抽取、严格未来屏蔽训练候选、物品摘要/用户画像原始提示词和 schema、同步/Batch 请求执行、返回校验、历史数值 ID 映射与三类数据集导出。新增源文件哈希继续记录于 `source_manifest.json`。

`raw_filters.py` 提取自原始 raw preprocessing 的实际筛选函数；`filter_data.py` 和 `extract_metadata.py` 沿用同目录实现；`build_natural_train.py` 来自 `baseline/strict_temporal/build_lightgcn_strict.py`，仅改包导入及源训练文件参数别名；`llm_profile_common.py` 沿用原提示词/校验；`generate_assets.py` 提取原资产生成和解析函数并提供收敛后的四命令入口；`run_requests.py` 来自原 OpenAI-compatible 请求执行器，仅改包导入。

新 `datasets.py` 在外部生成前检查原始时间前缀与未来正例屏蔽，在导出前核对语义资产覆盖、画像输入及内容哈希。`pipeline.py` 串联各阶段。`samples.py` 将原 `optical_compression/build.py` 的配对和权重生成移到模块一；模块二新 `--samples` 入口直接渲染已经构造好的训练/验证/测试语义记录，保留旧 `--groups` 接口。

修正了上一版误将 CD 原始协议当作训练/验证用户完全互斥的限制。原始代码是在所有用户更早截点构造训练组，验证与测试 cohort 互斥，依靠屏蔽最终留出事件和未来正反馈防止时间泄漏；现在 `ranking/make_run.py` 按该实际协议检查。该修正不改变 GRPO 数值损失或现有模型权重。

原始上游中间 `sft_records.jsonl` 含目标注入列表，仅用来关联历史目标与样本 ID。最终数据一律来自严格自然候选构建，不发布该中间件为可训练语义样本。新入口可在小数据上测试，历史固定规模 3174/2588/9522/11522 不作为其他数据的硬限制；实际原始数据与运行环境不变仍是精确复现历史结果的必要条件。

## 三模块端到端入口

新增根目录 `run_pipeline.py` 与可编辑的 `examples/cd_lightgcn/pipeline.json`，串联原始数据准备、两轮语义生成请求与校验、模块二及模块三。模块二 `pipeline.py` 调用实际 renderer 构建三个划分，`measure_tokens.py` 使用真实模型 processor 编码所有输入并核验长度。模块三 `pipeline.py` 调用现有数值训练实现，从基座新建 SFT LoRA，在本次验证集选择后生成 GRPO 参考并继续训练，再锁定两个模型进行测试。`workflow.py` 保留阶段命令、状态与文件哈希；不复用任何旧实验适配器或评分作为完整流程的前置输入。
