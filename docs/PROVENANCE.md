# 代码来源与整理范围

代码从本地实验项目整理，原始文件相对实验根目录的路径与 SHA256 见 `source_manifest.json`。仓库结构固定为历史语义、光学压缩、重排序三个模块。原项目保留原状。

| 新位置 | 原始实现 |
| --- | --- |
| `semantic_context/lightgcn.py` | `amazon2023_user_disjoint_protocol.py`，完整原始上游实现 |
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
