# 模块一：历史语义整理

本模块接收已验证的离线语义资产，按冻结历史构建每个用户请求的上下文。画像生成本身依赖外部离线流程；本仓库不自动调用模型服务。画像必须仅依据对应 `profile_history_key` 的可用历史，不能包含目标交互。

- `assets.py`：原实验的物品语义、画像字段格式化与读取。
- `prepare.py`：连接候选组、长期画像和近期交互，检查自然候选与历史边界。
- `lightgcn.py`：原始 CD 用户互斥、时间划分与 LightGCN 上游实现，可独立运行；精确复现实验时优先使用已冻结候选。

原始组字段为 `sample_id, split, user_id, target_item_id, candidate_item_ids, candidate_scores, history_item_ids, long_history_item_ids, recent_history_item_ids, profile_history_key`。当前示例限定十候选，历史按时间从旧到新排列；目标不得注入候选或出现在历史中。`prepare` 可以保留目标未命中的组，进入命中子集渲染前须另行筛选，并保留完整用户数用于整体评估。

`item_summaries.jsonl` 以 `original_item_id` 为键，含 `title` 与结构化 `summary`；`user_profiles.jsonl` 以 `profile_history_key` 为键，含结构化 `profile`。完整最小样例见 `examples/cd_lightgcn/`。历史边界与画像生成依据无法只凭生成后的字符串证明，应随真实资产保留生成记录。

```bash
python -m semantic_context.prepare --groups examples/cd_lightgcn/train_groups.jsonl --item-summaries examples/cd_lightgcn/item_summaries.jsonl --user-profiles examples/cd_lightgcn/user_profiles.jsonl --output artifacts/demo/train_context.jsonl
```

`python -m semantic_context.lightgcn --help` 查看原始上游参数。它保留原实验的多个候选协议，不能将其中的训练负采样列表当作自然召回命中列表。
