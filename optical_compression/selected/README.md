# 已选中的 CD 排版代码

[compact.py](compact.py) 是本项目选中的 CD compact 排版实现，[selection.json](selection.json) 保存选择标记。

页面采用上下分区，宽 640px，正文 DejaVuSans 14px、每行前进 16px，高度向上对齐到 32px 倍数。代码将用户上下文与单个候选物品绘制为图片。

从仓库根目录运行：

```bash
python -m optical_compression.pipeline \
  --datasets artifacts/cd_prepare/datasets \
  --layout optical_compression/selected \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/cd_densepage
```

输入为模块一生成的 `datasets/`；输出为三个划分的图片、图文记录和实际 token 统计。接口 `render_compact(task)` 接收 `sample_id`、`profile`（用户文字）、`item`（候选物品）、`path`（输出 PNG 路径）。物品包含 `title`、`artists`、`genres`、`era`、`moods`、`summary`，其中 artists/genres/moods 为列表。

新数据集的选择流程见[模块二说明](../README.md)；选好后将批量命令的 `--layout` 指向对应 `frozen/` 目录。
