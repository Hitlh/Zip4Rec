# 已选定的 CD 排版代码

[`compact.py`](compact.py) 是本项目选定并保留的原 CD `compact` 排版实现。当前选择状态见 [`selection.json`](selection.json)，后续批量渲染直接调用该代码。

页面宽度为 640px，正文为 DejaVuSans 14px，正文每行前进 16px，高度向上对齐到 32px 倍数。原大标题、蓝绿分区栏、字段展示方式和字体均保留。该代码只负责把用户上下文和单个候选物品绘制成图片，无 API 请求或模型权重依赖。

对模块一产生的三个数据划分直接调用选中代码：

```bash
python -m optical_compression.pipeline \
  --datasets artifacts/cd_prepare/datasets \
  --layout optical_compression/selected \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/cd_densepage
```

`--model` 只用于真实 processor token 与长度检查。单独在 Python 中绘制一张图片可调用：

```python
from optical_compression.selected.compact import render_compact

render_compact({
    'sample_id': 'example',
    'profile': user_context,
    'item': candidate_payload,
    'path': '/absolute/output/page.png',
})
```

`candidate_payload` 包含 `title`、`artists`、`genres`、`era`、`moods`、`summary`；列表字段须为列表。该策略使用 CD 字段结构。

对新数据集，先按 [agent 流程](../optimizer/README.md) 搜索一次，保存 `frozen/layout.py`、清单和字体，然后将批量命令的 `--layout` 改为这个 `frozen/` 目录即可。批量入口只调用选中代码，不重新搜索。新策略使用独立的输出目录并重新生成后续训练输入。
