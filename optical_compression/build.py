"""Build original compact CD pointwise SFT pairs and complete candidate groups."""
import argparse
import hashlib
import json
from pathlib import Path
from semantic_context.assets import load_validated_items, read_jsonl
from .renderer import render_compact

SYSTEM = ('You are a music recommendation relevance model. Inspect the DensePage containing '
          'a user profile and one candidate item. A means Relevant and B means Not relevant. '
          'Return exactly one letter: A or B.')
USER = 'Judge whether the candidate item is relevant to the user. Output only A or B.'


def materialize(samples, output):
    records, tasks = [], []
    for source in samples:
        row = dict(source)
        context, item = row.pop('user_context'), row.pop('candidate_payload')
        image = output / 'images' / (hashlib.sha256(
            (row['group_id'] + '\0' + row['candidate_item_id']).encode()).hexdigest() + '.png')
        row.update(densepage_path=str(image.resolve()), messages=[
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': str(image.resolve())},
                                       {'type': 'text', 'text': USER}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': row['label']}]}])
        records.append(row)
        tasks.append(dict(sample_id=row['sample_id'], profile=context, item=item, path=str(image)))
    group_order = {gid: i for i, gid in enumerate(dict.fromkeys(r['group_id'] for r in records))}
    pairs = sorted((r for r in records if r['_replay']), key=lambda r: (group_order[r['group_id']], r['label']))
    return pairs, records, tasks


def build(groups, items, output):
    from semantic_context.samples import pointwise
    _, samples = pointwise(groups, items)
    return materialize(samples, output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--groups', type=Path)
    source.add_argument('--samples', type=Path, help='Module-one semantic candidate JSONL')
    p.add_argument('--item-summaries', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.samples:
        samples = read_jsonl(a.samples)
        if not samples or len({r['sample_id'] for r in samples}) != len(samples):
            raise ValueError('empty or duplicate samples')
        pairs, complete, tasks = materialize(samples, a.output)
        group_count = len({r['group_id'] for r in samples})
    else:
        if not a.item_summaries:
            p.error('--item-summaries is required with --groups')
        groups = read_jsonl(a.groups)
        if not groups or len({g['sample_id'] for g in groups}) != len(groups):
            raise ValueError('empty or duplicate groups')
        pairs, complete, tasks = build(groups, load_validated_items(a.item_summaries), a.output)
        group_count = len(groups)
    a.output.mkdir(parents=True)
    rendered = [render_compact(t) for t in tasks]
    for name, records in [('sft_pairs', pairs), ('candidates', complete)]:
        (a.output / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    (a.output / 'manifest.json').write_text(json.dumps(dict(groups=group_count, pairs=len(pairs),
        candidates=len(complete), layout='compact640_font14', pages=rendered), indent=2) + '\n')


if __name__ == '__main__':
    main()
