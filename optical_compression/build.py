"""Build original compact CD pointwise SFT pairs and complete candidate groups."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from semantic_context.assets import load_validated_items, read_jsonl
from .renderer import render_compact

SYSTEM = ('You are a music recommendation relevance model. Inspect the DensePage containing '
          'a user profile and one candidate item. A means Relevant and B means Not relevant. '
          'Return exactly one letter: A or B.')
USER = 'Judge whether the candidate item is relevant to the user. Output only A or B.'


def build(groups, items, output):
    pairs, complete, tasks = [], [], []
    for group in groups:
        candidates = list(map(str, group['candidate_item_ids']))
        target = str(group['target_item_id'])
        if (len(candidates) != 10 or len(set(candidates)) != 10 or target not in candidates
                or len(group['candidate_scores']) != 10
                or not all(math.isfinite(float(s)) for s in group['candidate_scores'])):
            raise ValueError('renderer expects ten-candidate natural-hit groups')
        if group.get('target_injected', False) or set(candidates) & set(map(str, group.get('history_item_ids', []))):
            raise ValueError('invalid natural candidates')
        ranked = sorted(zip(candidates, group['candidate_scores']), key=lambda x: (-float(x[1]), x[0]))
        negatives = [item for item, _ in ranked if item != target]
        offset = int(hashlib.sha256(group['sample_id'].encode()).hexdigest()[:16], 16) % len(negatives)
        negative = negatives[offset]
        text = next(c['text'] for m in group['messages'] if m['role'] == 'user'
                    for c in m['content'] if c['type'] == 'text')
        profile = text.split('\n\nCandidate music items:', 1)[0].strip()
        for position, (item, score) in enumerate(zip(candidates, group['candidate_scores']), 1):
            # Hash filenames so externally supplied identifiers cannot escape the output folder.
            image = output / 'images' / (hashlib.sha256(
                (group['sample_id'] + '\0' + item).encode()).hexdigest() + '.png')
            label = 'A' if item == target else 'B'
            rank = next(i for i, (candidate, _) in enumerate(ranked, 1) if candidate == item)
            raw_weight = 1. if label == 'A' else .2 if rank <= 3 else .5 if rank <= 6 else 1.
            row = dict(sample_id=f"{group['sample_id']}-{label}-{item}", group_id=group['sample_id'],
                       split=group['split'], user_id=group['user_id'], target_item_id=target,
                       candidate_item_id=item, candidate_position=position, candidate_score=score,
                       label=label, densepage_path=str(image.resolve()),
                       _replay=(item in [target, negative]), raw_sample_weight=raw_weight,
                       messages=[{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                                 {'role': 'user', 'content': [{'type': 'image', 'image': str(image.resolve())},
                                                           {'type': 'text', 'text': USER}]},
                                 {'role': 'assistant', 'content': [{'type': 'text', 'text': label}]}])
            complete.append(row)
            if item in [target, negative]:
                pairs.append(row)
            tasks.append(dict(sample_id=row['sample_id'], profile=profile, item=items[item], path=str(image)))
    negative_weights = [r['raw_sample_weight'] for r in pairs if r['label'] == 'B']
    mean = sum(negative_weights) / len(negative_weights) if negative_weights else 1.
    for row in complete:
        row['sample_weight'] = row['raw_sample_weight'] / mean if row['label'] == 'B' else 1.
        row['_replay_weight'] = row['sample_weight']
    # SFT is sequential: retain the source builder's positive-then-negative pair order.
    group_order = {g['sample_id']: i for i, g in enumerate(groups)}
    pairs.sort(key=lambda r: (group_order[r['group_id']], r['label']))
    return pairs, complete, tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--item-summaries', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    groups = read_jsonl(a.groups)
    if not groups or len({g['sample_id'] for g in groups}) != len(groups):
        raise ValueError('empty or duplicate groups')
    pairs, complete, tasks = build(groups, load_validated_items(a.item_summaries), a.output)
    a.output.mkdir(parents=True)
    rendered = [render_compact(t) for t in tasks]
    for name, records in [('sft_pairs', pairs), ('candidates', complete)]:
        (a.output / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    (a.output / 'manifest.json').write_text(json.dumps(dict(groups=len(groups), pairs=len(pairs),
        candidates=len(complete), layout='compact640_font14', pages=rendered), indent=2) + '\n')


if __name__ == '__main__':
    main()
