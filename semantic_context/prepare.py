"""Join frozen natural candidates with validated history-only semantic assets."""
import argparse
import copy
import math
from pathlib import Path
from .assets import load_profiles, load_validated_items, read_jsonl
import json

RECENT_MARKER = '\n\nRecent positive interactions (oldest to newest):\n'


def organize(group, profiles, items):
    history = list(map(str, group['history_item_ids']))
    recent = list(map(str, group['recent_history_item_ids']))
    long = list(map(str, group['long_history_item_ids']))
    candidates = list(map(str, group['candidate_item_ids']))
    target = str(group['target_item_id'])
    if group.get('target_injected', False):
        raise ValueError('injected targets are outside the natural-candidate protocol')
    if history != long + recent or (recent and history[-len(recent):] != recent):
        raise ValueError('history must be ordered long-term followed by recent records')
    if target in history or set(candidates) & set(history):
        raise ValueError('target/candidates must be excluded from available history')
    if len(candidates) != 10 or len(set(candidates)) != 10:
        raise ValueError('CD example requires ten unique natural candidates')
    if len(group['candidate_scores']) != 10 or not all(math.isfinite(float(s)) for s in group['candidate_scores']):
        raise ValueError('invalid candidate scores')
    profile = profiles[str(group['profile_history_key'])]
    if recent:
        profile += RECENT_MARKER + '\n'.join(
            f"R{i}: {items[item]['title']} [summary: {items[item]['summary']}]"
            for i, item in enumerate(recent, 1))
    row = copy.deepcopy(group)
    row.update(candidate_item_ids=candidates, target_item_id=target)
    row['messages'] = [{'role': 'user', 'content': [{'type': 'text',
        'text': profile + '\n\nCandidate music items:'}]}]
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--item-summaries', type=Path, required=True)
    p.add_argument('--user-profiles', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    profiles, items = load_profiles(a.user_profiles), load_validated_items(a.item_summaries)
    records = [organize(g, profiles, items) for g in read_jsonl(a.groups)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
