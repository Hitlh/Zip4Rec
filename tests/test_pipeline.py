import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from semantic_context.assets import load_profiles, load_validated_items, read_jsonl
from semantic_context.prepare import organize
from optical_compression.build import build
from optical_compression.renderer import render_compact
from ranking.evaluate import evaluate
from ranking.rerank import rerank

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples/cd_lightgcn'


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.items = load_validated_items(EXAMPLES / 'item_summaries.jsonl')
        self.profiles = load_profiles(EXAMPLES / 'user_profiles.jsonl')
        self.group = read_jsonl(EXAMPLES / 'train_groups.jsonl')[0]

    def test_reject_history_and_candidate_leakage(self):
        for field, value in [('target_injected', True), ('target_item_id', 'CD10'),
                             ('recent_history_item_ids', ['CD11'])]:
            bad = dict(self.group, **{field: value})
            with self.assertRaises(ValueError):
                organize(bad, self.profiles, self.items)
        bad = copy.deepcopy(self.group)
        bad['candidate_item_ids'][0] = 'CD10'
        with self.assertRaises(ValueError):
            organize(bad, self.profiles, self.items)

    def test_sft_pair_matches_grpo_replay_and_page_is_label_free(self):
        context = organize(self.group, self.profiles, self.items)
        with tempfile.TemporaryDirectory() as folder:
            pairs, candidates, tasks = build([context], self.items, Path(folder))
            self.assertEqual([r['label'] for r in pairs], ['A', 'B'])
            self.assertEqual(len(candidates), 10)
            self.assertEqual({r['sample_id'] for r in pairs},
                             {r['sample_id'] for r in candidates if r['_replay']})
            negative = pairs[1]
            self.assertEqual(negative['sample_weight'], 1.)
            for row, task in zip(candidates, tasks):
                self.assertEqual(set(task), {'sample_id', 'profile', 'item', 'path'})
                self.assertNotIn('Candidate music items:', task['profile'])
                self.assertEqual(len([m for m in row['messages'] if m['role'] == 'assistant']), 1)
            page = render_compact(tasks[0])
            self.assertEqual(page['width'], 640)
            self.assertEqual(page['height'] % 32, 0)
            self.assertTrue(Path(tasks[0]['path']).is_file())

    def test_ranking_without_labels_and_metric_denominators(self):
        scores = [dict(sample_id=str(i), group_id='one', user_id='u', candidate_item_id=f'{i:02}',
                       target_item_id='02', label='A' if i == 2 else 'B',
                       candidate_score=10-i, condition_score=0.) for i in range(10)]
        result = evaluate(scores, alpha=0., all_users=100)
        self.assertAlmostEqual(result['fused']['NDCG@10'], 1 / math.log2(4))
        self.assertAlmostEqual(result['fused']['MRR@10'], 1 / 3)
        self.assertAlmostEqual(result['end_to_end_fused']['NDCG@10'], .005)
        unlabeled = [{k: v for k, v in row.items() if k not in ['label', 'target_item_id']} for row in scores]
        self.assertEqual(rerank(unlabeled, 0.)[0]['ranking'], result['rankings'][0]['ranking'])
        with self.assertRaises(ValueError):
            evaluate(scores, all_users=0)
        with self.assertRaises(ValueError):
            rerank([dict(r, condition_score=float('nan')) for r in unlabeled])

    def test_deterministic_ties_and_invalid_groups(self):
        scores = [dict(sample_id=str(i), group_id='g', user_id='u', candidate_item_id=f'{i:02}',
                       target_item_id='02', label='A' if i == 2 else 'B',
                       candidate_score=0., condition_score=0.) for i in reversed(range(10))]
        self.assertEqual(rerank(scores)[0]['ranking'], [f'{i:02}' for i in range(10)])
        with self.assertRaises(ValueError):
            evaluate(scores[:-1])
        scores[0]['sample_id'] = scores[1]['sample_id']
        with self.assertRaises(ValueError):
            evaluate(scores)


if __name__ == '__main__':
    unittest.main()
