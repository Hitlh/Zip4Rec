"""Offline checks for agent execution, immutable content, qualification, and integration."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

from optical_compression.document import document, fingerprint, plain_text, select_examples
from optical_compression.layout_engine import (Constraints, FrozenLayout, execute_layout,
    render_plan, validate_source)
from optical_compression.measurement import reading_result
from optical_compression.optimizer.agent import ChatClient, TOOLS, run_agent
from optical_compression.optimizer.tools import LayoutSession
from workflow import digest

ROOT = Path(__file__).resolve().parents[1]
INITIAL = (ROOT / 'optical_compression/layouts/initial.py').read_text()
PACKED = INITIAL.replace('max(640,', 'max(480,').replace('pack=False', 'pack=True').replace("'line_gap': 4", "'line_gap': 1")


def examples():
    output = []
    for i in range(10):
        doc = document(f'Interests: acoustic music {i}.\nEvidence: recent listening.',
                       dict(title=f'Album {i}', artists=['A', 'B'], summary='Detailed music description. ' * (i + 1)))
        output.append(dict(id=f'example_{i+1:02d}', document=doc, document_sha256=fingerprint(doc)))
    return output


class FakeMeter:
    identity = dict(processor_files={'processor_config.json': 'offline'}, protocol_sha256='fixed')

    def text(self, doc):
        return dict(total_input_tokens=5000)

    def image(self, path):
        with Image.open(path) as image:
            visual = (image.width * image.height + 1023) // 1024
        return dict(total_input_tokens=visual + 50, image_patch_tokens=visual,
                    non_image_patch_tokens=50, image_grid_thw=[[1, 10, 20]])


class FakeReader:
    def __init__(self):
        self.results = {}
        self.calls = 0

    def read(self, path):
        self.calls += 1
        return self.results[str(path)]

    def identity(self):
        return dict(model_weights_sha256={'model.safetensors': 'offline'})


def provide_readings(session, cid):
    for row in session.candidates[cid]['report']['records']:
        session.reader.results[row['image']] = json.dumps({k: ' '.join(v) for k, v in row['sections'].items()})


class LayoutOptimizerTests(unittest.TestCase):
    def session(self, directory, **kwargs):
        return LayoutSession(examples(), Constraints(), FakeMeter(), FakeReader(), directory, **kwargs)

    def test_general_code_preserves_long_words_and_top_bottom_sections(self):
        doc = document('Interests: ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' * 15,
                       dict(title='Title', notes=['Short note', 'Longer note']))
        cfg = Constraints()
        plan = execute_layout(PACKED, [doc], cfg)[0]
        with tempfile.TemporaryDirectory() as folder:
            result = render_plan(doc, plan, cfg, Path(folder) / 'image.png')
        self.assertTrue(result['geometry_valid'])
        for section in ['user', 'item']:
            actual = ''.join(''.join(result['sections'][section]).split())
            expected = ''.join(''.join(f['text'] for f in doc[section]).split())
            self.assertEqual(actual, expected)
        self.assertLess(plan['width'], 640)

    def test_code_can_choose_width_by_input_length(self):
        source = PACKED.replace('width = min(', "size = sum(len(f['text']) for f in document['user'])\n    width = min(")
        source = source.replace('max(480,', 'max(480 if size < 200 else 800,')
        docs = [document('Short', {'title': 'T'}), document('Word ' * 100, {'title': 'T'})]
        plans = execute_layout(source, docs, Constraints())
        self.assertEqual([p['width'] for p in plans], [480, 800])

    def test_removed_duplicated_reordered_text_and_wrong_section_are_rejected(self):
        doc = examples()[0]['document']
        plan = execute_layout(INITIAL, [doc], Constraints())[0]
        with tempfile.TemporaryDirectory() as folder:
            for change in ['remove', 'duplicate', 'swap', 'truncate']:
                modified = json.loads(json.dumps(plan))
                if change == 'remove':
                    modified['user_rows'] = modified['user_rows'][1:]
                elif change == 'duplicate':
                    modified['user_rows'] += modified['user_rows'][:1]
                elif change == 'swap':
                    modified['user_rows'], modified['item_rows'] = modified['item_rows'], modified['user_rows']
                else:
                    modified['user_rows'][0][0][2] -= 1
                with self.subTest(change=change), self.assertRaises(ValueError):
                    render_plan(doc, modified, Constraints(), Path(folder) / 'bad.png')

    def test_font_geometry_and_restricted_source_rules(self):
        for source in ['import os\n' + INITIAL,
                       INITIAL.replace('metrics.flow', 'metrics.__class__'),
                       INITIAL.replace('padding = 12', "padding = 12\n    open('/tmp/oops', 'w')"),
                       INITIAL.replace('constraints):', 'constraints=1):')]:
            with self.subTest(source=source[:60]):
                with self.assertRaises((ValueError, NameError)):
                    validate_source(source)
                    execute_layout(source, [examples()[0]['document']], Constraints())
        doc = examples()[0]['document']
        plan = execute_layout(INITIAL, [doc], Constraints())[0]
        plan['line_gap'] = -1
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            render_plan(doc, plan, Constraints(), Path(folder) / 'bad.png')
        doc = document('\U0010ffff', {'title': 'T'})
        plan = execute_layout(INITIAL, [doc], Constraints())[0]
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'no glyph'):
            render_plan(doc, plan, Constraints(), Path(folder) / 'bad.png')

    def test_worker_timeout_is_contained(self):
        source = 'def layout(document, metrics, constraints):\n    while True:\n        pass\n'
        with self.assertRaises(subprocess.TimeoutExpired):
            execute_layout(source, [examples()[0]['document']], Constraints(), timeout=.3)

    def test_training_selection_strips_labels_and_selects_length_extremes(self):
        rows = [dict(split='train', sample_id=f'{i}-A-target-secret', label='A', target_item_id='secret',
                     user_context=f'Context {i}', candidate_payload={'summary': 'word ' * (i + 1)}) for i in range(30)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            selected = select_examples(path)
            self.assertEqual(len(selected), 10)
            self.assertNotIn('secret', json.dumps(selected))
            self.assertIn('Context 0', json.dumps(selected))
            self.assertIn('Context 29', json.dumps(selected))
            rows[0]['split'] = 'test'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, 'split=train'):
                select_examples(path)

    def test_token_equality_and_one_failing_sample_block_export(self):
        with tempfile.TemporaryDirectory() as folder:
            session = self.session(Path(folder) / 'search')
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            session.baselines['example_01'] = 1
            report = session.render_and_measure(cid)
            provide_readings(session, cid)
            self.assertFalse(report['token_qualified'])
            self.assertFalse(session.check_readability(cid)['qualified'])
            self.assertIsNone(session.freeze({}))
            self.assertFalse((session.output / 'frozen').exists())
        with tempfile.TemporaryDirectory() as folder:
            session = self.session(Path(folder) / 'equality')
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            original = session.meter.image
            session.meter.image = lambda path: dict(original(path), total_input_tokens=5000)
            self.assertFalse(session.render_and_measure(cid)['token_qualified'])

    def test_readback_failure_and_unverified_baselines_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as folder:
            session = self.session(Path(folder) / 'search')
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            session.render_and_measure(cid)
            provide_readings(session, cid)
            session.reader.results[next(iter(session.reader.results))] = '{"user":"missing","item":"wrong"}'
            self.assertFalse(session.check_readability(cid)['qualified'])
            self.assertIsNone(session.freeze({}))
            with self.assertRaisesRegex(ValueError, 'pure-text counts'):
                self.session(Path(folder) / 'wrong_baseline', text_token_limits={'example_01': 1_000_000})
        self.assertFalse(reading_result('not json', {'user': ['x'], 'item': ['y']}, .98)['passed'])

    def test_frozen_bundle_is_portable_and_checks_source_and_font_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = self.session(root / 'search')
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            session.render_and_measure(cid)
            provide_readings(session, cid)
            self.assertTrue(session.check_readability(cid)['qualified'])
            frozen = session.freeze({'model': 'offline'})
            portable = root / 'portable'
            shutil.copytree(frozen, portable)
            strategy = FrozenLayout(portable)
            task = dict(sample_id='new', profile='New user context', item={'title': 'A new album'}, path=root / 'new.png')
            self.assertEqual(len(strategy.render_tasks([task])), 1)
            self.assertTrue((root / 'new.png').is_file())
            (portable / 'layout.py').write_text(INITIAL)
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                FrozenLayout(portable)
            shutil.copyfile(frozen / 'layout.py', portable / 'layout.py')
            (portable / 'fonts/body.ttf').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                FrozenLayout(portable)

    def test_visual_tool_loop_exports_actual_program_without_comparison_or_inspection_tool(self):
        with tempfile.TemporaryDirectory() as folder:
            supplied_initial = INITIAL.replace("'section_gap': 16", "'section_gap': 12")
            session = self.session(Path(folder) / 'search', initial_source=supplied_initial)
            self.assertEqual(session.inspect_inputs()['initial_source'], supplied_initial)
            cid = fingerprint(PACKED)[:16]
            class Client:
                turn = 0
                saw_initial_images = False
                saw_candidate_images = False
                def complete(client, messages):
                    image_messages = [m for m in messages if isinstance(m.get('content'), list)
                                      and any(p.get('type') == 'image_url' for p in m['content'])]
                    if client.turn == 0:
                        client.saw_initial_images = bool(image_messages)
                        self.assertIn(supplied_initial, messages[1]['content'])
                    if client.turn == 3:
                        client.saw_candidate_images = bool(image_messages)
                        provide_readings(session, cid)
                    actions = [('inspect_inputs', {}), ('revise_layout', dict(source=PACKED, reason='pack fields')),
                               ('render_and_measure', dict(candidate_id=cid)), ('check_readability', dict(candidate_id=cid))]
                    if client.turn == len(actions):
                        return dict(role='assistant', content='Done'), {}
                    name, args = actions[client.turn]
                    client.turn += 1
                    return dict(role='assistant', content=None, tool_calls=[dict(id=str(client.turn), type='function',
                        function=dict(name=name, arguments=json.dumps(args)))]), dict(model='offline', usage={})
            client = Client()
            path = run_agent(session, client, model='offline', max_turns=8)
            self.assertTrue(client.saw_initial_images and client.saw_candidate_images)
            self.assertEqual((path / 'layout.py').read_text(), PACKED)
            self.assertEqual({t['function']['name'] for t in TOOLS},
                {'inspect_inputs', 'revise_layout', 'render_and_measure', 'check_readability'})
            self.assertNotIn('data:image/png;base64,', (session.output / 'agent_trace.json').read_text())

    def test_chat_transport_uses_real_image_and_tool_message_protocol(self):
        import httpx
        real_client = httpx.Client
        def handler(request):
            self.assertEqual(str(request.url), 'https://example.test/v1/chat/completions')
            self.assertEqual(request.headers['Authorization'], 'Bearer offline-test')
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'configured-vision-model')
            self.assertEqual(body['messages'][0]['content'][0]['image_url']['detail'], 'high')
            self.assertFalse(body['parallel_tool_calls'])
            self.assertEqual(len(body['tools']), 4)
            return httpx.Response(200, json=dict(model='resolved-model', usage={'total_tokens': 12},
                choices=[dict(finish_reason='tool_calls', message=dict(role='assistant', content=None,
                    tool_calls=[dict(id='call1', type='function', function=dict(name='inspect_inputs', arguments='{}'))]))]))
        def factory(**kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)
        with mock.patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test',
                                           'OPENAI_BASE_URL': 'https://example.test/v1'}, clear=True), \
             mock.patch('httpx.Client', side_effect=factory):
            client = ChatClient(dict(model='configured-vision-model'))
            try:
                message, usage = client.complete([dict(role='user', content=[dict(type='image_url',
                    image_url=dict(url='data:image/png;base64,AAAA', detail='high'))])])
            finally:
                client.close()
        self.assertEqual(message['tool_calls'][0]['function']['name'], 'inspect_inputs')
        self.assertEqual(usage['model'], 'resolved-model')

    def test_module_two_renders_all_splits_with_frozen_strategy_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = self.session(root / 'search')
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            session.render_and_measure(cid)
            provide_readings(session, cid)
            session.check_readability(cid)
            frozen = session.freeze({})
            data = root / 'data'
            data.mkdir()
            manifest = dict(output_sha256={}, splits={})
            for split in ['train', 'valid', 'test']:
                rows = [dict(sample_id=f'{split}-{i}', group_id=split, candidate_item_id=str(i),
                             user_context='Interest: music', candidate_payload={'title': f'New item {i}'},
                             label='A' if i == 0 else 'B', _replay=i < 2) for i in range(10)]
                path = data / f'{split}_candidates.jsonl'
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                manifest['output_sha256'][path.name] = digest(path)
                manifest['splits'][split] = dict(natural_hit_groups=1, all_groups=1)
            (data / 'manifest.json').write_text(json.dumps(manifest))
            args = [sys.executable, '-m', 'optical_compression.pipeline', '--datasets', str(data),
                    '--output', str(root / 'dense'), '--layout', str(frozen)]
            for _ in range(2):
                subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
            complete = json.loads((root / 'dense/complete.json').read_text())
            self.assertEqual(complete['layout_signature'], digest(frozen / 'layout.json'))
            self.assertFalse(complete['processor_validated'])
            for split in ['train', 'valid', 'test']:
                report = json.loads((root / f'dense/{split}/manifest.json').read_text())
                self.assertEqual(report['layout']['kind'], 'agent_top_bottom')
                self.assertEqual(len(report['pages']), 10)


if __name__ == '__main__':
    unittest.main()
