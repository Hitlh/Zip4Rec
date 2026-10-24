"""Four agent tools with independent checks and deterministic best-strategy selection."""
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import platform

from workflow import digest
from ..document import fingerprint
from ..layout_engine import execute_layout, render_plan, validate_source, engine_hashes, supported_glyph
from ..measurement import reading_result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def runtime_versions():
    result = dict(python=platform.python_version())
    for package in ['Pillow', 'torch', 'transformers', 'httpx']:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


class LayoutSession:
    def __init__(self, examples, constraints, meter, reader, output, *, max_length=2048,
                 readability_threshold=.98, max_candidates=20, text_token_limits=None, initial_source=None):
        if not (0 < readability_threshold <= 1 and type(max_candidates) is int and max_candidates >= 1
                and type(max_length) is int and max_length >= 1):
            raise ValueError('invalid evaluation limits')
        self.examples, self.constraints = examples, constraints
        self.initial_source = initial_source if initial_source is not None else (
            Path(__file__).parents[1] / 'layouts/initial.py').read_text()
        validate_source(self.initial_source)
        for example in examples:
            chars = set(''.join(f['text'] for section in example['document'].values() for f in section))
            if any(not supported_glyph(constraints.font_path, constraints.font_size, char) for char in chars):
                raise ValueError('initialization font lacks dataset glyphs; choose a font covering the input language')
        self.meter, self.reader = meter, reader
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.max_length, self.threshold = max_length, readability_threshold
        self.max_candidates, self.candidates, self.best = max_candidates, {}, None
        self.baselines = {e['id']: meter.text(e['document'])['total_input_tokens'] for e in examples}
        if text_token_limits is not None and text_token_limits != self.baselines:
            raise ValueError('provided pure-text counts differ from the actual same-protocol processor counts')
        self.initialization = dict(examples=examples, constraints=asdict(constraints),
            initial_source_sha256=fingerprint(self.initial_source),
            font_sha256=digest(constraints.font_path), processor=meter.identity,
            text_token_limits=self.baselines, max_length=max_length,
            readability_threshold=readability_threshold, max_candidates=max_candidates,
            input_scope='training only; labels, targets and source IDs removed')
        write_json(self.output / 'initialization.json', self.initialization)

    def inspect_inputs(self):
        return dict(**self.initialization, candidate_ids=list(self.candidates), best=self.best,
            initial_source=self.initial_source)

    def revise_layout(self, source, reason):
        validate_source(source)
        cid = fingerprint(source)[:16]
        if cid in self.candidates:
            return dict(candidate_id=cid, reused=True)
        if len(self.candidates) >= self.max_candidates:
            raise ValueError('candidate budget exhausted')
        directory = self.output / 'candidates' / cid
        directory.mkdir(parents=True)
        (directory / 'layout.py').write_text(source)
        self.candidates[cid] = dict(source=source, reason=reason, directory=directory, report=None)
        return dict(candidate_id=cid, source_sha256=digest(directory / 'layout.py'))

    def candidate(self, candidate_id):
        if candidate_id not in self.candidates:
            raise ValueError('unknown candidate ID')
        return self.candidates[candidate_id]

    def render_and_measure(self, candidate_id):
        candidate = self.candidate(candidate_id)
        if candidate['report'] is not None:
            return candidate['report']
        docs = [e['document'] for e in self.examples]
        report = dict(candidate_id=candidate_id, qualified=False, records=[])
        try:
            plans = execute_layout(candidate['source'], docs, self.constraints)
            # A second evaluation rejects stateful rules that change the layout for identical inputs.
            if plans != execute_layout(candidate['source'], docs, self.constraints):
                raise ValueError('layout must be deterministic')
            for example, plan in zip(self.examples, plans):
                path = candidate['directory'] / (example['id'] + '.png')
                page = render_plan(example['document'], plan, self.constraints, path)
                counts = self.meter.image(path)
                if counts['image_patch_tokens'] <= 0 or counts['total_input_tokens'] <= 0:
                    raise ValueError('processor did not produce valid visual tokens')
                total, baseline = counts['total_input_tokens'], self.baselines[example['id']]
                report['records'].append(dict(id=example['id'], image=str(path), **page, **counts,
                    pure_text_tokens=baseline, strictly_below_text=total < baseline,
                    within_length_guard=total <= self.max_length))
            report['geometry_valid'] = True
            report['token_qualified'] = all(r['strictly_below_text'] and r['within_length_guard']
                                            for r in report['records'])
            report['mean_image_tokens'] = statistics.fmean(r['image_patch_tokens'] for r in report['records'])
            report['max_image_tokens'] = max(r['image_patch_tokens'] for r in report['records'])
            report['mean_total_tokens'] = statistics.fmean(r['total_input_tokens'] for r in report['records'])
        except (ValueError, KeyError, TypeError, TimeoutError, RuntimeError, subprocess.SubprocessError) as exc:
            report.update(geometry_valid=False, token_qualified=False, error=str(exc))
        candidate['report'] = report
        write_json(candidate['directory'] / 'evaluation.json', report)
        return report

    def check_readability(self, candidate_id):
        candidate = self.candidate(candidate_id)
        report = candidate['report']
        if report is None or not report.get('geometry_valid'):
            raise ValueError('render a geometrically valid candidate first')
        if 'readability' not in report:
            readings = []
            for row in report['records']:
                raw = self.reader.read(Path(row['image']))
                readings.append(dict(id=row['id'], **reading_result(raw, row['sections'], self.threshold)))
            report['readability'] = readings
            report['readability_passed'] = all(r['passed'] for r in readings)
            report['qualified'] = report['readability_passed'] and report['token_qualified']
            write_json(candidate['directory'] / 'evaluation.json', report)
        if report['qualified']:
            score = (report['mean_image_tokens'], report['max_image_tokens'], candidate_id)
            if self.best is None or score < tuple(self.best['score']):
                self.best = dict(candidate_id=candidate_id, score=list(score))
                write_json(self.output / 'best.json', self.best)
        return dict(candidate_id=candidate_id, qualified=report['qualified'],
                    readability=report['readability'], best=self.best)

    def dispatch(self, name, arguments):
        if name not in {'inspect_inputs', 'revise_layout', 'render_and_measure', 'check_readability'}:
            raise ValueError('unknown tool')
        return getattr(self, name)(**arguments)

    def freeze(self, agent_identity):
        write_json(self.output / 'best.json', self.best)
        summary = dict(status='qualified' if self.best else 'no_qualified_strategy', best=self.best,
            candidates=[dict(candidate_id=key, reason=value['reason'], report=value['report'])
                        for key, value in self.candidates.items()])
        write_json(self.output / 'search_summary.json', summary)
        if self.best is None:
            return None
        candidate = self.candidate(self.best['candidate_id'])
        target = self.output / 'frozen.pending'
        target.mkdir()
        (target / 'layout.py').write_text(candidate['source'])
        (target / 'fonts').mkdir()
        shutil.copyfile(self.constraints.font_path, target / 'fonts/body.ttf')
        license_path = Path(self.constraints.font_path).parent / 'LICENSE.txt'
        if license_path.is_file():
            shutil.copyfile(license_path, target / 'fonts/LICENSE.txt')
        cfg = asdict(self.constraints)
        cfg['font_path'] = 'fonts/body.ttf'
        files = {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob('*')) if p.is_file()}
        write_json(target / 'layout.json', dict(format='zip4rec-top-bottom-v1', qualified=True,
            constraints=cfg, files_sha256=files, engine_sha256=engine_hashes(),
            evaluation_limits=dict(max_length=self.max_length, readability_threshold=self.threshold,
                                   compression_rule='total_input_tokens < same-protocol pure_text_tokens, per example'),
            processor=self.meter.identity, reader=self.reader.identity(), agent=agent_identity,
            runtime=runtime_versions(), evaluation_code_sha256={
                'measurement.py': digest(Path(__file__).parents[1] / 'measurement.py'),
                'optimizer/tools.py': digest(Path(__file__))},
            initialization_sha256=digest(self.output / 'initialization.json'),
            evaluation_sha256=digest(candidate['directory'] / 'evaluation.json'),
            example_document_sha256=[e['document_sha256'] for e in self.examples],
            metrics={k: v for k, v in candidate['report'].items() if k != 'records' and k != 'readability'}))
        frozen = self.output / 'frozen'
        target.rename(frozen)
        return frozen
