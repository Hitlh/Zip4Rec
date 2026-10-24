"""Measure the original experiment's compact DensePage; readback scores are diagnostic."""
import argparse
import json
from pathlib import Path
import statistics

from workflow import digest
from .document import document, fingerprint, select_examples
from .measurement import TokenMeter, LocalReader, reading_result
from .renderer import compact_sections, render_compact


def audit_compact(examples, sources, meter, reader, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(renderer='original_cd_compact640_font14', complete=False,
                  readback_mode='diagnostic_only', processor=meter.identity, records=[])
    path = output / 'report.json'

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    save()
    for example in examples:
        source = sources[example['document_sha256']]
        image = output / 'images' / (example['id'] + '.png')
        task = dict(sample_id=example['id'], profile=source['user_context'],
                    item=source['candidate_payload'], path=str(image))
        page = render_compact(task)
        sections = compact_sections(task)
        record = dict(id=example['id'], document_sha256=example['document_sha256'],
            image=str(image.resolve()), image_sha256=digest(image), **page,
            **meter.image(image), pure_text_tokens=meter.text(example['document'])['total_input_tokens'],
            displayed_sections=sections)
        record['strictly_below_text'] = record['total_input_tokens'] < record['pure_text_tokens']
        if reader is not None:
            record['readback'] = reading_result(reader.read(image), sections)
        report['records'].append(record)
        save()
        print('Audited current compact DensePage: ' + example['id'], flush=True)
    report['samples'] = len(report['records'])
    report['tokens'] = {name: statistics.fmean(r[name] for r in report['records']) for name in
                        ['total_input_tokens', 'image_patch_tokens', 'pure_text_tokens']}
    report['strictly_below_text_samples'] = sum(r['strictly_below_text'] for r in report['records'])
    if reader is not None:
        parsed = [r['readback'] for r in report['records'] if r['readback']['parsed']]
        report['readback_summary'] = dict(parsed_samples=len(parsed), samples=len(report['records']),
            section_similarity={section: dict(
                mean=statistics.fmean(r['section_similarity'][section] for r in parsed),
                minimum=min(r['section_similarity'][section] for r in parsed),
                maximum=max(r['section_similarity'][section] for r in parsed)) if parsed else None
                for section in ['user', 'item']},
            mean_section_similarity=statistics.fmean(r['mean_section_similarity'] for r in parsed) if parsed else None)
        report['reader'] = reader.identity()
    report['renderer_sha256'] = digest(Path(__file__).parent / 'renderer.py')
    report['complete'] = True
    save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=Path, required=True, help='Real training semantic samples; select the same ten as optimize')
    p.add_argument('--model', type=Path, required=True, help='Local Qwen processor; weights needed only with --readback')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--readback', action='store_true', help='Also measure local-VLM transcription similarities, without thresholds')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-new-tokens', type=int, default=4096)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    examples = select_examples(a.samples)
    wanted = {e['document_sha256'] for e in examples}
    sources = {}
    with a.samples.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = fingerprint(document(row['user_context'], row['candidate_payload']))
            if key in wanted:
                sources.setdefault(key, row)
    meter = TokenMeter(a.model)
    reader = LocalReader(meter, device=a.device, max_new_tokens=a.max_new_tokens) if a.readback else None
    report = audit_compact(examples, sources, meter, reader, a.output)
    report['input_sha256'] = digest(a.samples)
    (a.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Current experiment layout report: ' + str(a.output / 'report.json'), flush=True)


if __name__ == '__main__':
    main()
