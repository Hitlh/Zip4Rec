"""Encode every rendered page using the actual VLM processor without truncation."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=2048)
    a = p.parse_args()
    import torch
    from transformers import AutoProcessor
    from ranking.conversation import process_conversation, record_messages, token_counts, read_jsonl
    processor = AutoProcessor.from_pretrained(str(a.model), local_files_only=True)
    records = read_jsonl(a.data)
    counts = []
    with torch.inference_mode():
        for row in records:
            encoded = process_conversation(processor, record_messages(row, include_assistant=False),
                                           add_generation_prompt=True)
            value = dict(sample_id=row['sample_id'], group_id=row['group_id'], **token_counts(processor, encoded))
            if value['total_input_tokens'] > a.max_length:
                raise ValueError('input exceeds model guard without truncation: ' + row['sample_id'])
            counts.append(value)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(dict(records=counts, samples=len(counts),
            total_input_tokens=sum(c['total_input_tokens'] for c in counts),
            mean_input_tokens=statistics.fmean(c['total_input_tokens'] for c in counts),
            max_input_tokens=max(c['total_input_tokens'] for c in counts),
            max_length=a.max_length, all_inputs_within_guard=True), stream, indent=2)


if __name__ == '__main__':
    main()
