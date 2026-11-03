"""Package reversible context views, with untouched targets and inherited holdouts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vox.phase_views import PHASES, identity, make_view


def prepare(source, output, tokenizer=None):
    if output.exists(): raise ValueError('Choose a new output folder')
    records = [json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not records: raise ValueError('No records to transform')
    groups, identities = {}, {}
    for record in records:
        group, split = record.get('episode_id') or record.get('topic') or record.get('id'), record.get('split')
        if not isinstance(group, str) or not group or split not in ('train', 'val', 'test'):
            raise ValueError('Each record needs an explicit source group and split')
        for index, key in ((groups, group), (identities, identity(record['messages']))):
            if key in index and index[key] != split: raise ValueError('Canonical source crosses holdout splits')
            index[key] = split
    encoder = None
    if tokenizer:
        from transformers import AutoTokenizer
        encoder = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True, trust_remote_code=False)
    # Validate everything before creating output, including lossless recovery.
    views = [make_view(record, phase) for record in records for phase in PHASES]
    ratios = {phase: {'context_tokens': 0, 'records': 0} for phase in PHASES}
    if encoder:
        for view in views:
            row = ratios[view['phase']]
            row['context_tokens'] += len(encoder.apply_chat_template(view['payload']['messages'][:-1], tokenize=True, add_generation_prompt=True))
            row['records'] += 1
    manifest = {'schema': 'xow-phase-view-package/1', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'records': len(records), 'views': len(views), 'source_groups': len(groups),
                'split_counts': {s: sum(r['split'] == s for r in records) for s in ('train', 'val', 'test')},
                'tokenizer': str(tokenizer) if tokenizer else None, 'token_counts': ratios if encoder else None,
                'limitations': ['Research view package, not a trained or distilled model',
                               'Recovery is exact software reversal, not learned semantic retention',
                               'Targets and system prompts remain canonical; raw reordered context must not be treated as causal dialogue',
                               'Equal characters do not imply equal tokens; duplicated views require matched total training budgets',
                               'No teacher KL, contrastive training or near-miss semantic improvement is claimed']}
    output.mkdir(parents=True)
    body = ''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in views)
    (output / 'views.jsonl').write_text(body, encoding='utf-8')
    manifest['views_sha256'] = hashlib.sha256(body.encode('utf-8')).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, help='Optional local tokenizer folder for inflation measurements')
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
