"""Freeze a small, attributable SODA social-dialogue curriculum, with original character examples."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vox.public_baseline import sha256, write_json, conversation_splits, download

REVISION = 'fdc848ab0183208ea7808206c91c724414d0a071'
HASHES = {'train': 'b21314575fdc4cbf9fc74f7af4854d55fba8cc9ba72f2f3791ae2df8b23032c4',
          'val': '3e813e5b14c4edca5849cb7922f9c38568ecb71f27931a7c4c29ec1160bd0d1c',
          'test': 'edd0a57b460fa7faa3782b332327f3a4bc9681dcaaec88e77727c47fd5870180'}


def normalize(row):
    utterances, speakers = row.get('dialogue'), row.get('speakers')
    if not utterances or len(utterances) != len(speakers) or len(set(speakers)) != 2:
        return None
    joined = []
    for speaker, text in zip(speakers, utterances):
        if not isinstance(text, str) or not 2 <= len(text.split()) <= 80 or any(t in text.lower() for t in ('<|', 'as an ai', 'language model')):
            return None
        if joined and speaker == joined[-1][0]: joined[-1][1] += '\n' + text
        else: joined.append([speaker, text])
    if len(joined) % 2: joined.pop()
    if not 6 <= len(joined) <= 16: return None
    first, second = joined[0][0], joined[1][0]
    if len(second) > 60 or any(x in second for x in '\n<>'): return None
    messages = [{'role': 'system', 'content': 'Portray ' + second + ', an original fictional conversation partner. Reply naturally to the other speaker and stay with the current situation.'}]
    messages += [{'role': 'user' if name == first else 'assistant', 'content': text} for name, text in joined]
    return messages


def _prepare_into(cache, original, output):
    import pyarrow.parquet as pq
    cache, output = map(Path, (cache, output))
    original = Path(original) if original is not None else None
    output.mkdir(parents=True, exist_ok=True); (output / 'data').mkdir()
    partitions, raw, seen = {}, [], set()
    originals = conversation_splits(original) if original is not None else {k: [] for k in HASHES}
    for split, count in [('train', 64), ('val', 12), ('test', 12)]:
        source = cache / ('valid.parquet' if split == 'val' else split + '.parquet')
        if sha256(source) != HASHES[split]: raise ValueError('SODA source hash mismatch')
        picked = []
        for batch in pq.ParquetFile(source).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                messages = normalize(row)
                if messages is None: continue
                key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
                if key in seen: continue
                seen.add(key)
                picked.append({'id': key, 'split': split, 'messages': messages, 'source': 'allenai/soda', 'original_index': row['original_index']})
                raw.append({'id': key, 'split': split, 'original': row})
                if len(picked) == count: break
            if len(picked) == count: break
        if len(picked) != count: raise ValueError('Not enough usable source conversations')
        additions = [{**r, 'split': split} for r in originals[split]]
        # Interleave complete conversations; target extraction happens afterwards.
        mixed = []
        for i in range(max(len(picked), len(additions))):
            if i < len(picked): mixed.append(picked[i])
            if i < len(additions): mixed.append(additions[i])
        partitions[split] = mixed
        with (output / 'data' / (split + '.jsonl')).open('w', encoding='utf-8') as f:
            for r in mixed: f.write(json.dumps({'episode_id': r['id'], 'series_id': r.get('source', r.get('topic')), 'text': json.dumps(r['messages'])}) + '\n')
    records = [r for rows in partitions.values() for r in rows]
    (output / 'conversations.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8')
    (output / 'original-soda-records.jsonl').write_text('\n'.join(json.dumps(r) for r in raw) + '\n', encoding='utf-8')
    previous = json.loads((original / 'public-sources.json').read_text(encoding='utf-8')) if original is not None else {}
    from vox.podcast_distill import PERSONAS
    manifest = dict(name='Xputty social-dialogue curriculum', dataset='allenai/soda', revision=REVISION,
        source_url='https://huggingface.co/datasets/allenai/soda', license='CC-BY-4.0', source_hashes=HASHES, dataset_card_sha256=sha256(cache / 'README.md'),
        attribution='SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization; Hyunwoo Kim et al.; Allen Institute for AI',
        previous_character_curriculum_sha256=sha256(original / 'conversations.jsonl') if original is not None else None,
        characters=previous.get('characters', PERSONAS), character_profiles_runtime_only=original is None,
        counts={k: len(v) for k, v in partitions.items()}, corpus_sha256=sha256(output / 'conversations.jsonl'),
        method='All complete assistant exchanges; rolling whole-pair context; source conversations remain partitioned',
        limitations=['SODA is synthetic fictional dialogue, not recordings of people or proof of factual knowledge',
                      'Exact conversation deduplication only; related narratives may remain',
                      'Source pretraining contamination is not ruled out; this is a small adaptation pilot'])
    write_json(output / 'public-sources.json', manifest)
    shutil.copy2(cache / 'README.md', output / 'SODA-DATASET-CARD.md')
    if original is not None: shutil.copy2(original / 'public-sources.json', output / 'CHARACTER-SOURCE-MANIFEST.json')
    return manifest


def prepare(cache, output, original=None):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Create a new social-dialogue snapshot')
    cache = Path(cache)
    for split, expected in HASHES.items():
        name = 'valid.parquet' if split == 'val' else split + '.parquet'
        download('https://huggingface.co/datasets/allenai/soda/resolve/' + REVISION + '/' + name, cache / name, expected)
    download('https://huggingface.co/datasets/allenai/soda/resolve/' + REVISION + '/README.md', cache / 'README.md',
             'e5a7c0a627d64ddcb65c6a40a12065349564094f7362591752b781101fcd420a')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + '-preparing-', dir=output.parent) as temporary:
        work = Path(temporary).resolve()
        if work.parent != output.parent: raise ValueError('Unexpected preparation directory')
        result = _prepare_into(cache, original, work)
        if output.exists(): raise FileExistsError('Output appeared while preparing; preserve it')
        work.rename(output)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'output'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--original', type=Path, help='Optional existing reviewed character curriculum; omit for a SODA-only pilot')
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
