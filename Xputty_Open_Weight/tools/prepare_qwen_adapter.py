"""Fetch the pinned 1.5B quantized base and matching tokenizer; no model execution."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vox.public_baseline import download, sha256, write_json
from tools.podcast_pilot import TEACHER_REPO, TEACHER_REV, TEACHER_FILE, TEACHER_SHA

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def prepare(cache, gguf=None):
    cache = Path(cache)
    base = Path(gguf).resolve() if gguf else cache / 'base.gguf'
    if gguf and sha256(base) != TEACHER_SHA:
        raise ValueError('Existing GGUF does not match the pinned Qwen2.5-1.5B reference')
    if not gguf:
        base = download('https://huggingface.co/' + TEACHER_REPO + '/resolve/' + TEACHER_REV + '/' + TEACHER_FILE, base, TEACHER_SHA)
    config = cache / 'tokenizer'
    hashes = {}
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json', 'README.md', 'LICENSE'):
        file = download('https://huggingface.co/' + MODEL + '/resolve/' + REVISION + '/' + name, config / name)
        hashes[name] = sha256(file)
    report = {'base': str(base.resolve()), 'config': str(config.resolve()), 'base_sha256': TEACHER_SHA,
              'base_repo': TEACHER_REPO, 'base_revision': TEACHER_REV, 'model_repo': MODEL, 'model_revision': REVISION,
              'metadata_sha256': hashes, 'license': 'Apache-2.0', 'note': 'Existing matching GGUF can be reused without copying; metadata only is fetched from the unquantized repository.'}
    write_json(cache / 'source.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--gguf', type=Path, help='Reuse an existing exact pinned base instead of downloading')
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
