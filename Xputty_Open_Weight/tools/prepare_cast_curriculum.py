"""Prepare original four-character conversation targets, grouped by episode topic."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vox.podcast_cast_curriculum import prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    manifest = prepare(parser.parse_args().output)
    print(json.dumps({'records': manifest['record_counts'], 'topics': {k: len(v) for k, v in manifest['topics'].items()},
                      'corpus_sha256': manifest['corpus_sha256']}, indent=2))
