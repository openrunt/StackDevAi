"""Reuse successful HF parity only for unchanged actual inputs and implementations."""
import importlib.metadata
import json
import platform
from pathlib import Path

from .public_baseline import sha256


def input_signature(root):
    root = Path(root)
    code = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / 'export/export_report.json').read_text(encoding='utf-8'))
    identity = {key: metadata[key] for key in ('source_model_state_sha256', 'source_configuration', 'source_training_policy', 'source_representation')}
    identity['files'] = {name: sha256(root / name) for name in (
        'export/model.safetensors', 'export/config.json', 'export/tokenizer.json', 'export/tokenizer_config.json',
        'export/special_tokens_map.json', 'export/tokenizer.model', 'data/vox.model')}
    identity['code'] = {name: sha256(code / name) for name in ('vox/model.py', 'vox/topology.py', 'vox/export.py', 'vox/verification_cache.py', 'tools/verify_hf_export.py')}
    identity['libraries'] = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'sentencepiece')}
    identity['platform'] = {'os': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor()}
    return identity


def can_reuse(root, signature):
    root = Path(root)
    try:
        cached = json.loads((root / 'hf-verification-cache.json').read_text(encoding='utf-8'))
        report = json.loads((root / 'hf-verification.json').read_text(encoding='utf-8'))
        return (cached['inputs'] == signature and cached['report_sha256'] == sha256(root / 'hf-verification.json')
                and cached['native_request_sha256'] == sha256(root / 'native-request.json')
                and report.get('logits_match') is True and report.get('all_tokenizer_probes_match') is True)
    except (OSError, ValueError, KeyError):
        return False
