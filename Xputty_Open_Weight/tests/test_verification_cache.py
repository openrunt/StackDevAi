import json
from vox.verification_cache import input_signature, can_reuse
from vox.public_baseline import sha256


def test_verification_reuse_requires_unchanged_weights_tokenizer_config_and_report(tmp_path):
    (tmp_path / 'export').mkdir(); (tmp_path / 'data').mkdir()
    for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.model'):
        (tmp_path / 'export' / name).write_bytes(b'original')
    (tmp_path / 'data/vox.model').write_bytes(b'tokenizer')
    (tmp_path / 'export/export_report.json').write_text(json.dumps({'source_model_state_sha256': 'known',
        'source_configuration': {'heads': 2}, 'source_training_policy': {'windows': [128]}, 'source_representation': 'raw'}))
    report = tmp_path / 'hf-verification.json'
    report.write_text(json.dumps({'logits_match': True, 'all_tokenizer_probes_match': True}))
    (tmp_path / 'native-request.json').write_text('{}')
    signature = input_signature(tmp_path)
    (tmp_path / 'hf-verification-cache.json').write_text(json.dumps({'inputs': signature, 'report_sha256': sha256(report),
        'native_request_sha256': sha256(tmp_path / 'native-request.json')}))
    assert can_reuse(tmp_path, signature)
    (tmp_path / 'export/model.safetensors').write_bytes(b'changed')
    assert not can_reuse(tmp_path, input_signature(tmp_path))
    (tmp_path / 'export/model.safetensors').write_bytes(b'original')
    (tmp_path / 'data/vox.model').write_bytes(b'changed tokenizer')
    assert not can_reuse(tmp_path, input_signature(tmp_path))
    report.write_text(json.dumps({'logits_match': False, 'all_tokenizer_probes_match': True}))
    assert not can_reuse(tmp_path, signature)
