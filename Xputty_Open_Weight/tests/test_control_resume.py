import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM

from vox import public_control


@pytest.mark.parametrize('scratch_manifest,prefix_cache', [(True, False), (False, False), (False, True)])
def test_complete_social_targets_resume_exactly_and_reject_changed_protocol(tmp_path, monkeypatch, scratch_manifest, prefix_cache):
    torch.manual_seed(223)
    base = tmp_path / 'base'
    vocab = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'user': 3, 'assistant': 4}
    vocab.update({f'q{i}': len(vocab) + i for i in range(24)})
    vocab.update({f'a{i}': len(vocab) + i for i in range(24)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tok.chat_template = "{% for m in messages %}{{m['role'] + ' ' + m['content'] + ' [EOS] '}}{% endfor %}{% if add_generation_prompt %}{{'assistant '}}{% endif %}"
    tok.save_pretrained(base)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
        attention_dropout=0.0 if prefix_cache else 0.05, pad_token_id=1, eos_token_id=2))
    model.save_pretrained(base, safe_serialization=True)
    for name in ('README.md', 'LICENSE'):
        (base / name).write_text('Synthetic test fixture only', encoding='utf-8')
    groups = {}
    for split, offset in [('train', 0), ('val', 8), ('test', 16)]:
        groups[split] = [{'id': str(i), 'messages': [
            {'role': 'user', 'content': f'q{i}'}, {'role': 'assistant', 'content': f'a{i}'},
            {'role': 'user', 'content': f'q{i} q{i}'}, {'role': 'assistant', 'content': f'a{i} a{i}'}
        ]} for i in range(offset, offset + 8)]
    project = tmp_path / 'project'
    (project / 'data').mkdir(parents=True)
    if scratch_manifest:
        (project / 'data/corpus_report.json').write_text('{"fixture": 1}')
    else:
        (project / 'conversations.jsonl').write_text('{"fixture": 1}\n')
    (project / 'public-sources.json').write_text('{}')
    monkeypatch.setattr(public_control, 'fetch_control', lambda *args: base)
    monkeypatch.setattr(public_control, 'conversation_splits', lambda *args: groups)
    monkeypatch.setitem(public_control.REFERENCES, 'test-fixture', {'model': 'fixture', 'revision': 'fixture', 'folder': 'fixture'})
    kwargs = dict(project=project, cache=tmp_path, context=128, reference='test-fixture', blocks=1,
                  eval_samples=8, target_mode='all-turns', train_samples=16, prefix_cache=prefix_cache)
    first = public_control.adapt_control(output=tmp_path / 'resumed', steps=1, **kwargs)
    assert first['training_targets'] == 16 and first['target_protocol'] == 'complete-assistant-rolling-v1'
    # A stale standalone selected file must not corrupt the authoritative atomic checkpoint.
    torch.save({'stale': True}, tmp_path / 'resumed/selected-adapter.pt')
    second = public_control.adapt_control(output=tmp_path / 'resumed', steps=2, resume=True, **kwargs)
    full = public_control.adapt_control(output=tmp_path / 'full', steps=2, **kwargs)
    a = torch.load(tmp_path / 'resumed/adaptation-checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path / 'full/adaptation-checkpoint.pt', weights_only=True)
    assert a['step'] == b['step'] == 2
    assert all(torch.equal(a['weights'][key], b['weights'][key]) for key in a['weights'])
    assert a['processed'] == b['processed'] and a['targets'] == b['targets']
    assert second['resumed_from_update'] == 1
    assert second['selected_update'] == full['selected_update']
    with pytest.raises(ValueError, match='changed'):
        public_control.adapt_control(output=tmp_path / 'resumed', steps=3, resume=True, **{**kwargs, 'target_mode': 'first'})
