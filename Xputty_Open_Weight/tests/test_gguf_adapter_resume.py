import json

import gguf
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from tools import train_gguf_adapter
from vox.gguf_lora import tensor_name


def test_quantized_adapter_resumes_exactly_and_rejects_tokenizer_changes(tmp_path, monkeypatch):
    torch.manual_seed(739)
    base_config = tmp_path / 'config'
    vocab = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'user': 3, 'assistant': 4}
    vocab.update({f'q{i}': len(vocab) + i for i in range(24)})
    vocab.update({f'a{i}': len(vocab) + i for i in range(24)})
    raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tokenizer.chat_template = "{% for m in messages %}{{m['role'] + ' ' + m['content'] + ' [EOS] '}}{% endfor %}{% if add_generation_prompt %}{{'assistant '}}{% endif %}"
    tokenizer.save_pretrained(base_config)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                         attention_dropout=0.0, eos_token_id=2, pad_token_id=1)
    config.save_pretrained(base_config)
    model = Qwen2ForCausalLM(config)
    base = tmp_path / 'base.gguf'
    writer = gguf.GGUFWriter(str(base), 'qwen2')
    writer.add_embedding_length(32)
    for name, parameter in model.named_parameters():
        path, suffix = name.rsplit('.', 1)
        if path == 'model.embed_tokens': mapped = 'token_embd.weight'
        elif path == 'model.norm': mapped = 'output_norm.weight'
        elif path.endswith('input_layernorm'): mapped = 'blk.' + path.split('.')[2] + '.attn_norm.weight'
        elif path.endswith('post_attention_layernorm'): mapped = 'blk.' + path.split('.')[2] + '.ffn_norm.weight'
        else: mapped = tensor_name(path, suffix)
        writer.add_tensor(mapped, parameter.detach().numpy())
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    original_base = base.read_bytes()
    groups = {}
    for split, offset in [('train', 0), ('val', 8), ('test', 16)]:
        groups[split] = [{'id': str(i), 'messages': [
            {'role': 'user', 'content': f'q{i}'}, {'role': 'assistant', 'content': f'a{i}'},
            {'role': 'user', 'content': f'q{i} q{i}'}, {'role': 'assistant', 'content': f'a{i} a{i}'}
        ]} for i in range(offset, offset + 8)]
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'conversations.jsonl').write_text('{"fixture": 1}\n')
    (dataset / 'public-sources.json').write_text(json.dumps({'runtime_persona_rules': 'Fixture runtime rules.'}))
    monkeypatch.setattr(train_gguf_adapter, 'conversation_splits', lambda *args: groups)
    kwargs = dict(dataset=dataset, base=base, config=base_config, context=256, layers=1,
                  max_train_examples=8, max_eval_examples=8, target_mode='all-turns')
    train_gguf_adapter.train(output=tmp_path / 'resumed', steps=1, **kwargs)
    # Explicit resume must clear a prior safe-stop request.
    (tmp_path / 'resumed/stop-request').write_text('stop')
    resumed = train_gguf_adapter.train(output=tmp_path / 'resumed', steps=2, resume=True, **kwargs)
    full = train_gguf_adapter.train(output=tmp_path / 'full', steps=2, **kwargs)
    a = torch.load(tmp_path / 'resumed/adapter-checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path / 'full/adapter-checkpoint.pt', weights_only=True)
    assert a['step'] == b['step'] == 2
    assert all(torch.equal(a['weights'][key], b['weights'][key]) for key in a['weights'])
    assert resumed['selected_update'] == full['selected_update']
    assert resumed['frozen_prefix_cache']['reused'] is True
    assert resumed['persona_rules'] == 'Fixture runtime rules.'
    assert base.read_bytes() == original_base
    path = base_config / 'tokenizer_config.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='another base, corpus or configuration'):
        train_gguf_adapter.train(output=tmp_path / 'resumed', steps=3, resume=True, **kwargs)
