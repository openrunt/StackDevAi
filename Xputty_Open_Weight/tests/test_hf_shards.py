import json
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoModelForCausalLM

from tools.gguf_to_hf import convert
from vox.gguf_lora import tensor_name
from vox.public_baseline import sha256
from export_chat import weights_identity


def test_selected_gguf_roundtrips_into_loadable_hf_shards(tmp_path):
    torch.manual_seed(739)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=False)
    model = Qwen2ForCausalLM(config).eval()
    source = tmp_path / 'source'; source.mkdir()
    config.save_pretrained(source)
    writer = gguf.GGUFWriter(str(source / 'selected.gguf'), 'qwen2')
    writer.add_embedding_length(config.hidden_size)
    for name, value in model.state_dict().items():
        if name == 'model.embed_tokens.weight': native = 'token_embd.weight'
        elif name == 'model.norm.weight': native = 'output_norm.weight'
        elif 'input_layernorm' in name: native = 'blk.' + name.split('.')[2] + '.attn_norm.weight'
        elif 'post_attention_layernorm' in name: native = 'blk.' + name.split('.')[2] + '.ffn_norm.weight'
        else:
            path, suffix = name.rsplit('.', 1)
            native = tensor_name(path, suffix)
        writer.add_tensor(native, value.numpy().astype(np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    original = sha256(source / 'selected.gguf')
    (source / 'export_report.json').write_text(json.dumps({'native_weights': {'file': 'selected.gguf', 'sha256': original}}))
    out = tmp_path / 'hf'
    result = convert(source, out, shard_bytes=2048)
    loaded = AutoModelForCausalLM.from_pretrained(out, local_files_only=True, torch_dtype='auto', use_safetensors=True).eval()
    assert result['shards'] > 1
    assert weights_identity(out) == result['weights_sha256']
    assert sha256(source / 'selected.gguf') == original
    assert loaded.config.tie_word_embeddings is False
    assert set(loaded.state_dict()) == set(model.state_dict())
    for name, value in model.state_dict().items():
        assert torch.equal(value.half(), loaded.state_dict()[name]), name
    ids = torch.tensor([[1, 2, 5, 9]])
    with torch.no_grad():
        assert torch.allclose(model.half()(ids).logits, loaded(ids).logits, atol=1e-5)
