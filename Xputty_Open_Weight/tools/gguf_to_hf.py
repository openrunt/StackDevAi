"""Stream exact selected Qwen2 GGUF tensors into portable F16 Safetensors shards."""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vox.public_baseline import sha256, write_json


def hf_name(name):
    direct = {'token_embd.weight': 'model.embed_tokens.weight', 'output.weight': 'lm_head.weight', 'output_norm.weight': 'model.norm.weight'}
    if name in direct: return direct[name]
    pieces = name.split('.')
    if len(pieces) != 4 or pieces[0] != 'blk' or pieces[3] not in ('weight', 'bias'):
        raise ValueError('Unsupported Qwen2 GGUF tensor: ' + name)
    mapping = {'attn_norm': 'input_layernorm', 'ffn_norm': 'post_attention_layernorm',
        'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v': 'self_attn.v_proj',
        'attn_output': 'self_attn.o_proj', 'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj', 'ffn_down': 'mlp.down_proj'}
    return 'model.layers.' + pieces[1] + '.' + mapping[pieces[2]] + '.' + pieces[3]


def convert(source, output, shard_bytes=512 * 1024**2):
    from vox.limits import limit_cpu, memory_guard
    torch = limit_cpu(2)
    import gguf
    from safetensors.torch import save_file
    from vox.gguf_lora import unpack
    from export_chat import weights_identity
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a fresh Hugging Face export folder')
    report = json.loads((source / 'export_report.json').read_text(encoding='utf-8'))
    native = report['native_weights']
    gguf_file = (source / native['file']).resolve()
    if gguf_file.parent != source or sha256(gguf_file) != native['sha256']:
        raise ValueError('Selected GGUF identity does not match its report')
    reader = gguf.GGUFReader(str(gguf_file))
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    if config['model_type'] != 'qwen2' or reader.fields['general.architecture'].contents() != 'qwen2':
        raise ValueError('This exporter supports Qwen2 only; no architecture substitution')
    if config['hidden_size'] != int(reader.fields['qwen2.embedding_length'].contents()):
        raise ValueError('Tokenizer/config folder disagrees with selected GGUF dimensions')
    config['tie_word_embeddings'] = False
    config['torch_dtype'] = 'float16'
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + '-exporting-', dir=output.parent) as temporary:
        work = Path(temporary).resolve()
        if work.parent != output.parent: raise ValueError('Unexpected export directory')
        pending, pending_bytes, mapping, pieces, total = {}, 0, {}, [], 0
        def flush():
            nonlocal pending, pending_bytes
            if not pending: return
            name = f'model-part-{len(pieces) + 1:05d}.safetensors'
            save_file(pending, str(work / name), metadata={'format': 'pt'})
            for key in pending: mapping[key] = name
            pieces.append(name); pending, pending_bytes = {}, 0
            print(json.dumps({'phase': 'safetensors shard', 'completed': len(pieces), 'rss_bytes': memory_guard(4)}), flush=True)
        for tensor in reader.tensors:
            name = hf_name(tensor.name)
            shape = tuple(map(int, tensor.shape[::-1]))
            size = 2
            for dimension in shape: size *= dimension
            if pending and pending_bytes + size > shard_bytes: flush()
            value = torch.empty(shape, dtype=torch.float16)
            if len(shape) == 2:
                for start in range(0, shape[0], 4096):
                    end = min(shape[0], start + 4096)
                    value[start:end].copy_(unpack(tensor, start, end))
            else:
                value.copy_(unpack(tensor))
            pending[name] = value
            pending_bytes += size; total += size
        flush()
        rename = {old: f'model-{i + 1:05d}-of-{len(pieces):05d}.safetensors' for i, old in enumerate(pieces)}
        for old, new in rename.items(): (work / old).rename(work / new)
        write_json(work / 'model.safetensors.index.json', {'metadata': {'total_size': total}, 'weight_map': {k: rename[v] for k, v in mapping.items()}})
        write_json(work / 'config.json', config)
        for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'chat_template.jinja', 'generation_config.json',
                     'LICENSE', 'APPLICATION-LICENSE.txt', 'UPSTREAM-MODEL-CARD.md', 'public-sources.json', 'SODA-DATASET-CARD.md', 'CHARACTER-SOURCE-MANIFEST.json'):
            if (source / name).exists(): shutil.copy2(source / name, work / name)
        shutil.copy2(Path(__file__).resolve().parents[1] / 'export_chat.py', work / 'chat.py')
        report['source_native_weights'] = report.pop('native_weights')
        report.update(weights_format='sharded-safetensors-f16-from-selected-gguf', weights_sha256=weights_identity(work),
                      export_note='F16 rounding of the actual selected quantized GGUF tensors; not the upstream unquantized weights',
                      shard_count=len(pieces), tensor_count=len(mapping), weights_bytes=total, verified_hf_transformers=False)
        write_json(work / 'export_report.json', report)
        (work / 'README.md').write_text('# ' + report.get('name', 'Qwen2 adapted model') + '\n\n'
            'Portable Hugging Face Qwen2 weights derived from the exact selected GGUF recorded in export_report.json. '
            'F16 conversion adds rounding; native quantized arithmetic and full-precision execution are not bit-identical. '
            'Use AutoModelForCausalLM.from_pretrained with local_files_only=True and torch_dtype="auto". '
            'The browser chat is python chat.py and uses these actual local Safetensors, with no remote fallback. '
            'Base weights retain Apache-2.0; dataset and application notices accompany this export.\n', encoding='utf-8')
        if output.exists(): raise FileExistsError('Destination appeared during export; preserve it')
        work.rename(output)
    return {'output': str(output), 'shards': len(pieces), 'tensor_count': len(mapping), 'bytes': total, 'weights_sha256': report['weights_sha256']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    print(json.dumps(convert(**vars(p.parse_args())), indent=2))
