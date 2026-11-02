"""Load real sharded Qwen2 weights and check samples from every selected GGUF tensor."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vox.public_baseline import sha256, write_json


def verify(model, gguf_file):
    from vox.limits import limit_cpu, memory_guard
    torch = limit_cpu(2)
    import gguf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vox.gguf_lora import unpack
    from tools.gguf_to_hf import hf_name
    from export_chat import weights_identity
    model, gguf_file = Path(model), Path(gguf_file)
    report = json.loads((model / 'export_report.json').read_text(encoding='utf-8'))
    if sha256(gguf_file) != report['source_native_weights']['sha256'] or weights_identity(model) != report['weights_sha256']:
        raise ValueError('Actual selected model identity changed')
    loaded, info = AutoModelForCausalLM.from_pretrained(model, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype='auto', output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Transformers did not load every expected tensor: ' + str(info))
    loaded.eval()
    reader = gguf.GGUFReader(str(gguf_file))
    checked = []
    with torch.no_grad():
        for tensor in reader.tensors:
            name = hf_name(tensor.name)
            actual = loaded.get_parameter(name)
            if len(tensor.shape) == 2:
                for row in (0, int(tensor.shape[1]) - 1):
                    expected = unpack(tensor, row, row + 1).half()
                    if not torch.equal(expected, actual[row:row + 1]):
                        raise ValueError('Converted tensor row mismatch: ' + name)
            elif not torch.equal(unpack(tensor).half(), actual):
                raise ValueError('Converted tensor mismatch: ' + name)
            checked.append(name)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    outputs = []
    for prompt in ('Which planet is the largest in our solar system? Answer briefly.',
                   'In a story, slow zombies never sleep but humans do. Why might that be dangerous?'):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True, return_tensors='pt')
        began = time.perf_counter()
        with torch.inference_mode():
            generated = loaded.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=48,
                                        pad_token_id=tokenizer.eos_token_id)
        row = {'prompt': prompt, 'text': tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True),
               'wall_seconds': time.perf_counter() - began}
        outputs.append(row); print(json.dumps(row), flush=True)
    result = {'schema': 'xow-sharded-hf-runtime/1', 'weights_sha256': report['weights_sha256'],
              'source_gguf_sha256': report['source_native_weights']['sha256'], 'loaded_without_missing_or_unexpected_keys': True,
              'sampled_tensors_checked': len(checked), 'all_sampled_f16_values_match_source': True,
              'parameter_count': sum(p.numel() for p in loaded.parameters()), 'dtype': str(next(loaded.parameters()).dtype),
              'observed_rss_bytes': memory_guard(4), 'outputs': outputs,
              'limitations': ['Sampled first/last rows in each matrix; full equality is additionally covered on the small round-trip fixture',
                             'F16 storage rounding and native quantized activation arithmetic are not bit-identical',
                             'Successful loading and two replies do not establish conversational quality']}
    write_json(model / 'hf-runtime-verification.json', result)
    report['verified_hf_transformers'] = True
    write_json(model / 'export_report.json', report)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True); p.add_argument('--gguf-file', type=Path, required=True)
    print(json.dumps(verify(**vars(p.parse_args())), indent=2))
