"""Matched-data scratch attention pilot. Separate from the browser's dense preset."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vox.limits import limit_cpu, memory_guard, environment
from vox.data import CorpusSamples
from vox.model import ModelConfig, TinyLlama
from vox.topology import MASK_EXPERIMENTS, mask_indices


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024**2), b''): h.update(part)
    return h.hexdigest()


def run(data, output, steps=32, context=512, seeds=(739, 751, 761), eval_samples=8, dim=32, layers=1, heads=2, ffn_dim=None):
    if output.exists(): raise ValueError('Choose a new experiment folder')
    if not 1 <= steps <= 10000 or not 1 <= eval_samples <= 100 or len(seeds) != len(set(seeds)) or not 1 <= len(seeds) <= 10:
        raise ValueError('Invalid pilot budget')
    torch = limit_cpu(2)
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor(model_file=str(data / 'vox.model'))
    base_cfg = dict(vocab_size=tokenizer.get_piece_size(), dim=dim, layers=layers, heads=heads,
                    ffn_dim=ffn_dim or dim * 3, context=context)
    ModelConfig(**base_cfg).validate()
    paths = [p for p in data.iterdir() if p.is_file() and (p.suffix in ('.json', '.jsonl', '.u32', '.u8', '.model', '.vocab'))]
    manifest = {p.name: digest(p) for p in sorted(paths)}
    output.mkdir(parents=True)
    modes = ['dense', *MASK_EXPERIMENTS]
    report = {'schema': 'xow-phi-attention-pilot/1', 'status': 'running', 'config': base_cfg, 'steps': steps, 'seeds': list(seeds),
              'data_files': manifest, 'environment': environment(), 'runs': [],
              'code': {name: digest(Path(__file__).resolve().parents[1] / name) for name in ('vox/model.py', 'vox/topology.py', 'vox/data.py', 'vox/limits.py', 'vox/cpu_affinity.py', 'tools/phi_ablation.py')},
              'limitations': ['Small random-init next-token diagnostic; not distilled and not a coherent podcast host',
                              'Dense kernels for every arm; edge count is not actual sparse FLOPs',
                              'No phase-augmentation, near-miss semantic scoring, speech, or VR acceptance',
                              'Fixed update/token budget; elapsed compute and permitted edges are reported separately',
                              'Validation differences in this short pilot are not a population-level quality result']}
    def save():
        temporary = output / 'report.tmp'
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(output / 'report.json')
    for name, expected in report['code'].items():
        raw = (Path(__file__).resolve().parents[1] / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected: raise ValueError('Source changed during experiment startup')
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(raw)
    save()
    for seed in seeds:
        samples = {}
        for split, count in [('train', steps), ('val', eval_samples), ('test', eval_samples)]:
            source = CorpusSamples(data, split, context, stage='base')
            samples[split] = [source.sample(i, seed if split == 'train' else seed + 17) for i in range(count)]
        sample_hash = {split: hashlib.sha256(b''.join(x.tobytes() + y.tobytes() for x, y, _ in rows)).hexdigest() for split, rows in samples.items()}
        # Rotate arm execution order across seeds to reduce one fixed warm-up ordering effect.
        order = modes[(seeds.index(seed)) % len(modes):] + modes[:(seeds.index(seed)) % len(modes)]
        initial_identity = None
        for mode in order:
            torch.manual_seed(seed)
            model = TinyLlama(ModelConfig(**base_cfg, experiment=mode))
            identity = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
            if initial_identity is None: initial_identity = identity
            if identity != initial_identity: raise AssertionError('Arms must start with identical parameters')
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
            def score(split):
                total, tokens = 0., 0
                model.eval()
                with torch.inference_mode():
                    for x, y, info in samples[split]:
                        _, loss = model(torch.tensor(x[None]), torch.tensor(y[None]))
                        total += float(loss) * info['loss_tokens']; tokens += info['loss_tokens']
                return {'loss': total / tokens, 'target_tokens': tokens}
            before = score('val')
            model.train(); began = time.perf_counter(); targets = 0
            for x, y, info in samples['train']:
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(torch.tensor(x[None]), torch.tensor(y[None]))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); targets += info['loss_tokens']; memory_guard(4)
            elapsed = time.perf_counter() - began
            val, test = score('val'), score('test')
            edges = context * (context + 1) // 2 if mode == 'dense' else sum(map(len, mask_indices(context, angle=MASK_EXPERIMENTS[mode])))
            row = {'seed': seed, 'mode': mode, 'parameters': model.n_parameters(), 'initial_weights_sha256': identity,
                   'sample_hashes': sample_hash, 'before_validation': before, 'validation': val, 'test': test,
                   'input_positions': context * steps, 'target_tokens': targets, 'training_seconds': elapsed,
                   'eligible_edges_per_sequence': edges}
            checkpoint = output / (str(seed) + '-' + mode + '.pt')
            torch.save({'config': asdict(model.config), 'weights': model.state_dict(), 'steps': steps, 'seed': seed}, checkpoint)
            row['checkpoint_sha256'] = digest(checkpoint)
            report['runs'].append(row); save()
            print(json.dumps(row), flush=True)
    report['status'] = 'complete'; save()
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=32); p.add_argument('--context', type=int, default=512)
    p.add_argument('--seeds', type=int, nargs='+', default=[739, 751, 761]); p.add_argument('--eval-samples', type=int, default=8)
    p.add_argument('--dim', type=int, default=32); p.add_argument('--layers', type=int, default=1)
    p.add_argument('--heads', type=int, default=2); p.add_argument('--ffn-dim', type=int)
    run(**vars(p.parse_args()))
