from dataclasses import replace
import pytest
import torch
from vox.topology import mask_indices, OFFSETS, MASK_EXPERIMENTS
from vox.model import TinyLlama, ModelConfig, generate
from tools.phi_overlap import measure


def test_atlas_import_checks_numeric_records_and_rejects_unsafe_members(tmp_path):
    import csv, io, json, math, zipfile
    from tools.phi_overlap import verify_atlas
    measured = measure(32)
    fieldnames = ['position','phase_difference_deg','phase_alignment_cos','shared_edges_row',
                  'golden_only_edges_row','hybrid_only_edges_row','edge_jaccard_row','changed_attention_row',
                  'cumulative_shared','cumulative_golden_only','cumulative_hybrid_only','cumulative_jaccard']
    stream = io.StringIO(); writer = csv.writer(stream); writer.writerow(fieldnames)
    cs = cg = ch = 0
    for row in measured['per_position']:
        cs += row['shared']; cg += row['g_only']; ch += row['h_only']; n = row['n']
        writer.writerow([n, n * measured['delta_angle_degrees'], math.cos(math.radians(n * measured['delta_angle_degrees'])),
                         row['shared'], row['g_only'], row['h_only'], row['jaccard'], int(row['changed']),cs,cg,ch,cs/(cs+cg+ch)])
    supplied = {k: v for k, v in measured.items() if k not in ('schema','controls','limitations','source_sha256')}
    def package(name, content, unsafe=False):
        path = tmp_path / name
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('phi_overlap_metrics.json', json.dumps(supplied))
            z.writestr('phi_overlap_by_position.csv', content)
            if unsafe: z.writestr('../escape.txt', 'x')
        return path
    assert verify_atlas(package('good.zip', stream.getvalue()), measured)['all_json_and_csv_rows_match']
    with pytest.raises(ValueError, match='Unsafe'):
        verify_atlas(package('unsafe.zip', stream.getvalue(), True), measured)
    body = list(csv.reader(io.StringIO(stream.getvalue())))
    body[1][2] = 'nan'
    damaged = io.StringIO(); csv.writer(damaged).writerows(body)
    with pytest.raises(ValueError, match='CSV mismatch'):
        verify_atlas(package('nan.zip', damaged.getvalue()), measured)


def test_actual_ablation_shares_samples_and_initial_weights(tmp_path):
    from tools.phi_ablation import run
    from vox.plain_text import build
    from vox.tokenizer import train
    from vox.data import prepare
    paths = []
    for i in range(4):
        p = tmp_path / (str(i) + '.txt')
        p.write_text((f'Chapter {i}. The travellers found a blue cup under the lamp. They discussed where to leave it.\n' * 50), encoding='utf-8')
        paths.append(p)
    data = tmp_path / 'data'; build(paths, data); train(data, 512); prepare(data)
    output = tmp_path / 'run'
    result = run(data, output, steps=1, context=64, seeds=[739], eval_samples=1, dim=32, layers=1)
    assert result['status'] == 'complete' and len(result['runs']) == 6
    assert len({r['initial_weights_sha256'] for r in result['runs']}) == 1
    assert len({r['sample_hashes']['train'] for r in result['runs']}) == 1
    assert len({r['target_tokens'] for r in result['runs']}) == 1
    assert (output / 'source/vox/model.py').exists()
    with pytest.raises(ValueError, match='new experiment'): run(data, output)


def test_atlas_counts_and_equal_degree_controls():
    measured = measure(512)
    assert sum(r['changed'] for r in measured['per_position']) == 298
    assert measured['final']['gold_only'] + measured['final']['hybrid_only'] == 847
    golden = mask_indices(512, angle='golden')
    for policy in ('custom', 'dual', 'counter', 'random'):
        rows = mask_indices(512, angle=policy)
        assert rows[:128] == mask_indices(128, angle=policy)
        for i, row in enumerate(rows):
            assert tuple(sorted(set(row))) == row
            assert all(0 <= j <= i for j in row)
            backbone = {i} | set(range(max(0, i - 16), i)) | {i - off for off in OFFSETS if off <= i}
            assert backbone <= set(row)
            if policy in ('dual', 'random'): assert len(row) == len(golden[i])
    assert mask_indices(512, angle='dual') != golden
    for bad in ('typo', None):
        with pytest.raises(ValueError): mask_indices(32, angle=bad)
    for bad in (-1, 0, 2049, True):
        with pytest.raises(ValueError): mask_indices(bad)


def test_720_rotation_is_not_density_and_sweep_keeps_prefix_identity():
    original = mask_indices(512, angle='custom')
    assert mask_indices(512, angle='rotate720') == original
    sweep = mask_indices(512, angle='sweep720')
    assert sweep != original
    assert sweep[:128] == mask_indices(128, angle='sweep720')
    denser = mask_indices(512, angle='custom', geometric=32)
    assert all(set(a) <= set(b) for a, b in zip(original, denser))
    assert sum(map(len, original)) < sum(map(len, denser)) < 512 * 513 // 2


@pytest.mark.parametrize('experiment', list(MASK_EXPERIMENTS))
def test_research_masks_are_causal_and_reject_incremental_cache(experiment):
    torch.set_num_threads(2)
    cfg = ModelConfig(dim=32, layers=1, heads=2, ffn_dim=64, vocab_size=256, context=128, experiment=experiment)
    model = TinyLlama(cfg).eval()
    ids = torch.randint(0, 256, (1, 96))
    with torch.no_grad():
        full, _ = model(ids)
        prefix, _, cache = model(ids[:, :64], return_cache=True)
        assert torch.allclose(full[:, :64], prefix, atol=1e-6)
        with pytest.raises(ValueError, match='full-sequence'):
            model(ids[:, 64:65], past=cache)
    dense = TinyLlama(replace(cfg, experiment='dense'))
    assert dense.n_parameters() == model.n_parameters()
    out = generate(model, [1], max_new_tokens=2)
    assert len(out) == 2
