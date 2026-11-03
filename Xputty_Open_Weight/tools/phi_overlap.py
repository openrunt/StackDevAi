"""Audit the supplied atlas against the live topology; never executes ZIP content."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vox.topology import mask_indices, PHI_CUSTOM, GOLDEN


def measure(count=2048):
    golden = mask_indices(count, angle='golden')
    hybrid = mask_indices(count, angle='custom')
    rows, shared, only_g, only_h, changed = [], 0, 0, 0, 0
    for i, (g, h) in enumerate(zip(golden, hybrid)):
        a, b = set(g), set(h)
        common, left, right = len(a & b), len(a - b), len(b - a)
        shared += common; only_g += left; only_h += right; changed += a != b
        rows.append({'n': i + 1, 'shared': common, 'g_only': left, 'h_only': right,
                     'jaccard': common / (common + left + right), 'changed': a != b})
    delta = 2 * math.pi / PHI_CUSTOM**2 - 2 * math.pi / GOLDEN**2
    controls = {}
    for policy in ('golden', 'custom', 'dual', 'counter', 'random', 'rotate720', 'sweep720'):
        selected = mask_indices(count, angle=policy)
        controls[policy] = {'edges': sum(map(len, selected)),
                            'changed_rows_from_hybrid': sum(a != b for a, b in zip(hybrid, selected)),
                            'same_row_degrees_as_golden': all(len(a) == len(b) for a, b in zip(golden, selected)),
                            'mask_sha256': hashlib.sha256(json.dumps(selected).encode()).hexdigest()}
    denser = mask_indices(count, angle='custom', geometric=32)
    union = [set(a) | set(b) for a, b in zip(golden, hybrid)]
    controls['hybrid_32_neighbors'] = {'edges': sum(map(len, denser)), 'geometric_neighbor_budget': 32}
    controls['golden_hybrid_union'] = {'edges': sum(map(len, union)), 'budget_matched': False}
    controls['dense_causal'] = {'edges': count * (count + 1) // 2}
    return {'schema': 'xow-phi-overlap/1', 'constant': PHI_CUSTOM, 'golden': GOLDEN,
            'golden_angle_degrees': 360 / GOLDEN**2, 'hybrid_angle_degrees': 360 / PHI_CUSTOM**2,
            'delta_angle_degrees': math.degrees(delta), 'per_position': rows,
            'final': {'shared': shared, 'gold_only': only_g, 'hybrid_only': only_h,
                      'jaccard': shared / (shared + only_g + only_h), 'changed_rows': changed},
            'controls': controls, 'source_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'vox/topology.py').read_bytes()).hexdigest(),
            'limitations': ['Topology measurements, not semantic or speech quality',
                            'Dual is budget-matched edge selection, not complex-vector interference',
                            'All masks use dense attention kernels; no sparse-kernel speedup established']}


def verify_atlas(path, measured):
    with zipfile.ZipFile(path) as z:
        items = z.infolist()
        if len(items) > 64 or sum(i.file_size for i in items) > 20 * 1024**2:
            raise ValueError('Atlas exceeds inspection limits')
        names = set()
        for item in items:
            normalized = PurePosixPath(item.filename.replace('\\', '/'))
            if normalized.is_absolute() or '..' in normalized.parts or ':' in item.filename or item.filename in names:
                raise ValueError('Unsafe or duplicated atlas member')
            names.add(item.filename)
        supplied = json.loads(z.read('phi_overlap_metrics.json'))
        if len(supplied['per_position']) != len(measured['per_position']):
            raise ValueError('Atlas context differs from the requested measurement')
        for key in ('constant', 'golden', 'golden_angle_degrees', 'hybrid_angle_degrees', 'delta_angle_degrees'):
            if not math.isclose(supplied[key], measured[key], rel_tol=0, abs_tol=1e-10):
                raise ValueError('Atlas constant mismatch: ' + key)
        for actual, expected in zip(measured['per_position'], supplied['per_position']):
            if any(actual[key] != expected[key] for key in ('n', 'shared', 'g_only', 'h_only', 'changed')) or not math.isfinite(expected['jaccard']) or abs(actual['jaccard'] - expected['jaccard']) > 1e-12:
                raise ValueError('Atlas row mismatch: ' + str(actual['n']))
        for key, value in supplied['final'].items():
            if key not in measured['final'] or measured['final'][key] != value:
                raise ValueError('Atlas total mismatch: ' + key)
        csv_rows = list(csv.DictReader(io.StringIO(z.read('phi_overlap_by_position.csv').decode('utf-8-sig'))))
        if len(csv_rows) != len(measured['per_position']):
            raise ValueError('CSV row count mismatch')
        cs = cg = ch = 0
        for row, numeric in zip(csv_rows, measured['per_position']):
            cs += numeric['shared']; cg += numeric['g_only']; ch += numeric['h_only']; n = numeric['n']
            exact = {'position': n, 'shared_edges_row': numeric['shared'], 'golden_only_edges_row': numeric['g_only'],
                     'hybrid_only_edges_row': numeric['h_only'], 'changed_attention_row': int(numeric['changed']),
                     'cumulative_shared': cs, 'cumulative_golden_only': cg, 'cumulative_hybrid_only': ch}
            approximate = {'phase_difference_deg': n * measured['delta_angle_degrees'],
                           'phase_alignment_cos': math.cos(math.radians(n * measured['delta_angle_degrees'])),
                           'edge_jaccard_row': numeric['jaccard'], 'cumulative_jaccard': cs / (cs + cg + ch)}
            if any(int(row[k]) != v for k, v in exact.items()) or any(not math.isfinite(float(row[k])) or abs(float(row[k]) - v) > 1e-8 for k, v in approximate.items()):
                raise ValueError('Atlas CSV mismatch: ' + str(n))
        return {'zip_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'all_json_and_csv_rows_match': True,
                'members': {i.filename: hashlib.sha256(z.read(i)).hexdigest() for i in items}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--atlas', type=Path); p.add_argument('--count', type=int, default=2048)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Choose a new output file; previous measurements are retained')
    result = measure(a.count)
    if a.atlas: result['atlas_verification'] = verify_atlas(a.atlas, result)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_position'}, indent=2))


if __name__ == '__main__': main()
