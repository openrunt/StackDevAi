"""Deterministic Phi-Pi experimental causal selection mask.

This builds a DENSE BOOLEAN MASK despite few active edges. It is for functional
comparison, not CPU acceleration. The R01 CSR topology is the reference for a
future kernel that actually skips inactive Q/K comparisons.
"""

from functools import lru_cache
from math import sqrt, sin, cos, pi
import random

PHI_CUSTOM = 1.618314159265358979323846
GOLDEN = (1 + sqrt(5)) / 2
OFFSETS = (13, 21, 34, 55, 89)
MASK_EXPERIMENTS = {'fibra': 'custom', 'fibra-golden': 'golden',
                    'fibra-dual': 'dual', 'fibra-counter': 'counter', 'fibra-random': 'random'}


def point(n, angle=None):
    if angle is None:
        angle = 2 * pi / (PHI_CUSTOM * PHI_CUSTOM)
    r = sqrt(n)
    return (r * cos(n * angle), r * sin(n * angle))


@lru_cache(maxsize=32)
def mask_indices(count, mode="fibra", angle="custom", local=16, geometric=8, radius=55):
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 2048:
        raise ValueError("Experimental mask requires 1..2048 tokens")
    if mode not in ("fibra", "dense", "fibonacci", "local"):
        raise ValueError(mode)
    if angle not in ('custom', 'golden', 'dual', 'counter', 'random', 'rotate720', 'sweep720'):
        raise ValueError('Unknown angle policy: ' + str(angle))
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (local, geometric)) or not 0 < radius <= 2048:
        raise ValueError('Invalid local/geometric/radius selection budget')
    if angle in ('dual', 'random') and mode == 'fibra':
        golden = mask_indices(count, mode, 'golden', local, geometric, radius)
        hybrid = mask_indices(count, mode, 'custom', local, geometric, radius) if angle == 'dual' else None
        result = []
        for i, baseline in enumerate(golden):
            if angle == 'dual':
                common = set(baseline) & set(hybrid[i])
                left, right = sorted(set(baseline) - common), sorted(set(hybrid[i]) - common)
                # Alternate exclusive links; same row degree as golden, not the larger union.
                candidates = []
                for j in range(max(len(left), len(right))):
                    pair = ((left, right) if i % 2 == 0 else (right, left))
                    candidates.extend(side[j] for side in pair if j < len(side))
                selected = common | set(candidates[:len(baseline) - len(common)])
            else:
                selected = {i} | set(range(max(0, i - local), i)) | {i - off for off in OFFSETS if i >= off}
                available = sorted(set(range(i)) - selected)
                selected.update(random.Random(739000003 + i).sample(available, len(baseline) - len(selected)))
            result.append(tuple(sorted(selected)))
        return tuple(result)
    golden_angle, custom_angle = 2 * pi / GOLDEN**2, 2 * pi / PHI_CUSTOM**2
    a = golden_angle if angle == 'golden' else 2 * golden_angle - custom_angle if angle == 'counter' else custom_angle
    if angle == 'rotate720':
        a += 4 * pi  # Two whole rotations per integer position: geometrically equivalent.
    elif angle == 'sweep720':
        # Fixed horizon, never the runtime sequence length: prefix behavior must not change.
        a += 4 * pi / (2048 - 1)
    coords = [point(i + 1, a) for i in range(count)]
    out = []
    for i in range(count):
        if mode == "dense":
            allowed = list(range(i + 1))
        else:
            allowed = {i}
            if mode in ("fibra", "local"):
                allowed.update(range(max(0, i - local), i))
            if mode in ("fibra", "fibonacci"):
                allowed.update(i - off for off in OFFSETS if i >= off)
            if mode == "fibra":
                x, y = coords[i]
                matches = sorted(
                    (
                        ((x - a) ** 2 + (y - b) ** 2, j)
                        for j, (a, b) in enumerate(coords[:i])
                        if (x - a) ** 2 + (y - b) ** 2 <= radius**2
                    ),
                    key=lambda it: it[0],
                )
                allowed.update(j for _, j in matches[:geometric])
            allowed = sorted(allowed)
        out.append(tuple(allowed))
    return tuple(out)


def torch_mask(count, device, angle="custom", mode="fibra"):
    import torch

    rows = mask_indices(count, mode=mode, angle=angle)
    mask = torch.zeros((count, count), device=device, dtype=torch.bool)
    for i, cols in enumerate(rows):
        mask[i, list(cols)] = True
    return mask


def stats(count=128, mode="fibra", angle="custom"):
    rows = mask_indices(count, mode=mode, angle=angle)
    allowed = sum(len(x) for x in rows)
    dense = count * (count + 1) // 2
    return dict(
        count=count,
        mode=mode,
        angle=angle,
        eligible_edges=allowed,
        dense_edges=dense,
        fraction=round(allowed / dense, 5),
    )
