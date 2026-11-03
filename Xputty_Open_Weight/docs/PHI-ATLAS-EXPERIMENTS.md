# Phi overlap: reproduction and controlled pilots

Recorded 2026-11-02. The supplied atlas and all four accompanying notes were
reviewed, including both calculation scripts, every numeric JSON/CSV row and the
three figures. The ZIP's SHA256 is
`59472140e13fc1c90561e10dc47534f0eb73ff1209103628e3f42577cde9de57`.
Its scripts use `/mnt/data`; the portable audit below uses an explicit destination
and does not execute or extract archive code. Source files remain unchanged.

## What reproduced

The implementation uses the supplied literal `1.618314159265358979323846`,
approximately `1.618 + pi/10000`, rather than exact `golden_ratio + pi/10000`.
The golden angle is 137.5077640500 degrees; the hybrid is 137.4601561291 degrees.
The relative change is -0.0476079209 degrees per position.

At 2,048 positions the masks have 46,568 shared directed links, 5,028 golden-only
and 5,180 hybrid-only links. Their intersection/union is 82.0206%. A total of
1,834 rows change (89.5508%); this is a different denominator. All 2,048 JSON rows
and all CSV columns match the live implementation, including phase alignment.

The figure labels correctly distinguish mask overlap from hypothetical complex
vector alignment. One plotting detail needs care: the supplied 128-position band
curve uses `step(bins[1:], band, where='post')`, shifting each displayed band one
bin right. The underlying CSV/JSON and cumulative curve are correct. For a new
band plot, use all bin edges with the last band value repeated and `where='post'`.
Do not read the shifted orange segment as its exact
token range.

```
python tools/phi_overlap.py --atlas PATH/phi_overlap_atlas.zip --output storage/NEW-audit.json
```

The audit also reports graph hashes and edge budgets for the additional controls.
It rejects changed measurements, nonfinite CSV values, unsafe ZIP member paths,
duplicate members and oversized archives. Previous results are never overwritten.

## Implemented attention controls

| Experiment | Attention selection | Budget |
| --- | --- | --- |
| dense | All causal links | Dense reference |
| fibra-golden | Exact golden geometry plus local/Fibonacci backbone | Original golden graph |
| fibra | Supplied hybrid geometry plus the same backbone | Original hybrid graph |
| fibra-dual | Intersection plus alternating golden/hybrid exclusive links | Exactly golden's degree in every row |
| fibra-counter | Angular perturbation mirrored across the golden angle | Same neighbor-selection settings |
| fibra-random | Local/Fibonacci backbone plus seeded random older links | Exactly golden's degree in every row |

The dual variant is deliberately a budget-matched graph mixture. It does not
superpose complex hidden vectors and is not named or evaluated as that operation.
Counter-detuning tests sensitivity to the direction of angular change. Random
selection tests whether any nonlocal links are sufficient. All keep parameter
count fixed. Golden/custom/counter degrees can differ slightly because geometric
links can duplicate backbone links; their measured budgets are reported.

All arms still use dense Boolean masks and ordinary dense attention kernels.
Fewer eligible edges do not establish fewer actual kernel operations. Standard
GGUF/Hugging Face export continues to reject these research architectures because
ordinary deployment runtimes do not implement their attention rule. Dense remains
the browser training default. Experimental generation recomputes its full window;
incremental cache use is explicitly rejected, including single-token calls.

## Actual matched-data training

The new runner freezes samples, starting weights, tokenizer, source hashes and
evaluation windows per seed. Every arm sees the same inputs and target-token
budget. Checkpoints, reports and source snapshots are retained locally. Arm order
rotates across seeds; observed elapsed times are not isolated hardware benchmarks.
No held-out examples select the winning architecture in these short diagnostics.

| Model / setting | Dense | Golden | Hybrid | Balanced dual | Counter | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 537,696 parameters, three seeds, 32 updates: mean test loss | 8.71441 | 8.72499 | 8.72473 | 8.72444 | 8.72481 | 8.72428 |
| 48,116,224 parameters, one seed, eight updates: test loss | 7.58933 | 7.61907 | 7.62059 | 7.62175 | 7.62465 | 7.61685 |

Both use 512-token windows from the same prepared public SODA conversation
baseline. The micro arms each process 16,384 input positions per seed; the 48M
arms each process 4,096. Micro evaluation uses eight held-out windows per split;
48M uses four. These are very short initial-learning diagnostics, not converged
models, independent quality benchmarks or complete rejection of the hypothesis.
There is **no observed hybrid advantage** here. Dense has lower held-out loss
in these pilots. The research masks also took longer in the measured update loops.
No mask change is promoted to the default on this evidence.

```
python tools/phi_ablation.py --data storage/PUBLIC_PROJECT/data --output storage/NEW-micro-run
python tools/phi_ablation.py --data storage/PUBLIC_PROJECT/data --output storage/NEW-48m-run --dim 512 --layers 12 --heads 8 --ffn-dim 1472 --context 512 --steps 8 --seeds 739 --eval-samples 4
```

The runner accepts larger explicit budgets. The commands above reproduce the
reported scope, not a recommended final training duration. A 512-token pilot does
not measure the effects observed at token 2,048 or episode-length reasoning.

## The 720-degree question

A constant 720-degree rotation is two whole rotations. Adding it to every integer
position's angle produces the same selected graph: all 2,048 rows match the hybrid,
and the serialized graph hashes are identical. A rigid rotation of the entire
point field also preserves all pairwise distances. Neither operation adds density.

The separate `sweep720` audit adds `4*pi/2047` radians per position, giving two
extra relative sweeps between positions 1 and 2,048. Its horizon is fixed at 2,048,
not the current input length, so extending a sequence does not change its earlier
attention graph. This is an audit control; it has not been trained or deployed.

| Selection at 2,048 positions | Eligible connections |
| --- | ---: |
| Existing hybrid, eight geometric neighbors | 51,748 |
| Hybrid plus constant 720 degrees per step | 51,748 (identical graph) |
| Hybrid plus two additional sweeps across the fixed horizon | 55,651 |
| Golden/hybrid graph union | 56,776 |
| Hybrid, 32 geometric neighbors | 96,883 |
| Full dense causal attention | 2,098,176 |

The sweep changes 2,013 rows but increases total edges by only about 7.5%. It changes
selection more than density. Even the 32-neighbor graph is only about 4.62% of
dense causal connectivity. More graph links do not create source information, and
ordinary dense attention already exposes every earlier token. Multiple phase
channels or a learned mixture would require separate capacity and compute controls.

## Reversible context views

```
python tools/prepare_phase_views.py --source PATH/conversations.jsonl --output storage/NEW-phase-views --tokenizer PATH/LOCAL_TOKENIZER
```

This produces canonical, character-reversed, word-reversed and context-turn-reversed
views. Whitespace, Unicode, speaker metadata and timestamps recover exactly under
the documented inverse. System instructions and the final assistant target remain
untouched. Each view retains its source identity and split; a source crossing
holdouts is rejected. Tampering fails the recovery hash check. Nothing is treated
as an extra original episode or an actual new speech timing observation.

On the 96 actor records from the original cast curriculum, four views produce 384
records, with exact recovery. Using the selected Qwen tokenizer:

| View | Total context tokens |
| --- | ---: |
| Canonical | 24,513 |
| Character reversal | 42,733 |
| Word reversal | 24,507 |
| Context-turn reversal | 24,513 |

Character reversal costs about 74.33% more context tokens. Matched character counts
would therefore confound a training comparison. Token totals here use the chat
template's default behavior; they are not measured native inference throughput.

The package is a tested representation/recovery artifact. The ordinary trainer
does not silently ingest reversed contexts as natural causal conversation. Exact
software recovery is not evidence that a neural student learned semantic recovery.

## What remains unproven

- Teacher-logit phase-consistency training, reconstruction/contrastive objectives
  and their capacity/compute tradeoff. These pilots train ordinary next-token loss.
- Fewer distant semantic errors on held-out canonical dialogue. No near-miss score
  is reported; token overlap or embedding similarity alone can miss reversed
  causality, speaker swaps and negation.
- A converged multi-seed 48M or 2,048-position benefit, actual sparse-kernel
  acceleration, trained speech timing, expressive delivery or AzuCast/Quest gains.
- A coherent deployed podcast host. The latest 1.5B cast adaptation still confuses
  scenario meaning and repeats itself. Valid exports do not remove that failure.

Related primary work separates these mechanisms: [knowledge distillation](https://arxiv.org/abs/1503.02531)
uses teacher supervision; [reverse training](https://arxiv.org/abs/2403.13799) studies
directional fact learning; [RoFormer](https://arxiv.org/abs/2104.09864) rotates query/key
representations. None establishes a special semantic advantage for this constant
or for the graph mixture implemented here.
