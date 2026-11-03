# Compression objective and first actual compact student

The objective is a smaller model with more useful capability per byte and lower
response latency. All three must be evaluated together. Changing the attention
graph at a fixed parameter count was a preliminary topology experiment; it did
not demonstrate model compression or retained knowledge.

This experiment actually removes stored decoder layers. It compares:

1. The pinned, unadapted SmolLM2-135M-Instruct teacher.
2. A 46M student recovered with ordinary assistant-target cross-entropy (CE).
3. The identical small architecture recovered with CE plus teacher supervision.

**Result: the first compact candidates are smaller and faster, but fail language
quality. They are not recommended podcast hosts or replacements for current models.**

## What was built

The base is `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. Its original Safetensors SHA256 is
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
Upstream model weights retain Apache-2.0; application tooling retains AGPL-3.0.

The teacher has 134,515,008 parameters and 30 decoder blocks. Both students retain
blocks `[0,7,14,22,29]`, the original tokenizer, embeddings and output vocabulary.
They have 46,012,608 parameters. Embeddings/output weights remain frozen and tied;
the retained decoder blocks and final norm are trained. This is pretrained layer
pruning followed by recovery, not a newly pretrained 46M model, geometric masking,
or an implementation of ShortGPT's importance-based layer selection.

The source is the frozen public SODA/original dialogue mixture previously used
for the social pilots: 192 complete assistant training targets, 24 validation and
24 test targets, each with up to 384 tokens of context. Earlier messages are
context; the current full response and end marker are prediction targets. Existing
conversation/topic partitions remain separate. Original pretraining overlap is
unknown. There are no speech timing labels.

Both runs complete 256 updates from the same retained-layer initialization and
same sampled targets. Validation assistant NLL selects update 224 in both cases;
the final test does not select a checkpoint. Profiles in the chat are runtime
instructions, not four independent trained model weights.

| Model | Validation assistant NLL | Test assistant NLL |
| --- | ---: | ---: |
| Full teacher | 2.10114 | 2.52395 |
| Pruned student before recovery | 17.77184 | Not used for checkpoint selection |
| CE-only compact control | 5.79415 | 5.29481 |
| Teacher-guided compact student | 5.87165 | 5.34416 |

The large initial loss increase is evidence that this aggressive removal damaged
the pretrained model. Recovery helps, but does not restore teacher-level behavior.
Teacher guidance lowers the final test divergence from the teacher (3.42426 to
2.68158, temperature-scaled coarse KL) while slightly worsening test target NLL.
Closer teacher probabilities therefore do not establish more useful knowledge.

## Implemented teacher supervision

For each known response position, the frozen teacher supplies its top 64 token
probabilities at temperature 2 plus the probability mass of all other tokens.
The student matches those 65 categories with forward KL. The omitted vocabulary
is one aggregate category; this is not full-vocabulary KL, a reverse-KL MiniLLM
implementation, a learned phase-consistency objective or lossless knowledge transfer.

The guided objective is `0.5 * CE + 0.5 * temperature^2 * coarse_KL`; the control
uses CE alone. The teacher has the same tokenizer as the student, avoiding an
unjustified token-probability alignment across different vocabularies.

The shared teacher cache has an OS-backed single-writer lock, content checksum,
base/config/tokenizer/template identities, input/label identities and source/library
version checks. Each run atomically checkpoints current and selected trainable
weights, optimizer and RNG state. Explicit resume must preserve the protocol and
may extend the total update target. Tests compare uninterrupted and resumed weights
exactly. Stop requests save completed work; changed inputs are rejected.

## Actual portable size and response measurements

All three native exports use the same pinned Q8_0 converter. These are actual
standalone GGUF sizes, not adapter sizes or theoretical bytes-per-parameter.

| Export | File bytes | Median first text, warm | Median decode tokens/s |
| --- | ---: | ---: | ---: |
| 135M teacher | 144,810,944 | 0.138 s | 97.4 |
| 46M CE control | 50,679,136 | 0.051 s | 175.3 |
| 46M teacher-guided | 50,679,136 | 0.039 s | 252.8 |

This is about 65.0% smaller in deployed bytes. Native weights were checked against
their actual registered Ollama hashes. Both student HF exports were separately
reloaded with no missing or unexpected tensors; all 46 selected trainable tensors
matched the selected checkpoint and frozen embeddings matched the original base.
Conversion/loading checks are not answer-quality acceptance.

Measurements are one sequential CPU session on this laptop, two inference threads,
2,048 context limit, greedy decoding, after local training/tests finished. Warm
figures summarize four prompts. Generated lengths differ, so decode rates are
descriptive observations rather than an equal-length throughput benchmark. Short,
collapsed replies especially cannot be interpreted as improved audience experience.
First text is not first audio, and a configured context limit is not long-context
acceptance.

## Answer quality failed

Complete raw replies are preserved locally. For the same slow-zombie prompt:

- The teacher gave grammatical but contradictory advice, including calling slow
  zombies agile and fast. It often exceeded the requested two-sentence length.
- The small CE model replied: "Sure, I'm to see a very thing and what of it's no time."
- The small guided model replied: "I, a well, what it's time."

Sixteen authored forced-choice checks separately covered basic recall, supplied
scene state, causality, negation and unavailable information. Correct answer
positions are balanced; always choosing one letter gets 4/16. Observed results
were 2/16 for the teacher, 0/16 for CE and 5/16 for the guided student. Invalid or
truncated JSON occurred 0, 14 and 3 times respectively. The guided student's
mostly-A output and this tiny sample do not establish better knowledge; its free
speech is plainly unusable. These checks are developmental diagnostics, not a
validated general-knowledge benchmark. Formatting failure is reported separately.

The teacher itself is not strong enough in these tests to certify the desired
podcast behavior. Merely matching it more closely would be insufficient. A usable
candidate needs substantially better free responses and reliable held-out checks
alongside size and speed gains. Neither compressed candidate passes that gate.

## Reproduce the actual compression

The existing public preparation tools produce `data/train.jsonl`, `val.jsonl`,
`test.jsonl` and `conversations.jsonl`. Keep their provenance and whole-source
partitions. Use separate output directories for different objectives.

```
python tools/compact_distill.py --base PATH/SmolLM2-135M-Instruct --dataset PATH/FROZEN_SOCIAL_DATA --output storage/NEW-ce --cache-file storage/NEW-teacher.pt --keep 5 --steps 256 --train-samples 192 --eval-samples 24 --eval-every 32 --alpha 0
python tools/compact_distill.py --base PATH/SmolLM2-135M-Instruct --dataset PATH/FROZEN_SOCIAL_DATA --output storage/NEW-kd --cache-file storage/NEW-teacher.pt --keep 5 --steps 256 --train-samples 192 --eval-samples 24 --eval-every 32 --alpha 0.5
```

Add `--resume` and a larger total `--steps` only to continue the same run. Snapshot
source is retained because editing trainer code intentionally invalidates resume.
The exported full model is standalone; the smaller trainable checkpoint still
needs the exact recorded original base to reconstruct frozen weights.

```
python tools/convert_public.py --model storage/NEW-kd/export --cache PATH/PUBLIC_CACHE --precision q8_0
ollama create xputty-compact:YOUR_RUN -f storage/NEW-kd/export/Modelfile
python export_chat.py --model storage/NEW-kd/export --ollama-model xputty-compact:YOUR_RUN --port 8796
```

The printed chat link is private. Keep experimental models separate from a model
recommended for an unattended stream. Their native file is portable, but portability
does not repair their answers.

For comparison, create a local JSON list containing `label`, `tag`, `precision`
and the absolute `file` path for every registered export. Then run:

```
python tools/compact_benchmark.py --models PATH/models.json --output storage/NEW-comparison.json
```

The comparison retains all raw answers, format failures, selected letters, file
hashes, actual byte counts and streamed timings. It does not promote a model or
merge the three requirements into a score that could hide broken language behind
fast inference. Private data, caches and weights are excluded from source releases.

Primary references: [knowledge distillation](https://arxiv.org/abs/1503.02531),
[MiniLLM](https://arxiv.org/abs/2306.08543) and
[ShortGPT](https://arxiv.org/abs/2403.03853). These motivate distinct comparisons;
their published results are not reproduced or claimed by this pilot.
