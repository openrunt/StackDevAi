# Podcast model experiments

The 48M scratch model and the pretrained conversation controls are separate
experiments. A pretrained model is never silently substituted for scratch weights.
The application source distribution includes reproducible tools, not private
transcripts, local access keys or model weights.

## What the experiments have established

* The 48,116,224-parameter model completed 200 public-conversation updates with
  mixed 256/512/1024/2048 windows. Fixed held-out loss fell from 9.14991 to 6.43569.
  Its responses were still fragmented. This is learning evidence, not a coherent host.
* The 134,515,008-parameter pretrained control did not improve fixed validation
  loss during its 40-update adaptation; selection correctly retained update zero.
* A 494,032,768-parameter pretrained student trained on 80 original, character-
  conditioned examples (64 train, 8 validation, 8 test). The initial teacher
  outputs contained factual and stylistic defects. A separately versioned editorial
  review preserved every before/after edit and replaced all 80 targets.
* On that reviewed curriculum, the student's best checkpoint was update 80 of
  160. Validation loss fell from 3.57839 to 3.22448, but independent free responses
  were still weak and sometimes wrong. Lower loss on eight examples did not
  establish useful podcast behavior. The review was performed by the coding agent,
  not independently verified human annotation.

The pilot uses four original characters: Lyra (curious), Orion (skeptical), Nova
(playful) and Kai (practical). Whole curriculum topics remain separate between
train, validation and test. Original pretraining contamination is not ruled out.
Factual cards cite NASA; fictional situations and target dialogue are original.
No timing labels are fabricated.

## Distillation and quantization do different jobs

[Sequence-level knowledge distillation](https://aclanthology.org/D16-1139/) trains
a student on teacher-generated target sequences. Our first pilot implements that
family of method, followed by an explicit editorial-review variant. It does not
transfer all of a teacher's knowledge, and bad teacher replies remain bad targets
unless reviewed. The 0.5B experiment is a negative result, preserved for comparison.

Quantization reduces the numerical storage precision of weights. It does not make
a small, incoherent language model knowledgeable. Our actual deployment tests
cover F16 and Q8_0 GGUF conversion; model hashes and complete responses accompany
the tests. An F16 0.5B export generated at a median 19.33 decode tokens/second in
one two-thread CPU run, while still failing qualitative checks. Decode speed is
not first-audio latency or natural conversation timing.

[QLoRA](https://arxiv.org/abs/2305.14314) demonstrates training low-rank adapters
through frozen quantized weights. Our CPU GGUF experiment uses the same broad
principle but different implementation and formats: existing Q4_K/Q6_K base
tensors, rank-8 adapters, on-demand dequantization, and a chunked vocabulary head.
It is not the paper's NF4/double-quantization CUDA implementation.

[AWQ](https://arxiv.org/abs/2306.00978) uses activation statistics to protect
important channels during quantization. That is a candidate comparison, not an
algorithm implemented or benchmarked here. No AWQ/GPTQ benefit is claimed.

## CPU GGUF adapter experiment

`tools/train_gguf_adapter.py` requires an existing Qwen2 GGUF plus the matching
local Hugging Face configuration/tokenizer. The run preserves a SHA-256 identity
for the base, corpus, context and adapter layout. It never overwrites its base.
Each frozen prefix layer is expanded once across all independent examples;
only one frozen decoder layer is dense at a time. Cached contextual states are
separated by partition and reused for training the final decoder layers.

Only assistant response positions enter the objective. Validation selects the
best checkpoint, including update zero; the final test is evaluated after that
selection. Checkpoints contain adapter/optimizer state for explicit resume.
Adapters merge into a standalone mixed-precision GGUF: adapted matrices use Q8_0,
and unchanged tensors retain their original formats. The model still has 1.5B
parameters. This is adaptation, not a distilled 48M model.

The first actual 1.5B run completed 80 updates, selected update 70 by validation,
and exported a 1,203,607,168-byte mixed GGUF plus a 5.3 MB adapter. Its individual
responses improved but its eight-contribution conversation echoed earlier replies.
It is retained as an unsuccessful conversation pilot, not recommended as a host.
The optimized update loop took 266.4 seconds, excluding prefix preparation and
export; observed post-update RSS stayed below 2.86 GB. These are local CPU
measurements, not a comparison with a CUDA QLoRA implementation.

An additional **unadapted Qwen3.5-4B reference** is used to investigate the quality
gap. Its exact model identity is recorded separately; it is not renamed as trained
Xputty weights. Prompt-policy changes and sampling changes are recorded as runtime
configuration experiments, not claimed as learned improvements. The initial
reference makes grounding errors; fresh supplied-premise checks test the corrected
configuration. Full responses, including unsuccessful ones, must remain available.

```
python -m pip install -r requirements-public.txt
python tools/train_gguf_adapter.py --dataset PATH_TO_FROZEN_PILOT --base BASE.gguf --config MATCHING_TOKENIZER_FOLDER --output storage/NEW_RUN --steps 80
```

Add `--resume` and a larger total `--steps` to continue the same run. Create a new
run when changing data, base weights, context or adapter layout. The local pilot
curriculum was generated with `tools/podcast_pilot.py collect`, then `review`;
both preserve source/review manifests. Review its targets before a broader build.

The GGUF linear input/adapter gradients are checked against dense PyTorch;
layerwise cached-prefix calculations are checked against full causal execution;
merge tests verify actual weight changes and original-base preservation. A native
float32 reference comparison on the local 1.5B model had matching top token and
maximum absolute logit difference 0.01051. Native Q4 activation arithmetic differs
more (maximum 1.10434 on that single prompt). This is not bit-identical Q4 training.

## Complete later-reply curriculum

The original comparison used each conversation's first response only. The current
adapter trainer defaults to **all complete assistant responses**, with earlier
whole user/assistant pairs retained as space permits. Prior messages are masked;
the complete current reply and its end marker are learned. An overlong reply is
skipped instead of cut into an unfinished target. This changes the training
protocol; earlier first-reply results remain separately identified.

The additional source is [AllenAI SODA](https://huggingface.co/datasets/allenai/soda),
revision `fdc848ab0183208ea7808206c91c724414d0a071`, declared CC-BY-4.0.
It is synthetic social dialogue, not audio recordings or verified factual knowledge.
Source attribution, original selected records, dataset card, exact source hashes
and whole-conversation partitions are preserved. Related narratives and upstream
pretraining overlap are not fully decontaminated.

Reproduce a small SODA-only adaptation from a fresh checkout:

```
python -m pip install -r requirements-public.txt
python tools/prepare_qwen_adapter.py --cache storage/qwen15-base
python tools/prepare_social.py --cache storage/soda-source --output storage/social-data
python tools/train_gguf_adapter.py --dataset storage/social-data --base storage/qwen15-base/base.gguf --config storage/qwen15-base/tokenizer --output storage/social-run --steps 160 --context 384
```

Preparation explicitly downloads pinned public source files. It does not start on
opening the canvas. Supply `--gguf EXISTING_FILE` to the first command to reuse the
exact pinned base. The social preparer optionally accepts `--original` with an
existing reviewed character curriculum; omit it for SODA-only training. In the
SODA-only variant, the four named character profiles are runtime prompts, not
individually trained personalities. The recorded mixed local pilot has 192 train,
24 validation and 24 test targets. Of its training targets, 111 retain an earlier
assistant contribution and all 192 contain their complete assistant end marker.

The optional Hugging Face conversion uses the actual selected GGUF:

```
python tools/gguf_to_hf.py --source storage/social-run/export --output storage/social-run/hf
```

It streams F16 Safetensors shards with an index, preserves distinct embedding and
output matrices when quantization gives them different values, copies provenance,
and hashes every shard. F16 rounding is explicit. A round-trip test checks tensor
names/values and Transformers output on a small Qwen2 model. A larger model's own
runtime acceptance must also be recorded after its conversion. The local selected
1.5B social pilot has now passed that check: Transformers loaded all 339 tensors
without missing or unexpected keys, sampled F16 values matched the source GGUF,
and actual generation ran. See [the results and remaining work](EXPERIMENT-RESULTS.md).

## Interaction and verification improvements

The test chat displays a submitted message immediately, shows generation elapsed
time, preserves a new draft typed while waiting, and flags replies that hit their
token limit. Native chat streams generated text incrementally. When switching
characters, other characters' contributions become peer context; they are not
misrepresented as that character's own previous answers. Native model selection
is checked against its exact GGUF hash. A
reference can explicitly disable hidden reasoning and record its sampling settings;
changing those settings is not described as weight training.

Unchanged scratch-model re-exports can reuse a successful Transformers parity check
only when actual source/export weights, model configuration, tokenizers, verifier
code, library versions and platform match. The parity report and saved native
request also retain their hashes. Changed inputs or failed checks trigger execution
again. The build ledger records whether the previous check was reused.

## Compare complete responses

```
ollama create xputty-podcast:YOUR_RUN -f storage/NEW_RUN/export/Modelfile
python tools/probe_podcast.py --tag xputty-podcast:YOUR_RUN --weights storage/NEW_RUN/export/model-mixed.gguf --output storage/NEW_RUN/podcast-evaluation.json
python export_chat.py --model storage/NEW_RUN/export --ollama-model xputty-podcast:YOUR_RUN --port 8792
```

The chat checks the exact GGUF hash against Ollama, supports all four personas,
preserves the character instructions while dropping oldest context, and never
falls back to a remote or different model. Keep the printed access link private.
The probe script records eight original behavioral prompts and eight successive
contributions for manual review. Its fixed order is an experimental control, not
the production response-pool scheduler. No speech, VR or audience tests are implied.

## Release gate

A model is usable only after reviewing complete unseen responses for coherence,
relevance, contradiction, grounding, repetition and finished thoughts, then testing
the actual exported native weights. Loss, nonempty output, successful conversion
and keywords alone do not satisfy that gate. Keep unsuccessful checkpoints and
their measurements distinct from a model recommended for use.
