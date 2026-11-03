# Fast conversation control

This is an isolated pretrained-model adaptation experiment. It does not replace
the 48M scratch model, AzuCast audio or the live response-pool scheduler.

## Why test 360M

The official SmolLM2-360M-Instruct Q8 reference produced first text in 0.286 seconds
median and completed in 1.274 seconds median over four short local CPU requests.
Those prompts had different lengths; the first request included 0.577 seconds
of model loading. These are text timings, not speech latency or a broad benchmark.
In the separate eight-contribution conversation, the model repeated earlier lines,
confused speakers and sometimes generated another character's label. Its speed
does not make it a finished host.

The pinned base is `HuggingFaceTB/SmolLM2-360M-Instruct`, revision
`a10cc1512eabd3dde888204e902eca88bddb4951`, Apache-2.0. Its Safetensors SHA256 is
`e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
The 361,821,120-parameter model is distinct from the 48,116,224-parameter scratch
model and the adapted 1.5B pilot.

## Reproduce a public-data control

Run these commands from the application directory using its Python environment:

```
python -m pip install -r requirements-public.txt
python tools/prepare_social.py --cache storage/soda-source --output storage/social-data
python tools/public_baseline.py control --project storage/social-data --cache storage/public-source-cache --output storage/smol360-social --reference smollm360 --blocks 2 --target-mode all-turns --train-samples 192 --eval-samples 24 --context 384 --steps 160 --prefix-cache
```

This prepares a SODA-only dataset from pinned CC-BY-4.0 sources. The recorded local
mixed pilot additionally uses the separately reviewed original character curriculum;
it is not the same dataset as this SODA-only command. Dataset cards, attribution,
actual model identity and target counts accompany the export. Audio timing is absent.

Complete later replies retain earlier whole user/assistant pairs as context. Prior
messages do not contribute to target loss. Overlong replies are skipped instead
of cut into unfinished training examples. Legacy first-response controls remain
explicitly available with `--target-mode first`, which is that command's compatibility
default; use `all-turns` for this social experiment.

## Frozen-prefix cache

CPU placement is topology-aware. The earlier first-two-logical-processors rule
selected SMT siblings on the inspected two-core Windows laptop. New placement
chooses separate physical cores inside the inherited allowed set when Windows or
Linux exposes that mapping. It preserves a restrictive allowed set and leaves
placement to the OS when topology is unavailable, including macOS. Torch and BLAS
still request at most two execution threads. Environment reports include the actual
affinity and physical CPU count when available.

The Windows mapping follows Microsoft's
[processor relationship API](https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-processor_relationship);
Linux uses the kernel's documented
[CPU topology attributes](https://docs.kernel.org/admin-guide/cputopology.html).
Earlier timing observations must retain their original placement conditions;
they are not automatically measurements of two separate physical cores.

Only the last two blocks and final normalization weights train in this example.
The optional cache computes the causal frozen prefix once for each selected target.
It stores hidden states before the trainable final normalization and keeps token
positions and complete labels intact. Training then executes the selected tail.
All original layers are restored before export.

This cache is bounded to 512 MiB of additional activation storage. It requires a
fully frozen embedding/prefix and no stochastic prefix dropout. Unsupported model
architectures or an excessive cache fail explicitly. It does not speed up scratch
training where the prefix weights change, and it is not an inference KV cache.
Preparation time is recorded separately from the update loop.

Tests compare cached/full causal logits and tail gradients, check module restoration
on failure, and reject a stochastic prefix. Exact checkpoint resume is tested with
both cached execution and an uncached model with dropout. These checks establish
the tested execution behavior, not a universal speedup or model-quality improvement.

## Stop, resume and preserve results

To request a safe stop, create a file named `stop-request` inside the run directory.
The next completed update is evaluated and saved before export. Resume to a larger
**total** update count with the same command plus `--resume`:

```
python tools/public_baseline.py control --project storage/social-data --cache storage/public-source-cache --output storage/smol360-social --reference smollm360 --blocks 2 --target-mode all-turns --train-samples 192 --eval-samples 24 --context 384 --steps 320 --prefix-cache --resume
```

Explicit resume clears that run's stop marker. An OS-backed single-writer lock
prevents another worker from training the same output directory concurrently.
Optimizer, random state, current weights and best weights share one atomic checkpoint,
so an interruption between separate file writes does not combine inconsistent saves.
The validation set selects the model, including update zero; final test loss is
measured after selection and is not used to choose the checkpoint.

Resume checks the actual base/configuration, data and tokenized targets, context,
adapter layout, trainer/cache source hashes and library versions. Keep the same
source version for a run. A changed protocol requires a new output directory.
Older control checkpoints predating this resume protocol are preserved but are not
silently migrated. Source snapshots are also saved for inspection and reproduction.

## Export and inspect

The run exports full selected Safetensors, a separate selected-tail checkpoint,
tokenizer, provenance, cards and the portable chat application. The tail checkpoint
is a partial full-weight adaptation, not a rank-8 LoRA file. Its report records the
actual trainable parameter count and source identity.

```
python tools/convert_public.py --model storage/smol360-social/export --cache storage/public-source-cache --precision q8_0
ollama create xputty-podcast:smol360-social -f storage/smol360-social/export/Modelfile
python export_chat.py --model storage/smol360-social/export --ollama-model xputty-podcast:smol360-social --port 8794
```

Native conversion uses a pinned llama.cpp toolchain and records hashes. It does not
by itself verify generation. Read actual exported-model responses and distinguish
runtime prompt changes from training. The application retains AGPL-3.0; model and
dataset licenses are recorded separately. A successful export and lower loss do not
establish that a model is entertaining, grounded or ready for unattended streaming.
