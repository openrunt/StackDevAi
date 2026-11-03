# Public conversation experiments: results and remaining work

Recorded 2026-11-02. These are small development experiments on a two-core Windows
CPU laptop with no CUDA. Private podcast transcripts were not used in these public
pilots. Model weights are local artifacts, separate from the software release.

## What is concrete

| Experiment | Actual outcome | Interpretation |
| --- | --- | --- |
| Scratch 48M, 200 updates | Validation token loss 9.150 to 6.436; fragmented generated language | Working training/export control; not coherent conversation |
| SmolLM2 135M control | Validation selected update zero | Adaptation did not beat the pretrained reference |
| Reviewed 0.5B distillation, 160 updates | Selected update 80; lower loss, weak free responses | Loss improvement did not establish host quality |
| First-reply 1.5B adapter, 80 updates | Selected 70; repetitive multi-character responses | Later conversation needed explicit training and correct role history |
| All-reply social 1.5B adapter, 160 updates | Selected 120; validation 2.0991 to 1.7636, test 2.0821 | A better test candidate; responses remain shallow and sometimes weakly reasoned |

These losses use different corpora and targets; do not compare their absolute values
as a model ranking. Small adaptation holdouts cannot exclude upstream pretraining
overlap. The 1.5B variants retain a pretrained Qwen base: they are not compressed
48M models. Their actual exported embedding/output matrices are untied because
the source quantization stores different values, giving 1,777,088,000 stored values.

## Selected social pilot

- Data: pinned AllenAI SODA synthetic social dialogue (CC-BY-4.0) plus 80 original,
  coding-agent-reviewed character examples. The SODA-only preparation command is
  available for independent reproduction; it is a different dataset from this mix.
- Targets: 192 train, 24 validation, 24 test, partitioned by source conversation.
  All 192 training responses include their end marker; 111 include a previous
  assistant contribution. Prior messages are context, not prediction targets.
- Adaptation: rank-8 LoRA on the last four layers; 1,318,912 trained parameters.
  The frozen prefix is cached. Context 384; the wider base context is not evidence
  of long-context adaptation. Prefix preparation took about 26 minutes; the update
  loop took 723 seconds. Reported update time excludes preparation and export.
- Selected native file: 1,203,607,168-byte mixed Q4_K/Q6_K/Q8/F32 GGUF.
  SHA256 `6be5a8261161bd367a6ad9e4a38c57d57d799e4ea89aa7c037ee95ba9f467d93`.
- Portable adapter: about 5.3 MB; requires the exact recorded base and layout.
- Hugging Face: seven Safetensors shards, 339 tensors, 3,554,176,000 tensor bytes.
  Actual Transformers loading and generation passed. First/last rows of every
  matrix matched F16-rounded source values. Native quantized arithmetic is not
  claimed bit-identical to F16 Transformers execution.

The eight-contribution actor-perspective test produced shorter, distinct replies
about fast/slow zombies instead of the first adapter's recurring generic question.
It also began with "I'm not sure what you're talking about" and often gave obvious
points or follow-up questions. This does not meet the interesting-host release gate.
The same actor-role protocol was checked against the unadapted 1.5B base; it was
wordier and sometimes leaked speaker labels or closed the discussion prematurely.
This is a small qualitative comparison, not a broad preference score.

An unadapted 4B reference tested with the same actor-role protocol and hidden
reasoning disabled gave more elaborate, character-specific contributions. It also
converged on repeated "cognitive dissonance" framing and hit the 96-token limit
in several responses. Greater model size helps capacity, but does not by itself
solve finished delivery, grounding or conversational variety.

## Delivery and debugging fixes

The additional unadapted SmolLM2-360M Q8 control was fast in four short requests:
median first text 0.286 seconds and total 1.274 seconds. It also repeated earlier
contributions, confused speakers and emitted assistant-style commentary. A separate
160-update social adaptation selected update 160 (validation 1.89593 to 1.84807;
test 2.20930), but its native group responses still included advice about how to
speak, contradictions and repeated lines. This remains a negative host-quality
result. A higher count of replies ending before the limit did not make it coherent.

The optional frozen-prefix control matched the uncached 360M losses exactly at
updates 1, 10, 20, 30, 40, 50 and 60. Its update phase reached 60 in 120.61 seconds
versus 459.23 seconds uncached. Preparation added 400.71 seconds and 149,832,960
bytes of activation storage. The full cached 160-update phase took 296.32 seconds,
excluding preparation and export. Short experiments may not amortize that setup.

Those paired cached/uncached Torch measurements used logical processors 0 and 1.
Windows topology inspection subsequently showed both are siblings of one physical
core. The CPU limiter now selects one allowed logical processor per physical core
(0 and 2 here). This fixes placement; a new isolated speedup measurement has not
been established. Native Ollama inference runs in a separate process.

A separate original 24-episode cast curriculum kept complete prior self/peer
contributions and held entire topics out. The 1.5B adapter selected update 80 of
160: validation 3.75173 to 3.45463; test 3.40349. Its free conversation still
repeated generic questions and confused the topic. This is another negative
host-quality result. A six-request identity-framing ablation also found that
simple direct character instructions sometimes worked better than elaborate
roleplay instructions, including on the unadapted base. Prompt effects must be
separated from weight changes; six replies are not a general quality evaluation.

Chat now shows submitted input immediately, preserves a new draft while waiting,
streams native generated tokens, and marks length-limited replies. Switching the
selected character correctly maps peers to user context and self to assistant
history. A short natural-dialogue runtime prompt differs from the training prompt;
this change is recorded separately and is not credited to training alone.

A real authenticated HTTP streaming test returned 64 incremental text records,
rejected missing authorization (401) and a wrong Origin (403). First text arrived
in 7.18 seconds; total was 14.22 seconds for 64 output tokens. Other verification
work could be running, so this is an observed session, not an isolated benchmark.
That reply reached its length limit and was correctly marked incomplete. Streaming
reveals progress; it does not remove inference latency or guarantee a finished thought.

Scratch re-export verification now reuses a successful check only when actual
weights, configuration, tokenizer, verifier code, libraries, platform and saved
reports match their recorded hashes. Changed inputs trigger verification again.

## Verification coverage

The 0.5.2 suite passed 44 Python tests on Windows, Linux and macOS. Version 0.5.3
adds acceptance checks for exact resume, frozen-prefix parity, single-writer
locks, actor curriculum, GGUF adapter caching, CPU topology and reply budgets.
Browser client state tests exercise
immediate input, streaming UTF-8 boundaries, retry, draft preservation and actor
roles through a simulated DOM. Actual native HTTP and Transformers runtime checks
are separate from those tests. Version 0.5.2 passed Docker packaging, Codespaces
startup and the release workflow. Each new commit has its own CI run; consult
the branch Actions page for the current result.

## Still unfinished

The [actual compact-model pilot](COMPACT-MODELS.md) reduces 134,515,008 parameters
to 46,012,608 with pretrained layer pruning and recovery. Same-precision Q8 files
shrink from 144,810,944 to 50,679,136 bytes. Native first-text timings improve in
the small local run, but free language degrades severely. Neither small candidate
meets the joint goal; a geometry change, smaller file or lower teacher divergence
alone must not be reported as "a smaller model that knows more and is faster."

The [Phi atlas audit and attention pilots](PHI-ATLAS-EXPERIMENTS.md) now reproduce
the supplied topology and compare six actual training arms, including the exact
48,116,224-parameter architecture. Neither short pilot shows a hybrid advantage.
Reversible context packages recover exactly, but learned phase-consistent
distillation and semantic near-miss improvement remain unproven.

- Compelling, consistently coherent conversation on unseen topics. A lower loss
  and a valid exported file are insufficient. More examples, controlled behavioral
  comparisons and independent review remain necessary.
- Low first-response latency on this laptop. Parallel generation can compete for
  the same two cores; adding models does not create extra compute.
- A genuinely distilled 48M conversational student. Sequence-target pilots exist;
  broad teacher-logit distillation and a viable 48M student do not.
- Rich, learned individual personalities. Four current profiles are brief original
  traits and runtime prompts, not four independently trained celebrity models.
- Long-context conversational consistency beyond the trained 384-token social
  windows, source-family decontamination and broad factual evaluation.
- Speech prosody, overlap, emotional cues and full-duplex participation. Text
  timestamps alone do not train those abilities. AzuCast audio remains separate.
- Production response-pool scheduling, variable participation, audience steering,
  and VR acceptance of this new model. Fixed test order is only a comparison tool.
- Actual Chrome visual and Quest acceptance: computer-use runtime failed to write
  kernel assets here. Simulated DOM tests are not visual acceptance.
- CUDA/MPS hardware execution. CPU tests do not establish GPU behavior.
- AWQ/GPTQ or other unimplemented quantization methods: researched, not validated
  here. Tested native precision paths are Q8/F16 and the mixed-precision adapter.

Resume larger experiments through new versioned runs, retaining unsuccessful
controls. Do not overwrite a previous experiment or silently relabel it as improved.
