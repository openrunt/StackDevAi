"""Explicit pretrained comparison: optional, independent of the 48M scratch model.

Adapt only the final four decoder blocks. Retain a fixed validation set and export
the best validation checkpoint, including step zero if adaptation makes it worse.
No remote inference, fabricated timings, or silent substitution of this control.
"""

import json
import hashlib
import random
import shutil
import time
from pathlib import Path

from .public_baseline import (
    REFERENCES, conversation_splits, fetch_control, sha256, write_json,
)
from .run_lock import single_writer


def assistant_example(tokenizer, messages, context=256):
    """One first-response target; user/template tokens never contribute to loss."""
    import torch

    offset = 1 if messages[0]["role"] == "system" else 0
    if messages[offset]["role"] != "user" or messages[offset + 1]["role"] != "assistant":
        raise ValueError("Expected a known user/assistant pair")
    prefix = tokenizer.apply_chat_template(messages[:offset + 1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages[:offset + 2], tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat template is not prefix-stable")
    if len(prefix) > context - 16:
        return None
    ids = full[:context]
    labels = [-100] * len(prefix) + ids[len(prefix):]
    return torch.tensor([ids]), torch.tensor([labels])


def loss_on(model, examples, tail_blocks=None):
    import torch

    model.eval()
    total = tokens = 0
    with torch.inference_mode():
        for inputs, labels in examples:
            n = int((labels[:, 1:] != -100).sum())
            if tail_blocks is None:
                output = model(input_ids=inputs, labels=labels, use_cache=False)
            else:
                from .frozen_prefix import cached_forward
                output = cached_forward(model, inputs, labels, tail_blocks)
            total += float(output.loss) * n
            tokens += n
    if not tokens:
        raise ValueError("No assistant targets in evaluation")
    return {"assistant_token_loss": total / tokens, "target_tokens": tokens,
            "conversations": len(examples)}


def check_example_separation(training, validation, testing):
    def identities(examples):
        return {hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest() for x, y in examples}
    groups = [identities(e) for e in (training, validation, testing)]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError("Identical control response targets cross adaptation partitions")
    return {"train_unique": len(groups[0]), "val_unique": len(groups[1]), "test_unique": len(groups[2]), "overlap": 0}


def control_examples(tokenizer, rows, limit, context, target_mode="first"):
    """Keep legacy controls reproducible; new pilots may select complete all-reply targets."""
    from .response_targets import rolling_assistant_examples
    if target_mode not in ("first", "all-turns"):
        raise ValueError("Choose first or all-turns targets")
    result = []
    for row in rows:
        if target_mode == "all-turns":
            result.extend(rolling_assistant_examples(tokenizer, row["messages"], context))
        else:
            item = assistant_example(tokenizer, row["messages"], context)
            if item is not None:
                result.append(item)
        if len(result) >= limit:
            break
    return result[:limit]


def atomic_torch_save(torch, value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@single_writer
def adapt_control(project, cache, output, steps=60, context=256, reference="smollm135", blocks=4, eval_samples=24, retain_candidate=False,
                  target_mode="first", train_samples=1000, resume=False, prefix_cache=False):
    from .limits import limit_cpu, memory_guard
    torch = limit_cpu(2)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import __version__ as transformers_version

    project, output = Path(project), Path(output)
    if output.exists() and not resume:
        raise FileExistsError("Use a new control run directory")
    if resume and not (output / "run-config.json").is_file():
        raise ValueError("No resumable control configuration; preserve older controls and use a new run")
    output.mkdir(parents=True, exist_ok=True)
    if not 1 <= blocks <= 4 or context not in (128, 256, 384, 512) or not 8 <= eval_samples <= 64 or not 1 <= steps <= 10000:
        raise ValueError("Use 1–4 adapted blocks, 128/256/384/512 context, 8–64 eval samples and 1–10,000 updates")
    if not 8 <= train_samples <= 10000 or target_mode not in ("first", "all-turns"):
        raise ValueError("Use 8–10,000 training targets and first or all-turns mode")
    info = REFERENCES[reference]
    base = fetch_control(cache, reference)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.float32, trust_remote_code=False, use_safetensors=True)
    model.config.use_cache = False
    torch.manual_seed(739)
    splits = conversation_splits(project)

    training = control_examples(tokenizer, splits["train"], train_samples, context, target_mode)
    validation = control_examples(tokenizer, splits["val"], eval_samples, context, target_mode)
    testing = control_examples(tokenizer, splits["test"], eval_samples, context, target_mode)
    separation = check_example_separation(training, validation, testing)
    if min(len(training), len(validation), len(testing)) < 8:
        raise ValueError("Not enough usable assistant targets in each source partition")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.model.layers[-blocks:]:
        layer.requires_grad_(True)
    model.model.norm.requires_grad_(True)
    trainable = {k: p for k, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.00002, weight_decay=0.01)
    base_hash = sha256(base / "model.safetensors")
    # Prepared scratch corpora and standalone social snapshots use different manifests.
    identity_file = project / "data/corpus_report.json"
    if not identity_file.exists():
        identity_file = project / "conversations.jsonl"
    dataset_hash = sha256(identity_file)
    run_config = {"protocol": "hf-control-resume-v1", "base_sha256": base_hash, "dataset_sha256": dataset_hash,
                  "trainer_sha256": sha256(Path(__file__)), "torch_version": str(torch.__version__),
                  "transformers_version": transformers_version, "base_config_sha256": sha256(base / "config.json"),
                  "prefix_cache": prefix_cache,
                  "cache_code_sha256": sha256(Path(__file__).with_name("frozen_prefix.py")) if prefix_cache else None,
                  "context": context, "blocks": blocks, "eval_samples": eval_samples, "target_mode": target_mode,
                  "train_samples": train_samples, "reference": reference, "learning_rate": 0.00002,
                  "targets_sha256": hashlib.sha256(b"".join(x.numpy().tobytes() + y.numpy().tobytes()
                      for group in (training, validation, testing) for x, y in group)).hexdigest()}
    if resume and json.loads((output / "run-config.json").read_text(encoding="utf-8")) != run_config:
        raise ValueError("Resume configuration, weights or data changed; use a new run")
    if resume:
        (output / "stop-request").unlink(missing_ok=True)
    if not resume:
        write_json(output / "run-config.json", run_config)
        shutil.copy2(Path(__file__), output / "TRAINER-SNAPSHOT.py")
    cache_info, tail_blocks = None, None
    if prefix_cache:
        from .frozen_prefix import cache_prefix
        if not resume:
            shutil.copy2(Path(__file__).with_name("frozen_prefix.py"), output / "FROZEN-PREFIX-SNAPSHOT.py")
        def cache_progress(count, size):
            if count == 1 or count % 16 == 0:
                print(json.dumps({"phase": "preparing_frozen_prefix", "targets": count, "cached_bytes": size,
                                  "rss_bytes": memory_guard(4)}), flush=True)
        (training, validation, testing), cache_info = cache_prefix(model, [training, validation, testing], blocks, progress=cache_progress)
        tail_blocks = blocks
        print(json.dumps({"phase": "frozen_prefix_ready", **cache_info}), flush=True)
    before = json.loads((output / "before-validation.json").read_text(encoding="utf-8")) if resume else loss_on(model, validation, tail_blocks)
    if not resume:
        write_json(output / "before-validation.json", before)
    best_loss, best_step = before["assistant_token_loss"], 0
    best_weights = {k: p.detach().clone() for k, p in trainable.items()}
    history = [{"step": 0, "validation": before}]
    processed = targets = start_step = 0
    previous_seconds = 0.0
    if resume:
        state = torch.load(output / "adaptation-checkpoint.pt", map_location="cpu", weights_only=True)
        if state.get("run_config") != run_config:
            raise ValueError("Checkpoint identity does not match this run")
        start_step = state["step"]
        if steps < start_step:
            raise ValueError("Resume target must be at least the completed update count")
        with torch.no_grad():
            for key, parameter in trainable.items():
                parameter.copy_(state["weights"][key])
        optimizer.load_state_dict(state["optimizer"])
        history, processed, targets = state["history"], state["processed"], state["targets"]
        previous_seconds = history[-1].get("elapsed_seconds", 0.0)
        torch.set_rng_state(state["torch_rng_state"])
        # One atomic checkpoint is authoritative even if interrupted between two file writes.
        best_weights, best_step, best_loss = state["best_weights"], state["best_step"], state["best_loss"]
        del state
    else:
        atomic_torch_save(torch, {"weights": best_weights, "selected_update": best_step, "validation_loss": best_loss,
                                 "base_sha256": base_hash, "dataset_sha256": dataset_hash, "run_config": run_config}, output / "selected-adapter.pt")
        atomic_torch_save(torch, {"step": 0, "weights": best_weights, "optimizer": optimizer.state_dict(),
                                 "base_sha256": base_hash, "dataset_sha256": dataset_hash, "run_config": run_config,
                                 "history": history, "processed": 0, "targets": 0,
                                 "best_weights": best_weights, "best_step": best_step, "best_loss": best_loss,
                                 "torch_rng_state": torch.get_rng_state()}, output / "adaptation-checkpoint.pt")
    started = time.perf_counter()
    completed_step, last_loss = start_step, None

    def record_update(completed, training_loss):
        nonlocal best_loss, best_step, best_weights
        metrics = loss_on(model, validation, tail_blocks)
        if metrics["assistant_token_loss"] < best_loss:
            best_loss, best_step = metrics["assistant_token_loss"], completed
            best_weights = {k: p.detach().clone() for k, p in trainable.items()}
            atomic_torch_save(torch, {"weights": best_weights, "selected_update": best_step, "validation_loss": best_loss,
                                     "base_sha256": base_hash, "dataset_sha256": dataset_hash, "run_config": run_config}, output / "selected-adapter.pt")
        row = {"step": completed, "train_loss": training_loss, "validation": metrics,
               "elapsed_seconds": round(previous_seconds + time.perf_counter() - started, 3), "rss_bytes": memory_guard(4)}
        history.append(row)
        write_json(output / "progress.json", row)
        print(json.dumps(row), flush=True)
        atomic_torch_save(torch, {"step": completed, "weights": {k: p.detach().clone() for k, p in trainable.items()},
                    "optimizer": optimizer.state_dict(), "dataset_sha256": dataset_hash,
                    "base_sha256": base_hash, "run_config": run_config, "history": history,
                    "best_weights": best_weights, "best_step": best_step, "best_loss": best_loss,
                    "processed": processed, "targets": targets, "torch_rng_state": torch.get_rng_state()}, output / "adaptation-checkpoint.pt")

    for step in range(start_step, steps):
        if (output / "stop-request").exists():
            break
        model.train()
        x, y = training[random.Random(739 * 1000003 + step).randrange(len(training))]
        optimizer.zero_grad(set_to_none=True)
        if tail_blocks is None:
            loss = model(input_ids=x, labels=y, use_cache=False).loss
        else:
            from .frozen_prefix import cached_forward
            loss = cached_forward(model, x, y, tail_blocks).loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite training loss; last complete checkpoint preserved")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        optimizer.step()
        processed += int(x.shape[0] * x.shape[1])
        targets += int((y[:, 1:] != -100).sum())
        completed_step, last_loss = step + 1, float(loss.detach())
        if (step + 1) % 10 == 0 or step == 0 or step + 1 == steps:
            record_update(completed_step, last_loss)
    if completed_step > history[-1]["step"]:
        record_update(completed_step, last_loss)
    with torch.no_grad():
        for key, parameter in trainable.items():
            parameter.copy_(best_weights[key])
    atomic_torch_save(torch, {"weights": best_weights, "selected_update": best_step, "validation_loss": best_loss,
                             "base_sha256": base_hash, "dataset_sha256": dataset_hash, "run_config": run_config}, output / "selected-adapter.pt")
    model.eval()
    final_validation = loss_on(model, validation, tail_blocks)
    final_test = loss_on(model, testing, tail_blocks)  # Used only after validation selection.
    exported = output / "export"
    model.config.use_cache = True
    model.save_pretrained(exported, safe_serialization=True)
    tokenizer.save_pretrained(exported)
    for name in ("vocab.json", "merges.txt"):
        if (base / name).exists():
            shutil.copy2(base / name, exported / name)
    report = {
        "name": "Xputty Public Conversation Control · " + info["folder"], "initialization": "pretrained, not the 48M scratch model",
        "reference": reference, "base_model": info["model"], "base_revision": info["revision"], "base_weights_sha256": base_hash,
        "weights_sha256": sha256(exported / "model.safetensors"), "license": "Apache-2.0",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable.values()), "adapted_blocks": blocks,
        "representation": "hf-instruction-control-v1", "instruction_tuned": True,
        "requested_updates": steps, "completed_updates": history[-1]["step"], "selected_update": best_step,
        "selection_rule": "Lowest fixed validation assistant-token loss, including the original model at update zero",
        "dataset_sha256": dataset_hash, "source_manifest": "public-sources.json",
        "dataset_identity_file": identity_file.relative_to(project).as_posix(),
        "training_conversations": len(training) if target_mode == "first" else None,
        "available_source_conversations": {k: len(v) for k, v in splits.items()},
        "input_tokens_processed": processed, "assistant_targets_processed": targets,
        "context_tokens": context, "before_validation": before, "after_validation": final_validation,
        "target_mode": target_mode, "training_targets": len(training), "resumed_from_update": start_step,
        "frozen_prefix_cache": cache_info,
        "target_protocol": "complete-assistant-rolling-v1" if target_mode == "all-turns" else "legacy-first-response-v1",
        "example_separation": separation,
        "final_test": final_test, "history": history, "timing": "No speech audio or timing learned",
        "heldout_scope": "Held out of this adaptation. Pretraining contamination is not ruled out; SmolLM2 was already tuned on this upstream dataset.",
        "verified_hf_transformers": False, "hf_export_check": "Export saved; separate reload/generation check required", "verified_ollama": False,
    }
    corpus = json.loads((project / "public-sources.json").read_text(encoding="utf-8"))
    if corpus.get("characters"):
        from .podcast_distill import RULES
        report.update(name="Xputty Podcast Pilot · " + info["folder"], persona_profiles=corpus["characters"],
                      persona_rules=corpus.get("runtime_persona_rules", RULES), method=corpus.get("method"),
                      heldout_scope="Source conversations remain in their supplied adaptation partitions; related-topic and upstream pretraining overlap are not ruled out.")
    write_json(output / "training_report.json", report)
    write_json(exported / "export_report.json", report)
    for name in ("public-sources.json", "UPSTREAM-DATASET-CARD.md", "SODA-DATASET-CARD.md", "CHARACTER-SOURCE-MANIFEST.json"):
        if (project / name).exists():
            shutil.copy2(project / name, exported / name)
    shutil.copy2(base / "README.md", exported / "UPSTREAM-MODEL-CARD.md")
    shutil.copy2(base / "LICENSE", exported / "LICENSE")
    shutil.copy2(Path(__file__).resolve().parents[1] / "LICENSE", exported / "APPLICATION-LICENSE.txt")
    shutil.copy2(Path(__file__).resolve().parents[1] / "export_chat.py", exported / "chat.py")
    (exported / "Modelfile").write_text('FROM .\nPARAMETER num_ctx 2048\nPARAMETER temperature 0.6\n', encoding="utf-8")
    (exported / "README.md").write_text(
        f"# {report['name']}\n\nA distinct {report['parameter_count']:,}-parameter pretrained comparison model.\n"
        f"Base: {info['model']}@{info['revision']}. Final {blocks} blocks adapted on the public conversation training split.\n"
        f"Selected update: {best_step}/{steps}; selection used validation only.\n"
        "This is NOT a scratch-trained 48M model and is never substituted for it.\n"
        "Base model is Apache-2.0. Dataset license and attribution are recorded separately in public-sources.json and original cards.\n"
        "Pretraining may already have seen these public conversations. This evaluation measures adaptation retention, not novel generalization.\n"
        "The included chat application retains AGPL-3.0; see APPLICATION-LICENSE.txt.\n"
        "Run `python chat.py` for local browser chat using only these exported weights.\n"
        "No voice, audio timing, animation or real-time podcast participation is trained here.\n",
        encoding="utf-8",
    )
    if retain_candidate:
        export_candidate(output, cache)
    return report


def export_candidate(output, cache):
    """Retain the actual last adapted weights even when validation selects update 0."""
    from .limits import limit_cpu
    torch = limit_cpu(2)
    from transformers import AutoModelForCausalLM

    output = Path(output)
    report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    state = torch.load(output / "adaptation-checkpoint.pt", map_location="cpu", weights_only=True)
    base = fetch_control(cache, report.get("reference", "smollm135"))
    if state["base_sha256"] != sha256(base / "model.safetensors") or state["dataset_sha256"] != report["dataset_sha256"]:
        raise ValueError("Candidate source weights/corpus do not match the comparison run")
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.float32, trust_remote_code=False, use_safetensors=True)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["weights"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError("Candidate parameter mismatch: " + name)
            parameters[name].copy_(value)
    candidate = output / "candidate"
    model.save_pretrained(candidate, safe_serialization=True)
    for source in (output / "export").iterdir():
        if source.is_file() and source.name not in ("model.safetensors", "config.json", "export_report.json", "README.md"):
            shutil.copy2(source, candidate / source.name)
    report.update(name=f"Xputty Public Conversation · {report['parameter_count'] / 1e6:.1f}M adapted candidate",
                  reference_selected_update=report["selected_update"],
                  after_validation=report["history"][-1]["validation"], final_test=None,
                  selection_rule="Final adaptation checkpoint, retained separately from validation selection",
                  selected_update=state["step"], selection_status="Unselected final adaptation candidate; compare with the validation-selected reference",
                  weights_sha256=sha256(candidate / "model.safetensors"))
    write_json(candidate / "export_report.json", report)
    (candidate / "README.md").write_text(
        f"# {report['name']}\n\nPretrained {report['base_model']}@{report['base_revision']}, with {state['step']} actual public-data adaptation updates.\n"
        "This candidate did not beat the validation-selected reference; preserve that distinction when comparing it.\n"
        "It is not the scratch-trained 48M model. Upstream weights/data are declared Apache-2.0; cards, license and provenance accompany this export.\n"
        "Run `python chat.py` for local chat using these actual adapted weights. No remote fallback.\n",
        encoding="utf-8",
    )
    return candidate
