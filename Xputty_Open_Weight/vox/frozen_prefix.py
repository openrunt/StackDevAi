"""Optional exact-precision activation reuse for a frozen LLaMA/Qwen decoder prefix."""
from contextlib import contextmanager
import time


@contextmanager
def tail_only(model, blocks):
    layers = model.model.layers
    model.model.layers = layers[-blocks:]
    try:
        yield
    finally:
        model.model.layers = layers


def cached_forward(model, hidden, labels, blocks):
    with tail_only(model, blocks):
        return model(inputs_embeds=hidden, labels=labels, use_cache=False)


def cache_prefix(model, groups, blocks, maximum_bytes=512 * 1024 * 1024, progress=None):
    """Cache causal, pre-final-norm hidden states; labels and token positions stay intact.

    This only applies when the prefix is completely frozen and has no dropout.
    Caches remain ordinary CPU tensors so autograd can train the selected tail.
    """
    import torch
    if model.config.model_type not in ("llama", "qwen2"):
        raise ValueError("Frozen-prefix caching currently supports LLaMA and Qwen2 decoders")
    layers, norm = model.model.layers, model.model.norm
    if not 1 <= blocks < len(layers):
        raise ValueError("Leave a nonempty frozen prefix and trainable tail")
    prefix = layers[:-blocks]
    if any(p.requires_grad for layer in prefix for p in layer.parameters()) or any(p.requires_grad for p in model.model.embed_tokens.parameters()):
        raise ValueError("Embedding and all prefix weights must be frozen")
    if getattr(model.config, 'attention_dropout', 0.0) or any(isinstance(m, torch.nn.Dropout) and m.p for layer in prefix for m in layer.modules()):
        raise ValueError("A stochastic prefix cannot be cached equivalently; disable caching for dropout models")
    began, size, count, prepared = time.perf_counter(), 0, 0, []
    was_training = model.training
    try:
        model.eval()
        model.model.layers = prefix
        model.model.norm = torch.nn.Identity()
        with torch.no_grad():
            for group in groups:
                cached = []
                for ids, labels in group:
                    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state.detach().clone()
                    size += hidden.numel() * hidden.element_size()
                    if size > maximum_bytes:
                        raise MemoryError('Frozen-prefix activation cache exceeds its byte limit')
                    cached.append((hidden, labels))
                    count += 1
                    if progress:
                        progress(count, size)
                prepared.append(cached)
    finally:
        model.model.layers = layers
        model.model.norm = norm
        model.train(was_training)
    return prepared, {"cached_bytes": size, "targets": count, "seconds": round(time.perf_counter() - began, 3),
                      "precision": "unchanged model floating-point dtype; no activation quantization"}
