import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from vox.frozen_prefix import cache_prefix, cached_forward


def model_fixture():
    torch.manual_seed(739)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, attention_dropout=0.0))
    model.requires_grad_(False)
    model.model.layers[-1].requires_grad_(True)
    model.model.norm.requires_grad_(True)
    return model


def test_cached_prefix_matches_full_causal_logits_and_tail_gradients():
    model = model_fixture()
    ids = torch.tensor([[1, 7, 3, 9, 5, 2]])
    labels = torch.tensor([[-100, -100, -100, 9, 5, 2]])
    model.train()
    full = model(input_ids=ids, labels=labels, use_cache=False)
    full.loss.backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    layers, norm = model.model.layers, model.model.norm
    groups, report = cache_prefix(model, [[(ids, labels)]], 1)
    hidden, kept_labels = groups[0][0]
    assert not hidden.is_inference() and kept_labels is labels
    assert report['targets'] == 1 and report['cached_bytes'] == hidden.numel() * hidden.element_size()
    assert model.model.layers is layers and model.model.norm is norm and model.training
    cached = cached_forward(model, hidden, labels, 1)
    torch.testing.assert_close(cached.logits, full.logits, rtol=1e-6, atol=1e-7)
    cached.loss.backward()
    for key, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, gradients[key], rtol=1e-5, atol=1e-7)
    assert model.model.layers is layers and len(model.model.layers) == 3


def test_failed_cache_restores_modules_and_rejects_stochastic_prefix():
    model = model_fixture()
    layers, norm = model.model.layers, model.model.norm
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(MemoryError):
        cache_prefix(model, [[(ids, ids)]], 1, maximum_bytes=1)
    assert model.model.layers is layers and model.model.norm is norm and model.training
    model.config.attention_dropout = 0.05
    with pytest.raises(ValueError, match='stochastic'):
        cache_prefix(model, [[(ids, ids)]], 1)
