"""Shape/dtype smoke tests for the model forward passes, on a real batch off the training
dataloader. Needs the generated dataset (data/generated/data.csv) on disk.
"""

import pytest
import torch
import time
from chess_assistant.model.data import create_dataloader
from chess_assistant.model.model import (
    SquareClassifier,
    SquareClassifier2,
    SquareClassifierMultiHead,
)


@pytest.fixture(scope="module")
def dataloader():
    return create_dataloader("train", shuffle=True, batch_size=64)

@pytest.fixture(scope="module")
def model():
    return SquareClassifier()

def test_forward_pass(dataloader, model):
    # Model 1: a single 13-way logit vector per square.
    start = time.perf_counter()
    batch = next(iter(dataloader))
    output = model(batch[0], batch[1])
    assert output.shape == (64, 13)
    assert output.dtype == torch.float32
    end = time.perf_counter()
    print(f"Time taken for batched forward pass with 64 squares: {end - start:.6f}")

@pytest.fixture(scope="module")
def model2():
    return SquareClassifier2()

def test_forward_pass_model2(dataloader, model2):
    # Model 2: like model 1, a single 13-way logit vector per square (the single-head path the
    # model-choice ablation trains).
    batch = next(iter(dataloader))
    output = model2(batch[0], batch[1])
    assert output.shape == (64, 13)
    assert output.dtype == torch.float32

@pytest.fixture(scope="module")
def multihead_model():
    return SquareClassifierMultiHead()

def test_forward_pass_multihead(dataloader, multihead_model):
    # Model 3: the three factored heads, one empty logit + 2 color + 6 type.
    batch = next(iter(dataloader))
    logit_empty, logits_color, logits_type = multihead_model(batch[0], batch[1])
    assert logit_empty.shape == (64,)
    assert logits_color.shape == (64, 2)
    assert logits_type.shape == (64, 6)
    assert logit_empty.dtype == torch.float32
    assert logits_color.dtype == torch.float32
    assert logits_type.dtype == torch.float32


### log_prior buffer (Bayesian prior correction)
# Both the single-head model 2 and the multi-head model 3 carry the buffer (model 2 gained it for
# the model-choice ablation, so it can be prior-corrected identically), so these are parametrized
# over both. They need no dataset, so they stay independent of the dataloader fixtures above.
BUFFER_MODELS = [SquareClassifier2, SquareClassifierMultiHead]

@pytest.mark.parametrize("model_cls", BUFFER_MODELS)
def test_log_prior_buffer_always_registered_and_defaults_to_zeros(model_cls):
    """Registered unconditionally so state_dict keys never depend on construction arguments.
    All-zeros is the safe default: subtracting it changes nothing.
    """
    model = model_cls()
    assert model.log_prior.shape == (13,)
    assert torch.allclose(model.log_prior, torch.zeros(13))
    assert "log_prior" in model.state_dict()

@pytest.mark.parametrize("model_cls", BUFFER_MODELS)
def test_log_prior_is_a_buffer_not_a_parameter(model_cls):
    model = model_cls(log_prior=torch.full((13,), -2.0))
    names = dict(model.named_buffers())
    assert "log_prior" in names
    assert "log_prior" not in dict(model.named_parameters())
    # Must not attract gradients / weight decay.
    assert not model.log_prior.requires_grad

@pytest.mark.parametrize("model_cls", BUFFER_MODELS)
def test_log_prior_stored_as_float32_copy(model_cls):
    """A copy, not an alias: mutating the caller's tensor afterwards must not change the model."""
    source = torch.full((13,), -3.0, dtype=torch.float64)
    model = model_cls(log_prior=source)
    assert model.log_prior.dtype == torch.float32
    source[0] = 999.0
    assert model.log_prior[0].item() == -3.0

@pytest.mark.parametrize("model_cls", BUFFER_MODELS)
def test_log_prior_survives_state_dict_roundtrip(model_cls):
    log_prior = torch.linspace(-5.0, -1.0, 13)
    saved = model_cls(log_prior=log_prior).state_dict()
    restored = model_cls()
    restored.load_state_dict(saved)
    assert torch.allclose(restored.log_prior, log_prior, atol=1e-6)

@pytest.mark.parametrize("model_cls", BUFFER_MODELS)
def test_legacy_state_dict_without_log_prior_loads_non_strictly(model_cls):
    """Mirrors vision.py: weights saved before the buffer existed must still load, leaving the
    prior at all-zeros (no correction), and `log_prior` must be the ONLY tolerated missing key.
    """
    legacy = model_cls(log_prior=torch.full((13,), -2.0)).state_dict()
    del legacy["log_prior"]

    model = model_cls()
    missing, unexpected = model.load_state_dict(legacy, strict=False)
    assert set(missing) == {"log_prior"}
    assert not unexpected
    assert torch.allclose(model.log_prior, torch.zeros(13))

@pytest.mark.parametrize("model_cls", BUFFER_MODELS)
def test_legacy_state_dict_loads_strictly_only_when_buffer_absent(model_cls):
    """A strict load of a legacy checkpoint fails loudly -- the reason vision.py uses strict=False
    with an explicit allow-list rather than silently ignoring mismatches.
    """
    legacy = model_cls().state_dict()
    del legacy["log_prior"]
    with pytest.raises(RuntimeError, match="log_prior"):
        model_cls().load_state_dict(legacy)
