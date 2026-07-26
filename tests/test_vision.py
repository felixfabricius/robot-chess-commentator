from chess_assistant.vision import SquareEstimate, BoardEstimate, BoardEstimator
from chess_assistant.model.model import SquareClassifierMultiHead
from chess_assistant.model.config import TARGET_MAP
from omegaconf import OmegaConf
from pathlib import Path

import math
import torch
import pytest

CALIBRATION_METADATA_PATH = Path("data/generated/2026-07-01_175334/calibration_metadata.json")
SQUARE_IMAGE_PATH = Path("data/generated/2026-07-01_175334/board_2026-07-01_175602/squares/a1/a1.png")
SQUARES_DIR = SQUARE_IMAGE_PATH.parent.parent
CONFIG_PATH = Path("config.yaml")


### Structural tests: an untrained multi-head model is enough to exercise the plumbing.
@pytest.fixture(scope="module")
def board_estimator():
    return BoardEstimator(
        model_type="CNN",
        calibration_metadata_path=CALIBRATION_METADATA_PATH,
        model=SquareClassifierMultiHead(),
    )

def test_estimate_square(board_estimator):
    square_estimate = board_estimator.estimate_square(SQUARE_IMAGE_PATH)
    assert isinstance(square_estimate.empty, float)
    assert isinstance(square_estimate, SquareEstimate)

def test_estimate_board(board_estimator):
    board_estimate = board_estimator.estimate_board(SQUARES_DIR)
    assert isinstance(board_estimate.a1, SquareEstimate)
    assert isinstance(board_estimate, BoardEstimate)

def test_estimate_square_returns_valid_logprobs(board_estimator):
    square_estimate = board_estimator.estimate_square(SQUARE_IMAGE_PATH)
    # The 13 stored values are reconstructed log-probabilities; exp() forms a valid
    # distribution and softmax over them (as game.py re-applies) sums to ~1.
    values = torch.tensor([getattr(square_estimate, label) for label in TARGET_MAP])
    assert math.isclose(values.exp().sum().item(), 1.0, abs_tol=1e-4)
    assert math.isclose(torch.softmax(values, dim=-1).sum().item(), 1.0, abs_tol=1e-6)


### The production path: construct exactly as main.py does, from config.yaml.
### Pins the safetensors import, the config key, the kwarg name, and that the shipped
### weights still match the architecture. Any of those breaking is a crash on launch.
@pytest.fixture(scope="module")
def trained_board_estimator():
    config = OmegaConf.load(CONFIG_PATH)
    return BoardEstimator(
        "CNN",
        config,
        calibration_metadata_path=CALIBRATION_METADATA_PATH,
        model_weights_path=Path(config.vision.model_weights_path),
        device="cpu",
    )

def test_loads_shipped_weights_from_config(trained_board_estimator):
    assert isinstance(trained_board_estimator.model, SquareClassifierMultiHead)

def test_model_is_in_eval_mode(trained_board_estimator):
    # In train mode BatchNorm normalises each crop by its own batch-of-one statistics rather
    # than the running stats learned during training, which costs ~8-10pp of square accuracy.
    assert trained_board_estimator.model.training is False

def test_inference_does_not_mutate_batchnorm_running_stats(trained_board_estimator):
    # The behavioural signature of a train-mode model: every forward pass nudges
    # running_mean/running_var, so the loaded checkpoint's stats drift away from the trained
    # ones as a game is played. Checking the .training flag alone would not catch a model
    # that is re-entered into train mode further downstream.
    batchnorms = [
        module for module in trained_board_estimator.model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]
    assert batchnorms, "model has no BatchNorm2d; this test is guarding nothing"
    before = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in batchnorms]

    trained_board_estimator.estimate_square(SQUARE_IMAGE_PATH)

    for bn, (running_mean, running_var) in zip(batchnorms, before):
        assert torch.equal(bn.running_mean, running_mean)
        assert torch.equal(bn.running_var, running_var)

def test_inference_is_deterministic(trained_board_estimator):
    # Guards the inference transform: TRAIN_TRANSFORM carries ColorJitter / GaussianNoise /
    # RandomAffine, so wiring it in here instead of EVAL_TRANSFORM would make the robot read a
    # different board from the same photo. (Note this does *not* guard eval mode -- a
    # train-mode BatchNorm is still deterministic for a fixed input.)
    first = trained_board_estimator.estimate_square(SQUARE_IMAGE_PATH)
    second = trained_board_estimator.estimate_square(SQUARE_IMAGE_PATH)
    for label in TARGET_MAP:
        assert getattr(first, label) == getattr(second, label)


### Bayesian prior correction plumbing
def _prior_correction_estimator(tmp_path, prior_correction, log_prior):
    """A CNN BoardEstimator over a minimal calibration file, with a known log_prior on the model."""
    import json

    metadata_path = tmp_path / "calibration_metadata.json"
    metadata_path.write_text(
        json.dumps({"camera_natural_orientation": {"order": {"tl": "a8"}}}), encoding="utf-8"
    )
    model = SquareClassifierMultiHead(log_prior=log_prior)
    return BoardEstimator(
        model_type="CNN",
        model=model,
        calibration_metadata_path=metadata_path,
        prior_correction=prior_correction,
    )

def test_board_estimator_applies_prior_by_default(tmp_path):
    """Live inference should always use the correction when a prior is available, so the default
    is on -- the robot has no reason to deliberately use the worse decision rule.
    """
    log_prior = torch.full((13,), -2.0)
    estimator = _prior_correction_estimator(tmp_path, prior_correction=True, log_prior=log_prior)
    assert estimator.log_prior is not None
    assert torch.allclose(estimator.log_prior, log_prior)

def test_board_estimator_prior_can_be_switched_off(tmp_path):
    """Eval passes prior_correction explicitly so a checkpoint can be scored both ways."""
    estimator = _prior_correction_estimator(
        tmp_path, prior_correction=False, log_prior=torch.full((13,), -2.0)
    )
    assert estimator.log_prior is None

def test_board_estimator_default_zero_prior_is_harmless(tmp_path):
    """A model with no known prior (all-zeros buffer, e.g. legacy weights) still resolves to a
    tensor, but subtracting it is a no-op -- so "always on" is safe for old checkpoints.
    """
    estimator = _prior_correction_estimator(tmp_path, prior_correction=True, log_prior=None)
    assert torch.allclose(estimator.log_prior, torch.zeros(13))
