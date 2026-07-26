"""Smoke tests for the single-head training loop (train_single_head), used by the model-choice
ablation to train model 2. Needs the generated dataset (data/generated/data.csv) on disk, like
test_model.py, since it runs real batches off the training dataloader.
"""

import copy

import pytest
import torch
from torch import nn

from chess_assistant.model.data import create_dataloader
from chess_assistant.model.model import SquareClassifier2
from chess_assistant.model.train import train_single_head

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def dataloader():
    return create_dataloader("train", shuffle=True, batch_size=64)


def test_train_single_head_returns_expected_metric_keys(dataloader):
    # debug=True stops after a handful of batches; the reported loss is then zero, but the metric
    # keys (shared with the multi-head train() so W&B panels overlay) must still be present.
    model = SquareClassifier2().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_single_head(
        model=model,
        dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        debug=True,
        device=DEVICE,
    )
    assert set(metrics) == {"train/total/recent_loss", "train/total/n_recent_loss"}
    assert metrics["train/total/recent_loss"] == 0  # debug -> nothing accumulated


def test_train_single_head_updates_parameters_without_nan(dataloader):
    # A few real steps run (backward + optimizer.step happen before the debug early-break), so a
    # tracked parameter must move and stay finite -- i.e. gradients flowed and the loss never NaN'd.
    model = SquareClassifier2().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tracked = next(p for p in model.parameters() if p.requires_grad)
    before = copy.deepcopy(tracked.detach())

    train_single_head(
        model=model,
        dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        debug=True,
        device=DEVICE,
    )

    assert torch.isfinite(tracked).all()
    assert not torch.equal(tracked.detach(), before)
