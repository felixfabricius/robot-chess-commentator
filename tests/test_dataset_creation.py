"""Tests for squareDataset / create_dataloader: the shapes, dtypes and decomposed targets the
training loop relies on, and the guarantee that the augmentation pipeline leaves the mask channel
alone. Needs the generated dataset (data/generated/data.csv) on disk.
"""

from pathlib import Path

import pytest
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors

from chess_assistant.model.data import create_dataloader, squareDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM
from chess_assistant.model.config import INVERSE_COLOR_MAP, INVERSE_TYPE_MAP

# The generated dataset is gitignored, so skip rather than fail wherever it is absent (CI, a
# fresh clone).
pytestmark = pytest.mark.skipif(
    not Path("data/generated/data.csv").exists(),
    reason="needs the generated dataset (data/generated/data.csv), which is not in git",
)

@pytest.fixture
def dataset(scope="module"):
    return squareDataset()

def test_dataset_construction(dataset):
    assert isinstance(len(dataset), int) and len(dataset) > 0

def test_dataset_getitem(dataset):
    # dataset[0] is square a1, which holds a white rook ("R").
    _, _, is_piece, color_target, type_target = dataset[0]
    assert is_piece == 1.0
    assert INVERSE_COLOR_MAP[color_target] == "white"
    assert INVERSE_TYPE_MAP[type_target] == "R"

def test_metadata_shape(dataset):
    assert dataset[0][1].shape == (4,) # one-hot of which board corner is top-left in the image

### Test transformations
def test_transform(dataset):
    image = dataset[0][0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (4, 144, 144)
    assert image[3].max() == 1.0 # ensure mask channel is normalised correctly

@pytest.mark.parametrize(
    "mask",
    [
        torch.randn(10, 10),
        torch.randn(100, 100),
        1000 * torch.randn(200, 200)
    ]
)
def test_mask_transform(mask):
    # The photometric augmentations must leave a tv_tensors.Mask untouched: put the same mask
    # through the train and the eval pipeline and the pixel values should still agree.
    mask = tv_tensors.Mask(mask)
    modified_train_transform = v2.Compose([transform for transform in TRAIN_TRANSFORM.transforms[:-1]])
        # Drops the last transform of TRAIN_TRANSFORM, RandomAffine - the one geometric transform,
        # and so the only one that IS meant to move the mask.
        # If the order of the transforms changes, this test has to be adjusted.
    train_transformed_mask = modified_train_transform(mask)
    eval_transformed_mask = EVAL_TRANSFORM(mask)
    assert torch.allclose(train_transformed_mask, eval_transformed_mask, atol=1e-7)

def test_target_transform(dataset):
    # a1 is a (non-empty) white rook, so all three decomposed targets are populated.
    _, _, is_piece, color_target, type_target = dataset[0]
    assert isinstance(is_piece, float)
    assert is_piece == 1.0
    assert color_target in [0, 1]
    assert type_target in list(range(6))

### Test dataloader
@pytest.mark.parametrize("batch_size", [1, 4, 10, 64])
def test_dataloader(batch_size):
    dataloader = create_dataloader("train", batch_size=batch_size)
    batch = next(iter(dataloader))
    # (image, metadata, is_piece, color_target, type_target)
    assert len(batch) == 5
    assert batch[0].shape == (batch_size, 4, 144, 144)
    assert batch[1].shape == (batch_size, 4)
    assert batch[2].shape == (batch_size,)
    assert batch[3].shape == (batch_size,)
    assert batch[4].shape == (batch_size,)

### Test datatypes
def test_datatypes():
    dataloader = create_dataloader(split="train", batch_size=64)
    batch = next(iter(dataloader))
    assert batch[0].dtype == torch.float32
    assert batch[1].dtype == torch.float32
    # is_piece is a python float -> default collate -> float64; color/type are long targets
    assert batch[2].dtype == torch.float64
    assert batch[3].dtype == torch.long
    assert batch[4].dtype == torch.long