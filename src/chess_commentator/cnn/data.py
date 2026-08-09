"""Dataset and dataloader for the square classifier: one row of data/generated/data.csv is one
square crop, and the images themselves are the 4-channel (RGB + mask) arrays written next to each
crop by the data-generation pipeline.
"""

import polars as pl
import numpy as np
import json
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from pathlib import Path, PureWindowsPath

from chess_commentator.labels import decompose_label, TOP_LEFT_OHE_MAP

TRAIN_TRANSFORM = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.15, hue=0.03),
    v2.GaussianBlur(kernel_size=5, sigma=(1e-3, 1)), # kernel_size 5, so this is wider than the conv kernels
    v2.GaussianNoise(mean=0.0, sigma=0.02),
    v2.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.93, 1.07))
])

EVAL_TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def subsample_train_setups(
    data: pl.DataFrame, target_rows: int, seed: int
) -> tuple[pl.DataFrame, dict]:
    """Keep whole setups, in a seeded random order, until the row count lands closest to
    ``target_rows``. Returns the filtered frame plus a dict describing what was kept.

    Whole setups are dropped rather than individual squares, for the same reason the splits are by
    setup: the 64 squares of one capture share a lighting, a board and a camera pose, so dropping
    individual rows would leave a training set that is smaller but no less redundant, and the
    data-efficiency ablation would overstate how much genuinely new information each row buys.

    The kept setups are a *prefix* of one seeded permutation, and prefix sums only increase, so a
    smaller target always yields a SUBSET of a larger one: the 5k run's setups are nested inside
    the 10k run's. That is what makes the points of the ablation comparable to each other.
    """
    # Sort before shuffling: polars gives no ordering guarantee for unique(), and the selection
    # has to be reproducible from (seed, target_rows) alone.
    setup_ids = sorted(data["setup_id"].unique().to_list())
    random.Random(seed).shuffle(setup_ids)

    rows_by_setup = {
        row["setup_id"]: row["count"]
        for row in data["setup_id"].value_counts().iter_rows(named=True)
    }

    # Walk the shuffled order and keep the prefix whose total lands closest to the target -- the
    # setup that crosses the target may overshoot by less than stopping short of it would undershoot.
    # Once the running count reaches the target the distance can only grow, so stop there.
    # best_distance starts at the distance of the empty prefix (|0 - target_rows|).
    best_k, best_distance, cumulative = 0, target_rows, 0
    for k, setup_id in enumerate(setup_ids, start=1):
        cumulative += rows_by_setup[setup_id]
        if abs(cumulative - target_rows) < best_distance:
            best_k, best_distance = k, abs(cumulative - target_rows)
        if cumulative >= target_rows:
            break

    # max(..., 1): a target smaller than the smallest setup would otherwise select nothing.
    kept_setup_ids = setup_ids[: max(best_k, 1)]
    subsampled = data.filter(pl.col("setup_id").is_in(kept_setup_ids))
    info = {
        "target_rows": target_rows,
        "seed": seed,
        "n_setups": len(kept_setup_ids),
        "n_rows": subsampled.height,
        "setup_ids": sorted(kept_setup_ids),
    }
    return subsampled, info


class squareDataset(Dataset):
    """The rows of data.csv belonging to one setup split, as (image, metadata, is_piece,
    color_target, type_target) tuples.

    Splits are by *setup*, not by square, so no board setup ever appears in more than one split.
    TRAIN_TRANSFORM (augmentation) is applied on the train split, EVAL_TRANSFORM on val/test.

    ``subsample_target_rows`` drives the data-efficiency ablation and is honoured on the TRAIN
    split only -- val/test must stay identical across runs or the ablation's numbers are not
    comparable. ``subsample_info`` is None unless a subsample actually happened.
    """
    def __init__(
        self,
        csv_path: Path = Path("data/generated/data.csv"),
        split: str = "train",
        train_transform= v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        eval_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        target_transform = decompose_label,
        subsample_target_rows: int | None = None,
        subsample_seed: int = 0,
    ):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be of type train, val or test. Got {split}.")
        self.data = pl.read_csv(csv_path).filter(pl.col("setup_split").eq(split))
        self.subsample_info = None
        if split == "train" and subsample_target_rows is not None:
            self.data, self.subsample_info = subsample_train_setups(
                self.data, subsample_target_rows, subsample_seed
            )
        self.split = split
        self.transform = train_transform if split == "train" else eval_transform
        self.target_transform = target_transform
        self.setup_metadata_store = {}

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):
        square = self.data[idx, "square"]
        # Newly generated data writes square_image_path with .as_posix() (forward slashes), so it
        # is portable as-is. This PureWindowsPath parse stays only to read LEGACY csvs generated on
        # Windows with str(Path(...)), which stored "\" separators; PureWindowsPath is a pure path
        # (safe to instantiate on any OS) and accepts both "\" and "/", so it resolves either form.
        raw_path = PureWindowsPath(self.data[idx, "square_image_path"]).as_posix()
        # The CSV stores the annotated-image path; the model input is the 4-channel masked
        # array saved alongside it in the same per-square directory.
        img_path = Path(raw_path).parent / f"{square}_masked.npy"
        image = np.load(img_path)

        if self.transform:
            # TODO: remove this workaround. The only reason the RGB and the mask can't go through
            # self.transform(image) together is that the mask values are in {0, 1} rather than
            # {0, 255}, so ToDtype(scale=True) would shrink them to almost nothing.
            rgb = image[..., :3]
            mask = tv_tensors.Mask(image[..., 3])
                # CAREFUL: no .unsqueeze(dim=0) here - that would downgrade the mask back to a
                # plain tensor, and the tv_tensors.Mask wrapper is the whole trick: the v2
                # transforms apply GEOMETRIC ops (RandomAffine) to it, so the mask keeps tracking
                # the square, but skip PHOTOMETRIC ones (ColorJitter, GaussianBlur, GaussianNoise),
                # so the mask stays a clean 0/1 indicator instead of being jittered into noise.

            # Note that the same transform object transforms differently each time
            transformed_rgb = self.transform(rgb)
            transformed_mask = self.transform(mask).unsqueeze(dim=0) # only unsqueeze now!!

            image = torch.cat([transformed_rgb, transformed_mask], dim=0)
            # Shape is now (4, H, W) rather than (H, W, 4), and the type is a plain torch.Tensor
            # rather than a torchvision image (a Tensor subclass, useful for further transforms).
        if not isinstance(image, torch.Tensor):
            raise TypeError("images must be torch tensors. Pass transform to ensure.")

        # Decompose the 13-way label into per-head targets:
        #   is_piece (float 0/1), color_target (0/1 or IGNORE_INDEX), type_target (0..5 or IGNORE_INDEX)
        raw_label = self.data[idx, "label"]
        is_piece, color_target, type_target = self.target_transform(raw_label)

        # Metadata: one-hot of which board corner is top-left in the camera image (the board's
        # orientation), cached per setup. This replaced the old per-square / per-corner-pixel
        # metadata, which let the model fingerprint (and memorise) individual setups.
        metadata = torch.zeros(4, dtype=torch.float32)

        setup_id = self.data[idx, "setup_id"]
        if setup_id not in self.setup_metadata_store:
            setup_metadata_path = Path("data/generated") / setup_id / "calibration_metadata.json"
            with open(setup_metadata_path, "r", encoding="utf-8") as f:
                setup_metadata = json.load(f)
            self.setup_metadata_store[setup_id] = setup_metadata["camera_natural_orientation"]["order"]["tl"]
        metadata[TOP_LEFT_OHE_MAP[self.setup_metadata_store[setup_id]]] = 1

        return image, metadata, is_piece, color_target, type_target

    # Note on board-level evaluation: batching the 64 squares of one position together through
    # this Dataset was considered and abandoned - it silently breaks as soon as a single row is dropped
    # from the CSV. cnn/evaluate.py instead re-reads data.csv, walks the valid game positions of
    # the split, and runs the real BoardEstimator over each position's squares directory.


def create_dataloader(
    split: str,
    shuffle: bool = False,
    batch_size: int = 64,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    train_transform = TRAIN_TRANSFORM,
    eval_transform = EVAL_TRANSFORM,
    target_transform = decompose_label,
    csv_path = Path("data/generated/data.csv"),
    subsample_target_rows: int | None = None,
    subsample_seed: int = 0,
):
    """A DataLoader over the given split of data.csv. num_workers / persistent_workers /
    pin_memory come straight from the Hydra config (see cnn/config.yaml).

    subsample_target_rows drives the data-efficiency ablation; it is ignored on val/test (see
    squareDataset). What was actually kept is on `dataloader.dataset.subsample_info`.
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Split must be of type train, val or test. Got {split}.")
    dataset = squareDataset(
        csv_path=csv_path,
        split=split,
        train_transform=train_transform,
        eval_transform=eval_transform,
        target_transform=target_transform,
        subsample_target_rows=subsample_target_rows,
        subsample_seed=subsample_seed,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory
    )

    return dataloader
