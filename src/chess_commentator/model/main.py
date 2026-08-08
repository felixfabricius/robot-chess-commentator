"""Training entry point: Hydra for the config, Weights & Biases for the logging.

    uv run python -m chess_assistant.model.main
    uv run python -m chess_assistant.model.main model=2 training.epochs=10
    uv run python -m chess_assistant.model.main +debug=true

Defaults live in model/config.yaml and any of them can be overridden on the command line.
`debug` and `prefix` are the exception: they are not in config.yaml, and Hydra's struct mode
rejects unknown keys, so they need the `+` prefix to be appended (`+debug=true` cuts each epoch
to a few batches, for a quick smoke run).

Needs a W&B API key in .env, and data/generated/data.csv to exist. The best epoch's weights are
written to .cache/model_<run_id>.pt and uploaded as a W&B artifact; the copy the robot actually
runs on lives in weights/ as safetensors.
"""

import torch
import wandb
import copy
import numpy as np
from torch import nn
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from dotenv import load_dotenv

from chess_assistant.model.model import SquareClassifier, SquareClassifier2, SquareClassifierMultiHead
from chess_assistant.model.data import create_dataloader
from chess_assistant.model.train import train
from chess_assistant.model.evaluate import evaluate
from chess_assistant.model.config import (
    decompose_label,
    IGNORE_INDEX,
    log_prior_from_label_counts,
)
from chess_assistant.image_processing import MASK_VARIANTS

load_dotenv() # for api keys

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon GPU
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Used by evaluate; asserted here so a bad path fails before a run is even created.
    # (Class weights and the log-prior now come off the train dataloader instead, so that they
    # describe the rows actually trained on when data.subsample is active.)
    csv_path = Path(config.data.get("csv_path"))
    assert csv_path.exists()

    
    # The train dataloader is built up here, ahead of both wandb.init and the model, because the
    # data-efficiency ablation may subsample it: the run name, the log-prior and the class weights
    # all have to describe the data the model is actually shown, not the whole train split.
    dataloader_cfg = config.get("data", {}).get("dataloader", {})
    subsample_cfg = config.get("data", {}).get("subsample", {}) or {}
    train_dataloader = create_dataloader(
        split="train",
        shuffle=True,
        batch_size=config.training.get("batch_size", 64),
        num_workers=dataloader_cfg.get("num_workers", 0),
        persistent_workers=dataloader_cfg.get("persistent_workers", False),
        pin_memory=dataloader_cfg.get("pin_memory", False) and device == torch.device("cuda"),
        csv_path=csv_path,
        subsample_target_rows=subsample_cfg.get("target_rows"),
        subsample_seed=subsample_cfg.get("seed", 0),
    )
    subsample_info = train_dataloader.dataset.subsample_info

    # Whether the val metrics are scored with Bayesian prior correction. This changes what every
    # eval number means (including the checkpoint-selection loss), so it goes in the run name.
    prior_correction = bool(config.get("inference", {}).get("prior_correction", False))

    # Which mask/padding ablation tree this run reads, inferred from the data root: regenerate.py
    # writes variant trees as ``<root>_<variant>``, and the mask/crop it was cut with is not
    # otherwise recoverable from the config (it is baked into the crops). None for the default
    # (hull, per_square) tree, so ordinary runs stay unannotated. Longest matching suffix wins, so
    # no variant name that is a suffix of another could be misread.
    data_root_name = csv_path.parent.name
    mask_variant = max(
        (v for v in MASK_VARIANTS if v != "default" and data_root_name.endswith(f"_{v}")),
        key=len,
        default=None,
    )
    mask_modes = MASK_VARIANTS[mask_variant] if mask_variant is not None else None

    if config.data.weighting == "inverse_root" and config.note == "":
        config.note = "Weighting: Inverse Root"
    if prior_correction:
        config.note = f"{config.note} | Prior correction" if config.note else "Prior correction"
    if subsample_info is not None:
        # The REALISED size, not just the requested target: whole setups are kept, so the actual
        # size lands near the target rather than on it, and the realised count is what the
        # ablation's x-axis uses.
        subsample_label = (
            f"Subsample: {subsample_info['n_rows']} rows / {subsample_info['n_setups']} setups"
            f" (target {subsample_info['target_rows']})"
        )
        config.note = f"{config.note} | {subsample_label}" if config.note else subsample_label
    if mask_variant is not None:
        mask_label = f"Mask: {mask_variant} (mask={mask_modes[0]}, crop={mask_modes[1]})"
        config.note = f"{config.note} | {mask_label}" if config.note else mask_label
    if config.get("debug") and not config.get("prefix"):
        # A plain `config.prefix = ...` raises here: the config is in "struct mode", which exists
        # to stop typo'd keys being silently added. force_add is the sanctioned way through.
        OmegaConf.update(config, "prefix", "test run", force_add=True)
    run_name = (
        f"{('[' + config.get('prefix') + '] | ') if config.get('prefix') else ''}"
        f"Model: {config.model}"
        f"{f' | {config.note}' if config.note else ''}"
    )
    # Tags are what the W&B UI filters on, so the ablation arm gets one keyed to the requested
    # target (stable across seeds/splits, unlike the realised row count).
    run_tags = []
    if subsample_info is not None:
        run_tags.append("subsample")
        run_tags.append(f"subsample-{subsample_info['target_rows']}")
    if prior_correction:
        run_tags.append("prior-correction")
    if mask_variant is not None:
        run_tags.append("mask-ablation")
        run_tags.append(f"mask-{mask_variant}")
    # The description spells out everything needed to reproduce the arm, one sentence per
    # non-default choice, including anything not recoverable from the config alone (e.g. which
    # setups a subsample kept). None when nothing notable is active, so plain runs stay unadorned.
    note_parts = []
    if prior_correction:
        note_parts.append(
            "Bayesian prior correction applied: the training log-prior is subtracted from the "
            "13-way log-probs at eval (square- and board-level), so every eval metric here "
            "- including the checkpoint-selection loss - reflects the prior-corrected decision rule."
        )
    if subsample_info is not None:
        note_parts.append(
            f"Data-efficiency ablation. Train split subsampled to {subsample_info['n_rows']} rows "
            f"from {subsample_info['n_setups']} whole setups "
            f"(target {subsample_info['target_rows']}, seed {subsample_info['seed']}). "
            f"Val/test untouched. Kept setups: {', '.join(subsample_info['setup_ids'])}."
        )
    if mask_variant is not None:
        note_parts.append(
            f"Mask/padding ablation arm '{mask_variant}' (mask={mask_modes[0]}, "
            f"crop={mask_modes[1]}), reading crops from {csv_path.parent}."
        )
    run_notes = " ".join(note_parts) if note_parts else None
    run = wandb.init(
        project="chess-assistant",
        name=run_name,
        notes=run_notes,
        tags=run_tags,
        config=config,
        dir=".cache/wandb"
    )
    run.define_metric("epoch") # tells wandb I will log metric
    run.define_metric("*", step_metric="epoch") # for all other metrics matching "*", use epoch as x-axis

    # Data-efficiency ablation: record exactly which setups survived, so a run can be reproduced
    # (and so the x-axis of the ablation is the real row count, not the requested target).
    if subsample_info is not None:
        print(
            f"Subsampled train split to {subsample_info['n_rows']} rows from "
            f"{subsample_info['n_setups']} setups (target {subsample_info['target_rows']})"
        )
        run.summary["data/subsample/n_rows"] = subsample_info["n_rows"]
        run.summary["data/subsample/n_setups"] = subsample_info["n_setups"]
        run.summary["data/subsample/target_rows"] = subsample_info["target_rows"]
        run.summary["data/subsample/seed"] = subsample_info["seed"]
        run.summary["data/subsample/setup_ids"] = subsample_info["setup_ids"]

    # 13-way training log-prior, for optional Bayesian prior correction at eval/inference time.
    # Computed UNCONDITIONALLY rather than gated on inference.prior_correction: the correction is
    # purely post-hoc, so storing the prior with every checkpoint means one trained model can be
    # evaluated both with and without it (just flip the flag) instead of needing two runs.
    # Counted over the rows actually trained on, so that under data.subsample the prior describes
    # the distribution the model really learned rather than that of the full train split.
    label_counts = {
        row["label"]: row["count"]
        for row in train_dataloader.dataset.data["label"].value_counts().iter_rows(named=True)
    }
    log_prior = log_prior_from_label_counts(label_counts)

    assert config.model in [1, 2, 3]
    # Model 3 (SquareClassifierMultiHead) is the actively-developed model and the one the
    # training pipeline (5-tuple dataset, per-head losses) now targets. 1 and 2 remain as
    # selectable classes for reference / loading old checkpoints.
    if config.model == 1:
        model = SquareClassifier()
    elif config.model == 2:
        model = SquareClassifier2()
    else:
        model = SquareClassifierMultiHead(log_prior=log_prior)
    model = model.to(device) # model.to(device) is also in place so would be sufficient
    lowest_loss = float("inf")
    best_model_state = None
    optimizer_state = None
    best_epoch = 0
    
    val_dataloader = create_dataloader(
        split="val",
        shuffle=False,
        batch_size=config.training.get("batch_size", 64),
        num_workers=dataloader_cfg.get("num_workers", 0),
        persistent_workers=dataloader_cfg.get("persistent_workers", False),
        pin_memory=dataloader_cfg.get("pin_memory", False) and device == torch.device("cuda"),
        csv_path=csv_path,
    )

    # Hyperparameters
    lr = config.optimizer.lr
    weight_decay = config.optimizer.get("weight_decay", 1e-4)

    # Per-head class weights (inverse-sqrt-frequency), computed from the train split via
    # decompose_label. Gated on config.data.weighting exactly like the single-head models.
    if config.data.get("weighting") == "inverse_root":
        # Counted over the rows the model will actually train on rather than over the whole train
        # split: with data.subsample active the two differ, and weights derived from the full split
        # would not match the class balance the model is really shown.
        counts = train_dataloader.dataset.data["label"].value_counts()
        n_empty = 0
        n_piece = 0
        color_counts = torch.zeros(2, dtype=torch.float32)
        type_counts = torch.zeros(6, dtype=torch.float32)
        for row in counts.iter_rows(named=True):
            is_piece, color_target, type_target = decompose_label(row["label"])
            if is_piece == 0.0:
                n_empty += row["count"]
            else:
                n_piece += row["count"]
                color_counts[color_target] += row["count"]
                type_counts[type_target] += row["count"]
        # pos_weight for BCEWithLogitsLoss: inverse-sqrt-frequency ratio between the positive
        # (piece) and negative (empty) classes = (1/sqrt(n_piece)) / (1/sqrt(n_empty)).
        pos_weight_empty = torch.tensor(
            np.sqrt(n_empty) / np.sqrt(n_piece), dtype=torch.float32
        ).to(device)
        color_weights = (1 / torch.sqrt(color_counts)).to(device)
        type_weights = (1 / torch.sqrt(type_counts)).to(device)
        train_loss_fns = {
            "empty": nn.BCEWithLogitsLoss(pos_weight=pos_weight_empty),
            "color": nn.CrossEntropyLoss(weight=color_weights, ignore_index=IGNORE_INDEX),
            "type": nn.CrossEntropyLoss(weight=type_weights, ignore_index=IGNORE_INDEX),
        }
    else:
        train_loss_fns = {
            "empty": nn.BCEWithLogitsLoss(),
            "color": nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
            "type": nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
        }

    # Eval losses are always unweighted (mirrors the old unweighted eval_loss_fn).
    eval_loss_fns = {
        "empty": nn.BCEWithLogitsLoss(),
        "color": nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
        "type": nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
    }
    # Combination weights for the three heads (used for the back-propagated train loss and the
    # eval/total diagnostic). Sensible defaults if the section is missing from config.
    loss_weights_cfg = config.get("loss_weights", {})
    loss_weights = {
        "empty": loss_weights_cfg.get("empty", 1.0),
        "color": loss_weights_cfg.get("color", 1.0),
        "type": loss_weights_cfg.get("type", 1.0),
    }
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    epochs = config.training.get("epochs", 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config.get("scheduler", {}).get("eta_min", 1e-6)
    )
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}\n------------------------------")
        train_metrics = train(
            model=model,
            dataloader=train_dataloader,
            loss_fns=train_loss_fns,
            loss_weights=loss_weights,
            optimizer=optimizer,
            debug=config.get("debug", False),
            device=device
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_dataloader,
            loss_fns=eval_loss_fns,
            loss_weights=loss_weights,
            split="val",
            csv_path=csv_path,
            device=device,
            prior_correction=prior_correction
        )

        run.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **train_metrics, **val_metrics})

        # Update best model
        if val_metrics["eval/square/avg_loss"] < lowest_loss:
            lowest_loss = val_metrics["eval/square/avg_loss"]
            # These are snapshotted on whatever device training is running on. When that is CUDA,
            # the model weights and the AdamW momentum buffers are CUDA tensors and get pickled
            # with their device info, so a plain torch.load() on a GPU-less machine crashes; load
            # such a checkpoint with map_location="cpu". Moving the tensors to CPU here instead was
            # considered and rejected: model.to("cpu") moves the model in place, mid-training.
            best_model_state = copy.deepcopy(model.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch

        scheduler.step()

    # Persist the best version of the model
    cache_path = Path(".cache") / f"model_{run.id}.pt"
    cache_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": best_model_state,
            "optimizer_state_dict": optimizer_state
        },
        cache_path
    )
    artifact = wandb.Artifact(name="model_and_optimizer", type="model")
    artifact.add_file(cache_path)
    run.log_artifact(artifact)

    run.finish()

    return None 

if __name__ == "__main__":
    main()