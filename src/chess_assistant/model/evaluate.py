import torch
import torch.nn.functional as F
import polars as pl
import numpy as np
import wandb

from pathlib import Path

from chess_assistant.vision import BoardEstimator
from chess_assistant.game import ChessGame
from chess_assistant.model.config import (
    INVERSE_TARGET_MAP,
    INVERSE_COLOR_MAP,
    INVERSE_TYPE_MAP,
    IGNORE_INDEX,
    reconstruct_13way_logprobs,
    logprobs13_from_logits,
    recompose_label13,
)

_split_data_cache = {}


def _confusion_matrix(targets, preds, class_names):
    # W&B's confusion-matrix chart sorts axis labels alphabetically, ignoring table row
    # order; zero-padded index prefixes make the alphabetical order coincide with the
    # intended (TARGET_MAP / COLOR_MAP / TYPE_MAP) order.
    return wandb.plot.confusion_matrix(
        y_true=torch.cat(targets).to("cpu").numpy(),
        preds=torch.cat(preds).to("cpu").numpy(),
        class_names=class_names,
    )


def _square_level_multihead(model, dataloader, loss_fns, loss_weights, device, log_prior):
    """Square-level metrics for the multi-head model (model 3): per-head losses (empty / color /
    type) and their confusion matrices, plus the reconstructed 13-way loss, accuracy and confusion
    matrix. The 13-way view is kept because it is comparable with the single-head models 1 & 2, and
    it is what drives checkpoint selection.
    """
    n = len(dataloader.dataset)

    empty_loss_sum = 0.0
    color_loss_sum = 0.0
    type_loss_sum = 0.0
    square_loss_sum = 0.0  # reconstructed 13-way CrossEntropy
    n_nonempty = 0         # number of non-empty rows actually seen (denominator for color/type)
    n_correct_square = 0

    # per-head predictions/targets, collected across the eval set for the confusion matrices
    empty_preds, empty_targets = [], []
    color_preds, color_targets = [], []
    type_preds, type_targets = [], []
    all_preds13, all_labels13 = [], []

    for images, metadata, is_piece, color_target, type_target in dataloader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        # is_piece comes off the default collate as float64; BCEWithLogitsLoss needs the
        # target dtype to match the float32 logits.
        is_piece = is_piece.to(device, non_blocking=True).float()
        color_target = color_target.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)
        n_batch = is_piece.shape[0]

        # exactly the non-empty rows (color/type are IGNORE_INDEX iff empty)
        nonempty_mask = color_target != IGNORE_INDEX
        n_nonempty_batch = int(nonempty_mask.sum().item())

        with torch.no_grad():
            logit_empty, logits_color, logits_type = model(images, metadata)

            # ---- per-head losses ----
            # empty loss is over every row -> normalise by total n
            empty_loss_sum += loss_fns["empty"](logit_empty, is_piece).item() * n_batch
            # color/type CrossEntropyLoss(reduction="mean", ignore_index=...) averages over the
            # non-empty rows only; scale by that count and divide by total non-empty later.
            # Guard against an all-empty batch (mean over zero rows would be NaN).
            if n_nonempty_batch > 0:
                color_loss_sum += loss_fns["color"](logits_color, color_target).item() * n_nonempty_batch
                type_loss_sum += loss_fns["type"](logits_type, type_target).item() * n_nonempty_batch
                n_nonempty += n_nonempty_batch

            # ---- reconstructed 13-way view (kept, comparable to models 1 & 2) ----
            logprobs = reconstruct_13way_logprobs(
                logit_empty, logits_color, logits_type, log_prior=log_prior
            )  # (batch, 13)
            label13 = recompose_label13(color_target, type_target)
            square_loss_sum += F.cross_entropy(logprobs, label13).item() * n_batch
            preds13 = logprobs.argmax(dim=-1)
            n_correct_square += (preds13 == label13).sum().item()

        # ---- collect predictions/targets for the confusion matrices ----
        empty_preds.append((torch.sigmoid(logit_empty) > 0.5).long())
        empty_targets.append(is_piece.long())
        color_preds.append(logits_color.argmax(dim=-1)[nonempty_mask])
        color_targets.append(color_target[nonempty_mask])
        type_preds.append(logits_type.argmax(dim=-1)[nonempty_mask])
        type_targets.append(type_target[nonempty_mask])
        all_preds13.append(preds13)
        all_labels13.append(label13)

    empty_avg = empty_loss_sum / n
    color_avg = color_loss_sum / n_nonempty if n_nonempty > 0 else 0.0
    type_avg = type_loss_sum / n_nonempty if n_nonempty > 0 else 0.0
    square_avg = square_loss_sum / n  # reconstructed 13-way CE -> checkpoint metric
    total_avg = (
        loss_weights["empty"] * empty_avg
        + loss_weights["color"] * color_avg
        + loss_weights["type"] * type_avg
    )
    prop_correct = n_correct_square / n

    return {
        "eval/square/avg_loss": square_avg,  # reconstructed 13-way CE (drives checkpoint selection)
        "eval/square/prop_correct_square": prop_correct,
        "eval/square/confusion_matrix": _confusion_matrix(
            all_labels13, all_preds13, [f"{i:02d}_{INVERSE_TARGET_MAP[i]}" for i in range(13)]
        ),
        "eval/empty/avg_loss": empty_avg,
        "eval/empty/confusion_matrix": _confusion_matrix(
            empty_targets, empty_preds, ["00_empty", "01_piece"]
        ),
        "eval/color/avg_loss": color_avg,
        "eval/color/confusion_matrix": _confusion_matrix(
            color_targets, color_preds, [f"{i:02d}_{INVERSE_COLOR_MAP[i]}" for i in range(2)]
        ),
        "eval/type/avg_loss": type_avg,
        "eval/type/confusion_matrix": _confusion_matrix(
            type_targets, type_preds, [f"{i:02d}_{INVERSE_TYPE_MAP[i]}" for i in range(6)]
        ),
        "eval/total/avg_loss": total_avg,  # weighted combined loss (diagnostic, mirrors train/total)
    }


def _square_level_single_head(model, dataloader, device, log_prior):
    """Square-level metrics for the single-head models (model 1 / model 2), which emit one 13-way
    logit vector. There are no factored heads to score, so only the 13-way view is produced:
    the CrossEntropy loss (`eval/square/avg_loss`, the same checkpoint-selection metric the
    multi-head path reports, so the two are directly comparable), accuracy, and confusion matrix.

    The 13-way loss is F.cross_entropy over the normalised log-probs - identical to how the
    multi-head path scores its reconstructed distribution - so no separate eval loss object is
    needed (the eval loss is always unweighted, matching the multi-head reconstructed loss).
    """
    n = len(dataloader.dataset)

    square_loss_sum = 0.0
    n_correct_square = 0
    all_preds13, all_labels13 = [], []

    for images, metadata, _is_piece, color_target, type_target in dataloader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        color_target = color_target.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)
        n_batch = color_target.shape[0]

        with torch.no_grad():
            logits = model(images, metadata)
            # Normalised 13-way log-probs, prior-corrected iff log_prior is not None -- the drop-in
            # for the multi-head reconstruction, so the CE below is scored on the same footing.
            logprobs = logprobs13_from_logits(logits, log_prior=log_prior)  # (batch, 13)
            label13 = recompose_label13(color_target, type_target)
            square_loss_sum += F.cross_entropy(logprobs, label13).item() * n_batch
            preds13 = logprobs.argmax(dim=-1)
            n_correct_square += (preds13 == label13).sum().item()

        all_preds13.append(preds13)
        all_labels13.append(label13)

    return {
        "eval/square/avg_loss": square_loss_sum / n,  # 13-way CE (drives checkpoint selection)
        "eval/square/prop_correct_square": n_correct_square / n,
        "eval/square/confusion_matrix": _confusion_matrix(
            all_labels13, all_preds13, [f"{i:02d}_{INVERSE_TARGET_MAP[i]}" for i in range(13)]
        ),
    }


def evaluate(model, dataloader, loss_fns, loss_weights, split, csv_path, device, data_root=None,
             prior_correction=False, multi_head=True):
    """Evaluate the model on `split` and return a dict of W&B-loggable metrics.

    `multi_head` selects the square-level scoring: True for the factored model 3 (per-head losses
    plus the reconstructed 13-way view), False for the single-head models 1 & 2 (13-way view only).
    Either way `eval/square/avg_loss` is a 13-way CrossEntropy, so it is comparable across models
    and drives checkpoint selection identically. For single-head runs the square-level eval loss is
    an unweighted F.cross_entropy, so `loss_fns`/`loss_weights` are unused (pass None); for
    multi-head `loss_fns` is the per-head eval dict and `loss_weights` combines them.

    `data_root` is where the board-level pass looks for each position's square crops and its
    setup's calibration metadata. It defaults to the directory holding `csv_path`, which is the
    layout every generated tree has (`<root>/data.csv` next to `<root>/<setup_id>/...`), so
    pointing a run at a mask-ablation variant needs nothing but its `csv_path`.

    `prior_correction` switches on Bayesian prior correction (subtracting the model's training
    log-prior; see reconstruct_13way_logprobs / logprobs13_from_logits). It deliberately drives BOTH
    levels below - the square-level 13-way view and the board-level move decode - since a
    half-corrected scoreboard would be meaningless: both levels have to describe the same decision
    rule. Unlike live inference (which corrects whenever it can), this defaults to off and is set
    from the config, so one trained checkpoint can be scored both ways without retraining.

    Two levels:
      - Square level, from the dataloader: see _square_level_multihead / _square_level_single_head.
      - Board level, from the CSV: see below. Per-square accuracy is only a proxy; what the robot
        actually needs is to get the MOVE right, so that is measured directly.
    """
    model.eval() # important for batch norm; want to use mean and sd that were accumulated during training
    n = len(dataloader.dataset)

    # None when the correction is off, which is exactly what the reconstruction expects.
    # getattr keeps this working for a model handed in that predates the log_prior buffer.
    log_prior = getattr(model, "log_prior", None) if prior_correction else None

    if multi_head:
        square_metrics = _square_level_multihead(
            model, dataloader, loss_fns, loss_weights, device, log_prior
        )
    else:
        square_metrics = _square_level_single_head(model, dataloader, device, log_prior)

    ### On board level
    # Per-square accuracy is not the metric that matters: 63 correct squares and one wrong one can
    # still produce the wrong move, and a wrong square on an irrelevant part of the board costs
    # nothing. So the whole production pipeline is run here - BoardEstimator over the 64 crops of
    # a position, then ChessGame.estimate_move to rank the legal moves against that estimate - and
    # top-1 MOVE accuracy is scored. Squares can't be shuffled across boards for this, so it reads
    # data.csv directly rather than going through the dataloader.
    data_root = Path(data_root) if data_root is not None else Path(csv_path).parent
    cache_key = (str(csv_path), split)
    if cache_key not in _split_data_cache:
        _split_data_cache[cache_key] = (
            pl.read_csv(csv_path)
            .filter(
                pl.col("setup_split").eq(split),
                pl.col("valid_game_position")
            )
        )
    data = _split_data_cache[cache_key]

    # Each unique image_id is one board position (64 rows in the CSV).
    board_position_ids = data.unique("image_id")["image_id"]

    correct_moves = 0
    correct_move_normalised_rank = []
    n_valid = 0

    for board_position_id in board_position_ids:
        # The 64 rows of a position all carry the same setup/FEN/move, so .first() is enough.
        # TODO: assert that in a test rather than trusting it.
        board_position_data = (
            data.filter(pl.col("image_id").eq(board_position_id))
            .select(
                pl.col("setup_id").first(),
                pl.col("previous_board_fen").first(),
                pl.col("board_fen").first(),
                pl.col("move_uci").first()
            )
        )
        board_position_data = board_position_data.to_dicts()[0]
        squares_dir = data_root / board_position_data["setup_id"] / board_position_id / "squares"

        board_estimator = BoardEstimator(
            model_type="CNN",
            model=model,
            calibration_metadata_path=data_root / board_position_data["setup_id"] / "calibration_metadata.json",
            device=device,
            # Passed explicitly: BoardEstimator defaults this to True for live inference, but eval
            # has to mirror whatever the square-level pass above did, or the two levels of the same
            # scoreboard would describe different decision rules.
            prior_correction=prior_correction
        )
        board_estimator.estimate_board(squares_dir)

        # estimate_move() ranks legal moves against the board estimate and never consults
        # Stockfish, so no engine is spawned here. The with-block is the guarantee that stays
        # true: if anything below ever does reach for the engine, its process is still reaped.
        with ChessGame(fen=board_position_data["previous_board_fen"], model_type="CNN") as game:
            assert game.board.is_valid() # should not arrive at an invalid position this way
            estimated_moves = game.estimate_move(board_estimator.board_estimate)

            if len(estimated_moves) == 0:
                continue

            # estimated_moves is the legal moves, best-scoring first.
            if estimated_moves[0]["move"] == board_position_data["move_uci"]:
                correct_moves += 1

            for i, scored_move in enumerate(estimated_moves):
                if scored_move["move"] == board_position_data["move_uci"]:
                    # Normalised rank of the true move: 1.0 when it is ranked first.
                    # TODO: open question whether the denominator should be len(estimated_moves)
                    # or len(estimated_moves) - 1. As written, the worst-ranked move scores
                    # 1/n rather than 0, so the metric never reaches its floor; dividing by
                    # n - 1 would make it span the full [0, 1] but is undefined for n == 1.
                    correct_move_normalised_rank.append(1 - i / len(estimated_moves))
                    break

            n_valid += 1

    metrics = {
        "eval/square/n": n,
        **square_metrics,
        "eval/board/n_valid": n_valid,
        "eval/board/prop_correct_board": correct_moves / n_valid if n_valid > 0 else None,
        "eval/board/correct_normalised_rank": np.mean(correct_move_normalised_rank) if len(correct_move_normalised_rank) > 0 else None,
    }

    return metrics
