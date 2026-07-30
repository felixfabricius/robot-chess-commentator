"""
Benchmark harness: score one board-reading method over the recorded val/test positions and
append the per-board metrics to a CSV, so the CNN can be compared against the strongest VLM
approaches.

Methods (one per run, selected by `method`):
  - "cnn"           : the trained SquareClassifierMultiHead (needs model_weights_path).
  - "square_label"  : Claude, one call per square, hard label (method iii).
  - "square_logits" : Claude, one call per square, score per label (method iv).
  - "move"          : Claude, one call per board, ordered candidate moves from prev-FEN + after-image (method i).
  - "board"         : Claude, one call per board, ordered candidate positions (FEN) after the move (method ii).
  - "fen_whole"     : Claude, one call reading the whole board straight to a FEN (crude baseline).

Every method is reduced to the same handful of per-board metrics so results are comparable:
  - correct_square      : proportion of the 64 squares read correctly (square methods; "board" reads
                          it off its first returned FEN; None for move/fen_whole).
  - correct_board       : the predicted move equals the move actually played. For the square methods
                          this is game.estimate_move's top-ranked legal move; for move/board it is
                          the first *legal* candidate the model returned (see vision.first_legal_and_stats).
  - board_rank          : normalised rank of the true move, 1.0 when ranked first, None when
                          undefined (fewer than 2 candidates, or the true move is absent). NOTE the
                          basis differs across method families: square methods rank over ALL legal
                          moves; move/board rank over the model's returned legal candidates. Only
                          top-1 correct_board is cleanly paired across all methods.
  - first_output_illegal: (move/board/fen_whole only) whether the model's FIRST candidate was illegal.
  - none_legal          : (move/board/fen_whole only) whether not one returned candidate was legal.
  - n_suggested         : (move/board only) how many candidates the model returned (raw, before the
                          legality filter). None for the square methods and fen_whole.
  - input_tokens / output_tokens / cost / inference_time: per board. Cost is from the API's own
    usage times a per-model price table; time is summed wall-clock (indicative, not exact -- it
    includes network/backoff). Square methods sum these over their 64 calls; the CNN reports local
    compute time with zero token cost.

Output layout (no CIs -- those are computed downstream): one row per (run, split, setup). Each row
carries run-identifier columns (method, model_version, prompt_version, reasoning, prior_correction,
data_path) plus, per metric, the JSON-encoded list of that metric across the setup's boards AND its
setup-level mean (`<metric>` and `<metric>_mean`).
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import anthropic
import numpy as np
import polars as pl
from tqdm import tqdm

from chess_assistant.config import SQUARES, PIECES
from chess_assistant.game import ChessGame
from chess_assistant.vision import (
    BoardEstimator,
    LLMResponseError,
    estimate_board_after_llm,
    estimate_fen_whole_llm,
    estimate_move_llm,
    fen_board_to_labels,
    first_legal_and_stats,
)

METHODS = ("cnn", "square_label", "square_logits", "move", "board", "fen_whole")
SQUARE_METHODS = ("cnn", "square_label", "square_logits")

# The per-board metrics stored as (list-across-boards, setup-mean) pairs.
METRIC_NAMES = (
    "correct_square",
    "correct_board",
    "board_rank",
    "first_output_illegal",
    "none_legal",
    "n_suggested",
    "input_tokens",
    "output_tokens",
    "inference_time",
    "cost",
)

# USD per token (input, output). Cache terms are omitted: every board is a distinct image, so
# nothing is served from cache during evaluation. Unknown models price at 0 (cost reported as 0).
PRICING = {
    "claude-opus-5": (5e-6, 25e-6),
    "claude-opus-4-8": (5e-6, 25e-6),
    "claude-opus-4-7": (5e-6, 25e-6),
    "claude-sonnet-5": (3e-6, 15e-6),
    "claude-haiku-4-5": (1e-6, 5e-6),
}


def _cost(input_tokens: int, output_tokens: int, model: str) -> float:
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    return input_tokens * price_in + output_tokens * price_out


def _square_accuracy(board_estimate, true_labels: dict[str, str]) -> float:
    """Proportion of the 64 squares whose argmax label matches the true label."""
    correct = []
    for square in SQUARES:
        estimate = getattr(board_estimate, square)
        predicted = max(PIECES, key=lambda piece: getattr(estimate, piece))
        correct.append(predicted == true_labels[square])
    return float(np.mean(correct))


def _labels_accuracy(labels: dict[str, str], true_labels: dict[str, str]) -> float:
    return float(np.mean([labels[square] == true_labels[square] for square in SQUARES]))


def _rank(ordered_moves: list[str], actual_move: str, n: int) -> float | None:
    """Normalised rank of `actual_move` in `ordered_moves`: 1.0 when first, spanning down to 0.
    None when undefined -- fewer than 2 candidates, or the true move is absent from the list."""
    if n < 2:
        return None
    try:
        position = ordered_moves.index(actual_move)
    except ValueError:
        return None
    return 1 - position / (n - 1)


def _evaluate_board(
    method,
    *,
    cnn_estimator,
    llm_square_estimator,
    llm_client,
    model_version,
    prompt_version,
    reasoning,
    previous_fen,
    actual_move,
    true_labels,
    squares_dir,
    warped_image_path,
    corner_map,
):
    """Run one method on one board position and return a record keyed by METRIC_NAMES."""
    record = {name: None for name in METRIC_NAMES}

    if method in SQUARE_METHODS:
        estimator = cnn_estimator if method == "cnn" else llm_square_estimator
        start = time.monotonic()
        board_estimate = estimator.estimate_board(squares_dir)
        elapsed = time.monotonic() - start

        record["correct_square"] = _square_accuracy(board_estimate, true_labels)

        with ChessGame(fen=previous_fen) as game:
            estimated_moves = game.estimate_move(board_estimate)
        ordered = [candidate["move"] for candidate in estimated_moves]
        n = len(estimated_moves)
        record["correct_board"] = bool(ordered and ordered[0] == actual_move)
        record["board_rank"] = _rank(ordered, actual_move, n)
        # first_output_illegal / none_legal stay None: these methods only ever rank legal moves.

        if method == "cnn":
            record["input_tokens"] = 0
            record["output_tokens"] = 0
            record["cost"] = 0.0
            record["inference_time"] = elapsed  # local compute; no API cost
        else:
            record["input_tokens"] = estimator.board_input_tokens
            record["output_tokens"] = estimator.board_output_tokens
            record["inference_time"] = estimator.board_elapsed
            record["cost"] = _cost(estimator.board_input_tokens, estimator.board_output_tokens, model_version)
        return record

    # Whole-image strategies: move (i), board (ii), fen_whole. Each yields an ordered candidate
    # list that first_legal_and_stats reduces to a prediction + legality flags. A degenerate or
    # truncated response (LLMResponseError -- usually the model looping into a max_tokens runaway)
    # is recorded as a FAILED board rather than crashing the whole run; the wasted call is still
    # charged, since it cost real tokens.
    try:
        if method == "move":
            result = estimate_move_llm(
                llm_client, warped_image_path, previous_fen, corner_map,
                model=model_version, prompt_version=prompt_version, reasoning=reasoning,
            )
            candidates, kind = result["moves"], "move"
            record["n_suggested"] = len(result["moves"])  # candidates the model returned (raw)
        elif method == "board":
            result = estimate_board_after_llm(
                llm_client, warped_image_path, previous_fen, corner_map,
                model=model_version, prompt_version=prompt_version, reasoning=reasoning,
            )
            candidates, kind = result["fen_boards"], "board"
            record["n_suggested"] = len(result["fen_boards"])  # candidates the model returned (raw)
            # Bonus square metric: read it off the model's first returned board, if parseable.
            if result["fen_boards"]:
                labels = fen_board_to_labels(result["fen_boards"][0])
                if labels is not None:
                    record["correct_square"] = _labels_accuracy(labels, true_labels)
        else:  # fen_whole
            result = estimate_fen_whole_llm(
                llm_client, warped_image_path,
                model=model_version, prompt_version=prompt_version, reasoning=reasoning,
            )
            candidates, kind = [result["fen_board"]], "board"
            labels = fen_board_to_labels(result["fen_board"])
            if labels is not None:
                record["correct_square"] = _labels_accuracy(labels, true_labels)

        stats = first_legal_and_stats(previous_fen, candidates, kind)
        record["correct_board"] = bool(stats["first_legal"] is not None and stats["first_legal"] == actual_move)
        record["board_rank"] = _rank(stats["ordered_legal"], actual_move, len(stats["ordered_legal"]))
        record["first_output_illegal"] = stats["first_output_illegal"]
        record["none_legal"] = stats["none_legal"]
        usage, elapsed = result["usage"], result["elapsed"]
    except LLMResponseError as exc:
        # No usable answer for this board: count it as a failed prediction (no legal candidate),
        # but still charge the wasted call.
        print(f"  {warped_image_path.parent.name}: unusable response ({exc}) -- recorded as failed board")
        record["correct_board"] = False
        record["first_output_illegal"] = True
        record["none_legal"] = True
        if method in ("move", "board"):
            record["n_suggested"] = 0
        usage, elapsed = exc.usage, exc.elapsed

    if usage is not None:
        record["input_tokens"] = usage.input_tokens
        record["output_tokens"] = usage.output_tokens
        record["cost"] = _cost(usage.input_tokens, usage.output_tokens, model_version)
    record["inference_time"] = elapsed
    return record


def _setup_row(method, model_version, prompt_version, reasoning, prior_correction, data_path,
               split, setup_id, metrics):
    """One output row for a (split, setup): identifier columns, then per-metric JSON list + mean."""
    row = {
        "method": method,
        "model_version": model_version,
        "prompt_version": prompt_version,
        "reasoning": reasoning,
        "prior_correction": prior_correction,
        "data_path": Path(data_path).as_posix(),
        "split": split,
        "setup_id": setup_id,
        "n_boards": len(metrics["correct_board"]),
    }
    for name in METRIC_NAMES:
        values = metrics[name]
        row[name] = json.dumps(values)
        numeric = [float(v) for v in values if v is not None]
        row[f"{name}_mean"] = (sum(numeric) / len(numeric)) if numeric else None
    return row


# Identifier columns written by _setup_row that define "this run's configuration": two runs with
# the same values here, at the same (split, setup), are the same generation -- so resume skips them.
_CONFIG_COLS = ("method", "model_version", "prompt_version", "reasoning", "prior_correction")


def _config_expr(method, model_version, prompt_version, reasoning, prior_correction):
    """Polars boolean expression selecting rows produced by this run's configuration."""
    return (
        (pl.col("method") == method)
        & (pl.col("model_version") == model_version)
        & (pl.col("prompt_version") == prompt_version)
        & (pl.col("reasoning") == reasoning)
        & (pl.col("prior_correction") == prior_correction)
    )


def _done_setups(existing, method, model_version, prompt_version, reasoning, prior_correction):
    """The set of (split, setup_id) already present in `existing` for this run's configuration.
    Empty if `existing` is None or lacks the identifier columns (e.g. a foreign/empty CSV)."""
    if existing is None or not (set(_CONFIG_COLS) | {"split", "setup_id"}) <= set(existing.columns):
        return set()
    match = existing.filter(
        _config_expr(method, model_version, prompt_version, reasoning, prior_correction)
    )
    return set(zip(match["split"].to_list(), match["setup_id"].to_list()))


def _flush(output_path, base_existing, new_rows):
    """Write base_existing + new_rows to output_path and return the frame. diagonal_relaxed so a
    metric that is all-null in one run and float in another still concatenates (union of columns,
    widened dtypes). Called after every completed setup so progress survives a crash."""
    frame = pl.DataFrame(new_rows)
    if base_existing is not None:
        frame = pl.concat([base_existing, frame], how="diagonal_relaxed")
    frame.write_csv(output_path)
    return frame


def main(
    method: str,
    model_version: str | None = None,
    prompt_version: int = 1,
    model_weights_path: Path | None = None,
    prior_correction: bool = False,
    reasoning: str = "none",
    splits: tuple[str, ...] = ("val", "test"),
    resume: bool = True,
    max_setups: int | None = None,
    data_path: Path = Path("data/generated/data.csv"),
    output_path: Path = Path("evaluation/results.csv"),
):
    """Evaluate one `method` over the chosen splits and append per-(split, setup) rows to
    `output_path`. Returns the DataFrame written to disk.

    `model_version` labels the run in the output. It doubles as the version identifier for BOTH
    families: a Claude model id for the VLM methods (default "claude-sonnet-5"), or an identifier
    for the trained CNN (there are several -- different masks/crops/data amounts). For "cnn" it
    defaults to the weights path so each checkpoint is distinguishable; pass a friendlier label to
    override.

    The VLM methods make many billed API calls, so runs are made resumable and scopeable:
    - `splits`: which of "val"/"test" to score (both by default).
    - `resume` (default True): skip (split, setup) already present in `output_path` for THIS run's
      configuration; re-run after a failure and it continues where it stopped. `resume=False`
      regenerates and replaces those rows instead.
    - `max_setups`: cap the number of *not-yet-done* setups generated this run (re-run to do more).

    Progress is flushed to `output_path` after every completed setup, and a credit/API failure (or
    Ctrl-C) stops cleanly with everything so far already on disk. The unit is one setup: a setup
    that fails partway is not saved and is redone on the next run.
    """
    assert method in METHODS, f"method must be one of {METHODS}"
    assert set(splits) <= {"val", "test"}, f"splits must be a subset of val/test; got {splits}"
    data_path = Path(data_path)
    output_path = Path(output_path)
    data_root = data_path.parent

    # Resolve the run label. For the CNN the "version" is which checkpoint was scored, so it
    # defaults to the weights path; for the VLM methods it is the Claude model id.
    if method == "cnn":
        assert model_weights_path is not None, "method='cnn' needs model_weights_path"
        model_version = model_version or Path(model_weights_path).as_posix()
    else:
        model_version = model_version or "claude-sonnet-5"

    data = pl.read_csv(data_path).filter(
        # Only positions that are a real game move: a free-placement edit (valid_game_position
        # False) or the opening position of a game carries no previous_board_fen / move_uci.
        pl.col("valid_game_position"),
        pl.col("move_uci") != "",
        pl.col("previous_board_fen") != "",
    )

    # Backends: build the estimator/client this method needs, once.
    cnn_estimator = None
    llm_square_estimator = None
    llm_client = None
    if method == "cnn":
        cnn_estimator = BoardEstimator(
            model_type="CNN",
            model_weights_path=model_weights_path,
            # Top-left corner is overwritten per setup below; any setup's metadata seeds it.
            calibration_metadata_path=Path(data["calibration_metadata_path"][0]),
            prior_correction=prior_correction,
        )
    elif method in ("square_label", "square_logits"):
        llm_square_estimator = BoardEstimator(
            model_type="LLM",
            llm_method=method,
            model_version=model_version,
            prompt_version=prompt_version,
            reasoning=reasoning,
        )
    else:  # move / board / fen_whole
        llm_client = anthropic.Anthropic()

    # --- Resume / overwrite bookkeeping ---------------------------------------------------------
    existing = pl.read_csv(output_path) if output_path.exists() else None
    have_identifier_cols = existing is not None and (
        (set(_CONFIG_COLS) | {"split", "setup_id"}) <= set(existing.columns)
    )

    # Candidate (split, setup) jobs from the requested splits only, in a stable order.
    setup_jobs = [
        (split, setup_id)
        for split in splits
        for setup_id in sorted(
            data.filter(pl.col("setup_split") == split)["setup_id"].unique().to_list()
        )
    ]

    base = existing  # rows to append onto; may have this-config rows dropped for an overwrite run
    if resume:
        done = _done_setups(
            existing, method, model_version, prompt_version, reasoning, prior_correction
        )
        n_before = len(setup_jobs)
        setup_jobs = [job for job in setup_jobs if job not in done]
        if n_before != len(setup_jobs):
            print(f"resume: {n_before - len(setup_jobs)} setup(s) already done, {len(setup_jobs)} queued")

    if max_setups is not None:
        setup_jobs = setup_jobs[:max_setups]

    if not resume and have_identifier_cols:
        # Regenerate: drop this config's rows at the setups we're about to redo, so re-runs don't
        # duplicate. A NUL-joined key emulates (split, setup_id) tuple membership in one expression.
        regen_keys = [f"{split}\x00{setup_id}" for split, setup_id in setup_jobs]
        base = existing.filter(
            ~(
                _config_expr(method, model_version, prompt_version, reasoning, prior_correction)
                & pl.concat_str([pl.col("split"), pl.lit("\x00"), pl.col("setup_id")]).is_in(regen_keys)
            )
        )

    if not setup_jobs:
        print("nothing to do (all requested setups already generated, or max_setups=0)")
        return existing if existing is not None else pl.DataFrame([])

    # --- Generate, flushing after each completed setup so a failure keeps prior progress ---------
    new_rows = []
    for split, setup_id in tqdm(setup_jobs, desc=f"eval {method}", unit="setup"):
        try:
            df_setup = data.filter(
                (pl.col("setup_split") == split) & (pl.col("setup_id") == setup_id)
            )

            with open(Path(df_setup["calibration_metadata_path"][0]), "r", encoding="utf-8") as f:
                calibration_metadata = json.load(f)
            corner_map = calibration_metadata["camera_natural_orientation"]["order"]
            if cnn_estimator is not None:
                cnn_estimator.top_left_corner = corner_map["tl"]

            metrics = defaultdict(list)
            board_ids = sorted(df_setup["image_id"].unique().to_list())
            # Inner bar over boards (leave=False so it clears when the setup finishes); useful for
            # the slow single-call/64-call VLM methods where one setup is many API round-trips.
            for board_id in tqdm(board_ids, desc=setup_id, unit="board", leave=False):
                df_board = df_setup.filter(pl.col("image_id") == board_id)
                if df_board.height != 64:
                    print(f"Warning: {split}/{setup_id}/{board_id} has {df_board.height} rows, not 64")

                true_labels = {
                    r["square"]: r["label"]
                    for r in df_board.select("square", "label").to_dicts()
                }
                first_row = df_board.row(0, named=True)
                previous_fen = first_row["previous_board_fen"]
                actual_move = first_row["move_uci"]

                record = _evaluate_board(
                    method,
                    cnn_estimator=cnn_estimator,
                    llm_square_estimator=llm_square_estimator,
                    llm_client=llm_client,
                    model_version=model_version,
                    prompt_version=prompt_version,
                    reasoning=reasoning,
                    previous_fen=previous_fen,
                    actual_move=actual_move,
                    true_labels=true_labels,
                    squares_dir=data_root / setup_id / board_id / "squares",
                    warped_image_path=data_root / setup_id / board_id / "image_warped.png",
                    corner_map=corner_map,
                )
                for name, value in record.items():
                    metrics[name].append(value)

            if metrics["correct_board"]:  # skip a setup with no evaluable boards
                new_rows.append(_setup_row(
                    method, model_version, prompt_version, reasoning, prior_correction,
                    data_path, split, setup_id, metrics,
                ))
                # Persist after every completed setup: an API failure below then loses at most the
                # next (unfinished) setup, never a completed one.
                _flush(output_path, base, new_rows)
        except (anthropic.APIError, anthropic.APIConnectionError, KeyboardInterrupt) as exc:
            print(f"\nStopped during {split}/{setup_id}: {type(exc).__name__}: {exc}")
            print("Re-run to resume (completed setups are skipped).")
            break

    if not new_rows:
        # Failed on the very first queued setup, or every queued setup had no evaluable boards.
        return base if base is not None else pl.DataFrame([])
    return _flush(output_path, base, new_rows)


if __name__ == "__main__":
    # Score one board-reading method over the val/test positions and append the per-(split, setup)
    # rows to the output CSV. One run = one method + config.
    #
    #   # the trained CNN (model_version defaults to the weights path):
    #   uv run python -m chess_assistant.model.evaluation cnn \
    #       --model-weights-path weights/model_state_dict.safetensors
    #
    #   # the strongest VLM, predicting the move from the after-image, reasoning out loud first:
    #   uv run python -m chess_assistant.model.evaluation move --reasoning text
    #
    #   # generate the next 3 val setups only; safe to re-run (skips done setups) after a topping up
    #   uv run python -m chess_assistant.model.evaluation move --splits val --max-setups 3
    #
    # The VLM methods (square_label, square_logits, move, board, fen_whole) make billed Claude API
    # calls and need ANTHROPIC_API_KEY. Runs are resumable: progress is flushed after every setup,
    # and completed setups are skipped on re-run (pass --no-resume to regenerate).
    import argparse

    parser = argparse.ArgumentParser(
        description="Score one board-reading method over the recorded val/test positions."
    )
    parser.add_argument("method", choices=METHODS, help="Which board-reading method to score.")
    parser.add_argument(
        "--model-version",
        default=None,
        help="Run label. VLM: Claude model id (default claude-sonnet-5). CNN: an identifier for "
        "the checkpoint (defaults to the weights path).",
    )
    parser.add_argument("--prompt-version", type=int, default=1, help="Prompt version (LLM methods).")
    parser.add_argument(
        "--model-weights-path", type=Path, default=None, help="Safetensors checkpoint (method=cnn)."
    )
    parser.add_argument(
        "--prior-correction", action="store_true",
        help="CNN only: subtract the training log-prior (Bayesian prior correction).",
    )
    parser.add_argument(
        "--reasoning", choices=["none", "text", "thinking"], default="none",
        help="LLM reasoning mode before the answer (see vision._call_claude).",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["val", "test"], default=["val", "test"],
        help="Which splits to score (default: both).",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Regenerate setups already in the output CSV instead of skipping them.",
    )
    parser.add_argument(
        "--max-setups", type=int, default=None,
        help="Cap the number of not-yet-done setups generated this run (re-run to do more).",
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/generated/data.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("evaluation/results.csv"))
    args = parser.parse_args()

    main(
        method=args.method,
        model_version=args.model_version,
        prompt_version=args.prompt_version,
        model_weights_path=args.model_weights_path,
        prior_correction=args.prior_correction,
        reasoning=args.reasoning,
        splits=tuple(args.splits),
        resume=args.resume,
        max_setups=args.max_setups,
        data_path=args.data_path,
        output_path=args.output_path,
    )
