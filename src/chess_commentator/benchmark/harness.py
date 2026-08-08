"""
Benchmark harness: score one board-reading method over the recorded val/test positions and
append the per-board metrics to a CSV, so the CNN can be compared against the strongest VLM
approaches.

Methods (one per run, selected by `method`):
  - "cnn"           : the trained SquareClassifierMultiHead (needs model_weights_path). Runs locally.
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
    usage times a per-model price table, HALVED for the VLM methods because they run through the
    Message Batches API (50% discount). Square methods sum tokens/cost over their 64 calls. The CNN
    reports local compute time with zero token cost; the batched VLM methods leave inference_time
    None -- a batched call has no meaningful per-board wall-clock (see the batch flow below).

How the VLM methods run -- the Message Batches API:
    The VLM methods make many billed Claude calls, so instead of calling synchronously the harness
    builds every request up front and submits them to the Message Batches API (50% cheaper, and the
    server handles rate-limiting). A batch runs server-side and is durable: it keeps going even if
    this process dies. A run therefore has two phases -- submit, then (once the batch has finished)
    collect + score -- and is fully resumable:
      - The batch ids for each (config, split, setup) are written to a sidecar JSON next to the
        output CSV (`<output>.batches.json`) the instant a batch is created, so a crash or a
        deliberate `--no-wait` never loses -- or re-submits (re-pays for) -- work already running.
      - `main()` is idempotent: each run submits any not-yet-submitted setups, then (unless
        `--no-wait`) polls until they finish and collects them. A setup is "done" once its row is in
        the CSV; re-run any time to continue.
      - `--no-wait` submits and exits (for big overnight batches); a later re-run collects.

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

from chess_commentator.config import SQUARES, PIECES
from chess_commentator.game import ChessGame
from chess_commentator.vision import (
    BoardEstimate,
    BoardEstimator,
    EFFORT_LEVELS,
    LLMResponseError,
    SquareEstimate,
    build_board_params,
    build_fen_whole_params,
    build_move_params,
    build_square_params,
    fen_board_to_labels,
    first_legal_and_stats,
    parse_message,
    parsed_to_square_estimate,
    _collect_slots,
)

METHODS = ("cnn", "square_label", "square_logits", "move", "board", "fen_whole")
SQUARE_METHODS = ("cnn", "square_label", "square_logits")
# Everything except the CNN goes through the Message Batches API.
BATCH_METHODS = ("square_label", "square_logits", "move", "board", "fen_whole")

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

# The API's per-batch ceilings are 100k requests / 256MB; inline base64 images dominate size, so a
# large square-method setup (64 calls x many boards) is split across several batches to stay clear.
_MAX_REQUESTS_PER_BATCH = 1000
_MAX_BYTES_PER_BATCH = 200 * 1024 * 1024


def _cost(input_tokens: int, output_tokens: int, model: str, batch: bool = False) -> float:
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    cost = input_tokens * price_in + output_tokens * price_out
    return cost * 0.5 if batch else cost  # Message Batches API is 50% off


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


# --- Scoring (pure): turn a board estimate / parsed answer into a metrics record ------------------
# Scoring is separated from making the call, because the batch path scores responses that were
# produced minutes-to-hours earlier. Each function fills only the accuracy metrics; the caller adds
# the token/cost/time columns (which differ between the local CNN and the batched VLM methods).

def _score_square_board(board_estimate, previous_fen, actual_move, true_labels) -> dict:
    """Score a fully-populated BoardEstimate (the square methods and the CNN) into a record: square
    accuracy plus game.estimate_move's move ranking."""
    record = {name: None for name in METRIC_NAMES}
    record["correct_square"] = _square_accuracy(board_estimate, true_labels)
    with ChessGame(fen=previous_fen) as game:
        estimated_moves = game.estimate_move(board_estimate)
    ordered = [candidate["move"] for candidate in estimated_moves]
    n = len(estimated_moves)
    record["correct_board"] = bool(ordered and ordered[0] == actual_move)
    record["board_rank"] = _rank(ordered, actual_move, n)
    # first_output_illegal / none_legal stay None: these methods only ever rank legal moves.
    return record


def _score_whole_image(method, parsed, previous_fen, actual_move, true_labels) -> dict:
    """Score one whole-image answer (move/board/fen_whole) into a record. `parsed` is the JSON
    object the model returned, or None when the call failed (errored/expired/max_tokens truncation)
    -- a failure is recorded as a failed board (no legal candidate), mirroring the old synchronous
    LLMResponseError handling."""
    record = {name: None for name in METRIC_NAMES}
    if parsed is None:
        record["correct_board"] = False
        record["first_output_illegal"] = True
        record["none_legal"] = True
        if method in ("move", "board"):
            record["n_suggested"] = 0
        return record

    if method == "move":
        candidates, kind = _collect_slots(parsed, "move"), "move"
        record["n_suggested"] = len(candidates)  # candidates the model returned (raw)
    elif method == "board":
        candidates, kind = _collect_slots(parsed, "board"), "board"
        record["n_suggested"] = len(candidates)
        # Bonus square metric: read it off the model's first returned board, if parseable.
        if candidates:
            labels = fen_board_to_labels(candidates[0])
            if labels is not None:
                record["correct_square"] = _labels_accuracy(labels, true_labels)
    else:  # fen_whole
        fen_board = str(parsed.get("fen_board", "")).strip()
        candidates, kind = [fen_board], "board"
        labels = fen_board_to_labels(fen_board)
        if labels is not None:
            record["correct_square"] = _labels_accuracy(labels, true_labels)

    stats = first_legal_and_stats(previous_fen, candidates, kind)
    record["correct_board"] = bool(stats["first_legal"] is not None and stats["first_legal"] == actual_move)
    record["board_rank"] = _rank(stats["ordered_legal"], actual_move, len(stats["ordered_legal"]))
    record["first_output_illegal"] = stats["first_output_illegal"]
    record["none_legal"] = stats["none_legal"]
    return record


def _evaluate_cnn_board(cnn_estimator, previous_fen, actual_move, true_labels, squares_dir) -> dict:
    """Run the trained CNN over one board's 64 square cutouts and score it. Local compute, so the
    cost is zero and inference_time is the measured wall-clock."""
    start = time.monotonic()
    board_estimate = cnn_estimator.estimate_board(squares_dir)
    elapsed = time.monotonic() - start
    record = _score_square_board(board_estimate, previous_fen, actual_move, true_labels)
    record["input_tokens"] = 0
    record["output_tokens"] = 0
    record["cost"] = 0.0
    record["inference_time"] = elapsed
    return record


def _score_batched_board(method, board_id, df_setup, parsed_by_cid, model_version) -> dict:
    """Assemble one board's batched answers (looked up by custom_id) and score them, adding the
    summed token usage and the (batch-discounted) cost. A square whose call failed contributes an
    empty estimate; a failed whole-image call scores as a failed board."""
    df_board = df_setup.filter(pl.col("image_id") == board_id)
    true_labels = {r["square"]: r["label"] for r in df_board.select("square", "label").to_dicts()}
    first_row = df_board.row(0, named=True)
    previous_fen = first_row["previous_board_fen"]
    actual_move = first_row["move_uci"]

    input_tokens = output_tokens = 0
    if method in ("square_label", "square_logits"):
        board_estimate = BoardEstimate()
        for square in SQUARES:
            entry = parsed_by_cid.get(_square_custom_id(board_id, square))
            if entry is not None and entry["parsed"] is not None:
                setattr(board_estimate, square, parsed_to_square_estimate(method, entry["parsed"]))
            else:
                # A failed square: all-zero scores. Under estimate_move's softmax this is a shrug
                # (no information) rather than a corrupted reading of that square.
                setattr(board_estimate, square, SquareEstimate())
            if entry is not None and entry["usage"] is not None:
                input_tokens += entry["usage"].input_tokens
                output_tokens += entry["usage"].output_tokens
        record = _score_square_board(board_estimate, previous_fen, actual_move, true_labels)
    else:
        entry = parsed_by_cid.get(board_id)
        parsed = entry["parsed"] if entry is not None else None
        record = _score_whole_image(method, parsed, previous_fen, actual_move, true_labels)
        if entry is not None and entry["usage"] is not None:
            input_tokens = entry["usage"].input_tokens
            output_tokens = entry["usage"].output_tokens

    record["input_tokens"] = input_tokens
    record["output_tokens"] = output_tokens
    record["cost"] = _cost(input_tokens, output_tokens, model_version, batch=True)
    record["inference_time"] = None  # batched: no meaningful per-board wall-clock
    return record


def _setup_row(method, model_version, prompt_version, reasoning, prior_correction, effort, data_path,
               split, setup_id, metrics):
    """One output row for a (split, setup): identifier columns, then per-metric JSON list + mean."""
    row = {
        "method": method,
        "model_version": model_version,
        "prompt_version": prompt_version,
        "reasoning": reasoning,
        "prior_correction": prior_correction,
        "effort": effort,
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
# `effort` only bites in thinking mode, so non-thinking runs record the API default "high" (see
# main); an older CSV without the column is backfilled to "high" so its runs still match.
_CONFIG_COLS = ("method", "model_version", "prompt_version", "reasoning", "prior_correction", "effort")


def _config_expr(method, model_version, prompt_version, reasoning, prior_correction, effort):
    """Polars boolean expression selecting rows produced by this run's configuration."""
    return (
        (pl.col("method") == method)
        & (pl.col("model_version") == model_version)
        & (pl.col("prompt_version") == prompt_version)
        & (pl.col("reasoning") == reasoning)
        & (pl.col("prior_correction") == prior_correction)
        & (pl.col("effort") == effort)
    )


def _done_setups(existing, method, model_version, prompt_version, reasoning, prior_correction, effort):
    """The set of (split, setup_id) already present in `existing` for this run's configuration.
    Empty if `existing` is None or lacks the identifier columns (e.g. a foreign/empty CSV)."""
    if existing is None or not (set(_CONFIG_COLS) | {"split", "setup_id"}) <= set(existing.columns):
        return set()
    match = existing.filter(
        _config_expr(method, model_version, prompt_version, reasoning, prior_correction, effort)
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


# --- Batch-API plumbing --------------------------------------------------------------------------

def _load_setup(data, split, setup_id):
    """The setup's data rows and its corner_map (camera_natural_orientation order)."""
    df_setup = data.filter((pl.col("setup_split") == split) & (pl.col("setup_id") == setup_id))
    with open(Path(df_setup["calibration_metadata_path"][0]), "r", encoding="utf-8") as f:
        corner_map = json.load(f)["camera_natural_orientation"]["order"]
    return df_setup, corner_map


def _square_custom_id(board_id, square):
    """Batch custom_ids must match ^[a-zA-Z0-9_-]{1,64}$, so the square methods join board and
    square with '_' (a '/' is rejected). board_ids already contain '_'/'-', but the id is never
    split back apart -- only rebuilt to look results up -- so the ambiguity is harmless."""
    return f"{board_id}_{square}"


def _setup_requests(method, df_setup, data_root, setup_id, corner_map,
                    model_version, prompt_version, reasoning, effort):
    """Yield (custom_id, params) for every Claude call a VLM method needs over this setup's boards.
    The custom_id locates the answer for scoring: '<board_id>' for the whole-image methods,
    '<board_id>_<square>' for the square methods (unique within the setup's batch(es))."""
    board_ids = sorted(df_setup["image_id"].unique().to_list())
    for board_id in board_ids:
        df_board = df_setup.filter(pl.col("image_id") == board_id)
        previous_fen = df_board.row(0, named=True)["previous_board_fen"]
        board_dir = data_root / setup_id / board_id
        warped = board_dir / "image_warped.png"
        if method in ("square_label", "square_logits"):
            for square in SQUARES:
                base = board_dir / "squares" / square / f"{square}.png"
                annotated = base.parent / (base.stem + "_annotated" + base.suffix)
                yield _square_custom_id(board_id, square), build_square_params(
                    method, annotated, model_version, prompt_version, reasoning, effort
                )
        elif method == "move":
            yield board_id, build_move_params(
                warped, previous_fen, corner_map, model_version, prompt_version, reasoning, effort
            )
        elif method == "board":
            yield board_id, build_board_params(
                warped, previous_fen, corner_map, model_version, prompt_version, reasoning, effort
            )
        else:  # fen_whole
            yield board_id, build_fen_whole_params(
                warped, corner_map, model_version, prompt_version, reasoning, effort
            )


def _chunk_requests(requests):
    """Split (custom_id, params) pairs into batches under the API's per-batch count/size limits."""
    batch, n_bytes = [], 0
    for custom_id, params in requests:
        size = len(json.dumps(params))
        if batch and (len(batch) >= _MAX_REQUESTS_PER_BATCH or n_bytes + size > _MAX_BYTES_PER_BATCH):
            yield batch
            batch, n_bytes = [], 0
        batch.append((custom_id, params))
        n_bytes += size
    if batch:
        yield batch


def _create_batch(client, chunk):
    """Create one Message Batch from a chunk of (custom_id, params) and return its id."""
    requests = [{"custom_id": custom_id, "params": params} for custom_id, params in chunk]
    return client.messages.batches.create(requests=requests).id


def _await_batches(client, batch_ids, poll_interval):
    """Block until every batch has ended (succeeded/errored/expired all resolve to 'ended')."""
    pending = set(batch_ids)
    while pending:
        for batch_id in list(pending):
            if client.messages.batches.retrieve(batch_id).processing_status == "ended":
                pending.discard(batch_id)
        if pending:
            time.sleep(poll_interval)


def _collect_batches(client, batch_ids):
    """Map custom_id -> {"parsed": dict|None, "usage": usage|None} across the batches. A request that
    errored/expired/was canceled, or whose message did not parse (max_tokens truncation), yields
    parsed=None; usage is kept only when the model actually produced (and billed) a message."""
    results = {}
    for batch_id in batch_ids:
        for item in client.messages.batches.results(batch_id):
            if item.result.type == "succeeded":
                message = item.result.message
                try:
                    parsed = parse_message(message)
                except LLMResponseError:
                    parsed = None
                results[item.custom_id] = {"parsed": parsed, "usage": message.usage}
            else:
                results[item.custom_id] = {"parsed": None, "usage": None}
    return results


def _config_key(method, model_version, prompt_version, reasoning, prior_correction, effort):
    return "|".join(str(v) for v in
                    (method, model_version, prompt_version, reasoning, prior_correction, effort))


def _sidecar_path(output_path):
    return output_path.with_suffix(".batches.json")


def _load_sidecar(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_sidecar(path, state):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# --- Execution engines ---------------------------------------------------------------------------

def _run_cnn(data, data_root, setup_jobs, base, cnn_estimator, method, model_version,
             prompt_version, reasoning, prior_correction, effort, data_path, output_path):
    """Score the CNN locally, flushing after each setup so a Ctrl-C keeps completed setups."""
    new_rows = []
    for split, setup_id in tqdm(setup_jobs, desc="eval cnn", unit="setup"):
        try:
            df_setup, corner_map = _load_setup(data, split, setup_id)
            cnn_estimator.top_left_corner = corner_map["tl"]

            metrics = defaultdict(list)
            board_ids = sorted(df_setup["image_id"].unique().to_list())
            for board_id in tqdm(board_ids, desc=setup_id, unit="board", leave=False):
                df_board = df_setup.filter(pl.col("image_id") == board_id)
                if df_board.height != 64:
                    print(f"Warning: {split}/{setup_id}/{board_id} has {df_board.height} rows, not 64")
                true_labels = {
                    r["square"]: r["label"] for r in df_board.select("square", "label").to_dicts()
                }
                first_row = df_board.row(0, named=True)
                record = _evaluate_cnn_board(
                    cnn_estimator, first_row["previous_board_fen"], first_row["move_uci"],
                    true_labels, data_root / setup_id / board_id / "squares",
                )
                for name, value in record.items():
                    metrics[name].append(value)

            if metrics["correct_board"]:  # skip a setup with no evaluable boards
                new_rows.append(_setup_row(
                    method, model_version, prompt_version, reasoning, prior_correction, effort,
                    data_path, split, setup_id, metrics,
                ))
                _flush(output_path, base, new_rows)
        except KeyboardInterrupt:
            print(f"\nStopped during {split}/{setup_id}. Re-run to resume (completed setups are skipped).")
            break

    if not new_rows:
        return base if base is not None else pl.DataFrame([])
    return _flush(output_path, base, new_rows)


def _run_batch(data, data_root, setup_jobs, base, existing, client, method, model_version,
               prompt_version, reasoning, prior_correction, effort, data_path, output_path,
               resume, no_wait, poll_interval, collect_only=False):
    """Submit each queued setup's requests to the Message Batches API (recording batch ids to the
    sidecar as they are created), then -- unless --no-wait -- poll each setup's batches to
    completion, collect the results, score, and flush its row.

    `collect_only` skips submission entirely: no batch is ever created, so there is zero risk of
    resubmitting. It only collects setups whose batch ids are already on record in the sidecar; a
    queued setup with nothing recorded is warned and skipped. Use it to gather outstanding results
    safely."""
    sidecar_path = _sidecar_path(output_path)
    sidecar = _load_sidecar(sidecar_path)
    config_state = sidecar.setdefault(
        _config_key(method, model_version, prompt_version, reasoning, prior_correction, effort), {}
    )

    # Phase A -- submit: ensure every queued setup has batches. Persist ids immediately so a crash
    # (or --no-wait) never loses or re-submits work already running server-side. Skipped entirely
    # under collect_only, which never creates a batch.
    if not collect_only:
        for split, setup_id in setup_jobs:
            key = f"{split}/{setup_id}"
            if not resume:
                config_state.pop(key, None)  # --no-resume: forget any prior submission and resubmit
            if key in config_state:
                continue
            df_setup, corner_map = _load_setup(data, split, setup_id)
            requests = list(_setup_requests(
                method, df_setup, data_root, setup_id, corner_map,
                model_version, prompt_version, reasoning, effort,
            ))
            batch_ids = [_create_batch(client, chunk) for chunk in _chunk_requests(requests)]
            config_state[key] = {"batch_ids": batch_ids, "n_requests": len(requests)}
            _save_sidecar(sidecar_path, sidecar)
            print(f"submitted {key}: {len(requests)} request(s) in {len(batch_ids)} batch(es)")

    if no_wait and not collect_only:
        print("submitted; re-run without --no-wait to poll and collect once the batches finish.")
        return existing if existing is not None else pl.DataFrame([])

    # Phase B/C -- await each setup's batches, then collect + score + flush. Awaiting per setup in
    # order is fine: the batches run in parallel server-side, so later setups are usually already
    # done by the time their turn comes.
    new_rows = []
    try:
        for split, setup_id in tqdm(setup_jobs, desc=f"collect {method}", unit="setup"):
            entry = config_state.get(f"{split}/{setup_id}")
            if entry is None:
                # No batch on record -- only reachable under collect_only (submission never runs).
                print(f"skip {split}/{setup_id}: no submitted batch on record")
                continue
            batch_ids = entry["batch_ids"]
            _await_batches(client, batch_ids, poll_interval)
            parsed_by_cid = _collect_batches(client, batch_ids)

            df_setup, _ = _load_setup(data, split, setup_id)
            metrics = defaultdict(list)
            board_ids = sorted(df_setup["image_id"].unique().to_list())
            for board_id in board_ids:
                record = _score_batched_board(method, board_id, df_setup, parsed_by_cid, model_version)
                for name, value in record.items():
                    metrics[name].append(value)

            if metrics["correct_board"]:  # skip a setup with no evaluable boards
                new_rows.append(_setup_row(
                    method, model_version, prompt_version, reasoning, prior_correction, effort,
                    data_path, split, setup_id, metrics,
                ))
                _flush(output_path, base, new_rows)
    except (anthropic.APIError, KeyboardInterrupt) as exc:
        print(f"\nStopped while collecting: {type(exc).__name__}: {exc}")
        print("Re-run to resume: submitted batches are durable and completed setups are skipped.")

    if not new_rows:
        return base if base is not None else pl.DataFrame([])
    return _flush(output_path, base, new_rows)


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
    effort: str = "medium",
    no_wait: bool = False,
    poll_interval: int = 30,
    collect_only: bool = False,
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

    The VLM methods run through the Message Batches API (see the module docstring), so runs are
    resumable and scopeable:
    - `splits`: which of "val"/"test" to score (both by default).
    - `resume` (default True): skip (split, setup) already present in `output_path` for THIS run's
      configuration; re-run after an interruption and it continues where it stopped. `resume=False`
      regenerates and replaces those rows instead (and resubmits their batches).
    - `max_setups`: cap the number of *not-yet-done* setups generated this run (re-run to do more).
    - `effort` ("low".."max", default "medium"): thinking depth for `reasoning="thinking"` -- lower
      it to stop thinking from eating the whole token budget (and truncating the answer) and to cut
      cost. It is a config dimension (recorded per row, part of resume identity) only in thinking
      mode; other modes ignore it and record the API default "high".
    - `no_wait`: for the VLM methods, submit the batches and return without polling; a later re-run
      collects them. `poll_interval`: seconds between batch status checks while waiting.
    - `collect_only`: never submit -- only gather results for setups already recorded in the
      sidecar (a setup with nothing recorded is warned and skipped). Zero risk of resubmission.

    Progress is flushed to `output_path` after every completed setup, so an interruption keeps
    everything so far. The unit is one setup.
    """
    assert method in METHODS, f"method must be one of {METHODS}"
    assert set(splits) <= {"val", "test"}, f"splits must be a subset of val/test; got {splits}"
    assert effort in EFFORT_LEVELS, f"effort must be one of {EFFORT_LEVELS}; got {effort}"
    # `effort` is only sent in thinking mode (see vision._build_message_params), so a non-thinking
    # run genuinely executes at the API default "high" -- record it as such so the requested flag
    # never fragments the config identity of runs whose requests it didn't change.
    effort = effort if reasoning == "thinking" else "high"
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
    client = None
    if method == "cnn":
        cnn_estimator = BoardEstimator(
            model_type="CNN",
            model_weights_path=model_weights_path,
            # Top-left corner is overwritten per setup below; any setup's metadata seeds it.
            calibration_metadata_path=Path(data["calibration_metadata_path"][0]),
            prior_correction=prior_correction,
        )
    else:  # square_label / square_logits / move / board / fen_whole -> Message Batches API
        client = anthropic.Anthropic()

    # --- Resume / overwrite bookkeeping ---------------------------------------------------------
    existing = pl.read_csv(output_path) if output_path.exists() else None
    if existing is not None and "effort" not in existing.columns:
        # Rows written before `effort` existed ran at the API default -- backfill so they keep
        # matching (and resume-skipping) their configuration instead of looking foreign.
        existing = existing.with_columns(pl.lit("high").alias("effort"))
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
            existing, method, model_version, prompt_version, reasoning, prior_correction, effort
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
                _config_expr(method, model_version, prompt_version, reasoning, prior_correction, effort)
                & pl.concat_str([pl.col("split"), pl.lit("\x00"), pl.col("setup_id")]).is_in(regen_keys)
            )
        )

    if not setup_jobs:
        print("nothing to do (all requested setups already generated, or max_setups=0)")
        return existing if existing is not None else pl.DataFrame([])

    if method == "cnn":
        return _run_cnn(
            data, data_root, setup_jobs, base, cnn_estimator, method, model_version,
            prompt_version, reasoning, prior_correction, effort, data_path, output_path,
        )
    return _run_batch(
        data, data_root, setup_jobs, base, existing, client, method, model_version,
        prompt_version, reasoning, prior_correction, effort, data_path, output_path,
        resume, no_wait, poll_interval, collect_only,
    )


if __name__ == "__main__":
    # Score one board-reading method over the val/test positions and append the per-(split, setup)
    # rows to the output CSV. One run = one method + config.
    #
    #   # the trained CNN (model_version defaults to the weights path); runs locally:
    #   uv run python -m chess_commentator.model.evaluation cnn \
    #       --model-weights-path weights/model_state_dict.safetensors
    #
    #   # the strongest VLM, predicting the move from the after-image, reasoning out loud first.
    #   # This submits a Message Batch and polls until it finishes, then scores:
    #   uv run python -m chess_commentator.model.evaluation move --reasoning text
    #
    #   # generate the next 3 val setups only; safe to re-run (skips done setups) after a crash:
    #   uv run python -m chess_commentator.model.evaluation move --splits val --max-setups 3
    #
    #   # big run: submit now and come back later -- the first call returns after submitting,
    #   # the second collects whatever has finished:
    #   uv run python -m chess_commentator.model.evaluation square_logits --no-wait
    #   uv run python -m chess_commentator.model.evaluation square_logits          # collect
    #
    #   # collect outstanding results only, never submitting (zero resubmission risk):
    #   uv run python -m chess_commentator.model.evaluation square_logits --splits val --collect-only
    #
    # The VLM methods (square_label, square_logits, move, board, fen_whole) make billed Claude API
    # calls (via the Message Batches API, 50% off) and need ANTHROPIC_API_KEY. Runs are resumable:
    # batch ids are recorded to <output>.batches.json, progress is flushed after every setup, and
    # completed setups are skipped on re-run (pass --no-resume to regenerate).
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
        help="LLM reasoning mode before the answer (see vision._build_message_params).",
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
    parser.add_argument(
        "--effort", choices=list(EFFORT_LEVELS), default="medium",
        help="Thinking depth for --reasoning thinking (lower = shorter/cheaper thinking, fewer "
        "max_tokens truncations). Ignored by other modes (recorded as the API default 'high').",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="VLM methods: submit the batches and exit without polling; re-run to collect.",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="Never submit -- only collect results already recorded in the sidecar (skips setups "
        "with no recorded batch). Zero resubmission risk.",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30,
        help="VLM methods: seconds between batch status checks while waiting (default 30).",
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
        effort=args.effort,
        no_wait=args.no_wait,
        poll_interval=args.poll_interval,
        collect_only=args.collect_only,
        data_path=args.data_path,
        output_path=args.output_path,
    )
