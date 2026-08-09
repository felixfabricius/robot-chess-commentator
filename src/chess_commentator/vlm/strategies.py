"""The whole-image reading strategies, the pure board/move parsing helpers they rely on, and the
batch-request builders that let the benchmark harness run all of it through the Message Batches API.

Three whole-image strategies, coarser than the per-square backends in `perception.board_estimator`
and able to hallucinate an illegal result (which `first_legal_and_stats` absorbs):

- `estimate_move_llm` (i):        position before + photo after -> ordered candidate moves (UCI).
- `estimate_board_after_llm` (ii): same inputs -> ordered candidate positions (FEN board strings),
                                   from which an implied legal move is derived.
- `estimate_fen_whole_llm`:        the crude baseline -- one call, whole board, straight to FEN.

`infer_fen_from_image` is the pre-structured-output ancestor of the last one, kept only so old
results stay reproducible.

The `build_*_params` functions and `parsed_to_square_estimate` are the batch counterparts of the
synchronous calls: the harness builds every request up front and interprets the answers hours
later, so construction and interpretation are factored out to be shared. Both paths therefore send
identical requests and read answers identically -- only the transport differs.
"""
import math
from pathlib import Path

import anthropic
import chess

from chess_commentator.board import PIECES, SQUARES, SquareEstimate
from chess_commentator.vlm.client import (
    build_message_params,
    call_claude,
    encode_image_base64,
    infer_media_type,
)
from chess_commentator.vlm.prompts import (
    BOARD_SCHEMA,
    FEN_WHOLE_SCHEMA,
    MOVE_SCHEMA,
    PROMPTS,
    SQUARE_LABEL_SCHEMA,
    SQUARE_LOGITS_SCHEMA,
    SQUARE_MAX_TOKENS,
    collect_slots,
    orientation_sentence,
    previous_position_prompt,
)


def infer_fen_from_image(image_path: Path, model: str = "claude-sonnet-5", prompt_version: int = 1) -> str:
    client = anthropic.Anthropic()

    prompt = PROMPTS["fen_whole"][prompt_version]

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": infer_media_type(image_path),
                            "data": encode_image_base64(image_path),
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )

    return message.content[0].text


# --- Board / move parsing helpers (pure, no API; unit-tested) -------------------------------

def board_to_labels(board: chess.Board) -> dict[str, str]:
    """Map every square name to its 13-way label ("empty" or a python-chess piece symbol)."""
    labels = {}
    for sq in SQUARES:
        piece = board.piece_at(chess.parse_square(sq))
        labels[sq] = piece.symbol() if piece is not None else "empty"
    return labels


def fen_board_to_labels(fen_board: str) -> dict[str, str] | None:
    """Parse a piece-placement FEN (a full FEN is accepted; only its first field is used) into a
    64-square label map. Returns None if it is malformed."""
    if not isinstance(fen_board, str) or not fen_board.strip():
        return None
    placement = fen_board.strip().split()[0]
    board = chess.Board.empty()
    try:
        board.set_board_fen(placement)
    except ValueError:
        return None
    return board_to_labels(board)


def _parse_move(board: chess.Board, text: str) -> str | None:
    """Best-effort parse of a candidate move string into a legal-move UCI (accepts UCI and SAN),
    else None. The python-chess parse errors all subclass ValueError."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    for parse in (board.parse_uci, board.parse_san):
        try:
            move = parse(text)
        except ValueError:
            continue
        if move in board.legal_moves:
            return move.uci()
    return None


def implied_move(previous_fen: str, predicted_labels: dict[str, str]) -> str | None:
    """The legal move from `previous_fen` whose resulting piece placement equals
    `predicted_labels`, or None if no legal move reproduces that placement (i.e. the predicted
    board is not reachable in one legal move)."""
    board = chess.Board(previous_fen)
    for move in board.legal_moves:
        board.push(move)
        matches = board_to_labels(board) == predicted_labels
        board.pop()
        if matches:
            return move.uci()
    return None


def first_legal_and_stats(previous_fen: str, candidates: list[str], kind: str) -> dict:
    """Reduce an ordered candidate list from a whole-image VLM to the commentator's answer plus
    legality stats. `kind="move"` treats each candidate as a UCI/SAN move; `kind="board"` treats
    each as a FEN board string whose implied legal move is derived.

    Returns:
        first_legal: the first candidate implying a legal move (its UCI), or None.
        ordered_legal: the legal implied moves in candidate order, de-duplicated (for board_rank).
        first_output_illegal: whether the model's FIRST candidate was illegal (None if the list
            was empty).
        none_legal: whether not a single candidate was legal.
    """
    assert kind in ("move", "board")
    board = chess.Board(previous_fen)

    ordered_legal: list[str] = []
    first_output_illegal = None
    for idx, candidate in enumerate(candidates):
        if kind == "move":
            move_uci = _parse_move(board, candidate)
        else:
            labels = fen_board_to_labels(candidate)
            move_uci = implied_move(previous_fen, labels) if labels is not None else None
        if idx == 0:
            first_output_illegal = move_uci is None
        if move_uci is not None and move_uci not in ordered_legal:
            ordered_legal.append(move_uci)

    return {
        "first_legal": ordered_legal[0] if ordered_legal else None,
        "ordered_legal": ordered_legal,
        "first_output_illegal": first_output_illegal,
        "none_legal": len(ordered_legal) == 0,
    }


# --- Whole-image VLM strategies (methods i and ii) ------------------------------------------

def estimate_move_llm(client, warped_image_path, previous_fen, corner_map,
                      model="claude-sonnet-5", prompt_version=1, reasoning="none", max_tokens=1024):
    """Method i: given the position before the move and the warped image after, return an ordered
    list of candidate moves (UCI), best guess first. Pair with first_legal_and_stats(kind="move")
    to get the prediction and legality stats. Output is bounded (fixed move_1..move_N slots), so a
    default max_tokens of 1024 is ample."""
    prompt = previous_position_prompt(PROMPTS["move"][prompt_version], previous_fen, corner_map)
    parsed, usage, elapsed = call_claude(
        client, model, [warped_image_path], prompt, MOVE_SCHEMA,
        reasoning=reasoning, max_tokens=max_tokens,
    )
    return {
        "moves": collect_slots(parsed, "move"),
        "reasoning": parsed.get("reasoning"),
        "usage": usage,
        "elapsed": elapsed,
    }


def estimate_board_after_llm(client, warped_image_path, previous_fen, corner_map,
                             model="claude-sonnet-5", prompt_version=1, reasoning="none", max_tokens=1024):
    """Method ii: given the position before the move and the warped image after, return an ordered
    list of candidate positions after the move (FEN board strings), best guess first. Pair with
    first_legal_and_stats(kind="board"). Output is bounded (fixed board_1..board_N slots)."""
    prompt = previous_position_prompt(PROMPTS["board"][prompt_version], previous_fen, corner_map)
    parsed, usage, elapsed = call_claude(
        client, model, [warped_image_path], prompt, BOARD_SCHEMA,
        reasoning=reasoning, max_tokens=max_tokens,
    )
    return {
        "fen_boards": collect_slots(parsed, "board"),
        "reasoning": parsed.get("reasoning"),
        "usage": usage,
        "elapsed": elapsed,
    }


def estimate_fen_whole_llm(client, image_path, corner_map, model="claude-sonnet-5", prompt_version=1,
                           reasoning="none", max_tokens=1024):
    """The crude whole-board baseline: one call reading the entire board straight to a FEN board
    string (no previous position). Structured-output twin of `infer_fen_from_image`, but reporting
    token usage and timing so it can be compared on cost like the other strategies. `corner_map`
    (the setup's camera_natural_orientation order) tells the model which real square sits at each
    photo corner, so it can emit the FEN in canonical a8-top-left orientation."""
    prompt = PROMPTS["fen_whole"][prompt_version].format(orientation=orientation_sentence(corner_map))
    parsed, usage, elapsed = call_claude(
        client, model, [image_path], prompt, FEN_WHOLE_SCHEMA,
        reasoning=reasoning, max_tokens=max_tokens,
    )
    return {
        "fen_board": str(parsed.get("fen_board", "")).strip(),
        "reasoning": parsed.get("reasoning"),
        "usage": usage,
        "elapsed": elapsed,
    }


# --- Batch-API building blocks: build a request / turn a parsed answer into an estimate ----------
# The batch harness (benchmark/harness.py) builds every request up front and scores the responses
# minutes-to-hours later, so request construction and answer interpretation are factored out here
# to be shared with the synchronous path -- both send identical requests and interpret answers the
# same way; only the transport (live call vs Message Batches) differs.

def parsed_to_square_estimate(llm_method, parsed, image_path=None) -> SquareEstimate:
    """Turn one square-classification answer into a SquareEstimate of logit-like scores, exactly as
    the synchronous `estimate_square` does. square_label -> a one-hot (chosen label 1.0, rest 0),
    which under estimate_move's CrossEntropyLoss ranks moves the same as the old squared-error path.
    square_logits -> the per-label scores normalised into a distribution and stored as its log, so
    the values are log-probabilities like the CNN's (a drop-in for estimate_move); a degenerate
    all-zero/negative response falls back to a uniform prior."""
    square_estimate = SquareEstimate(image_path=image_path, copied=False, copied_from=None)
    if llm_method == "square_label":
        setattr(square_estimate, parsed["label"], 1.0)
    else:  # square_logits (iv)
        scores = {label: max(float(parsed["scores"].get(label, 0.0)), 0.0) for label in PIECES}
        total = sum(scores.values())
        for label in PIECES:
            prob = (scores[label] / total) if total > 0 else (1.0 / len(PIECES))
            setattr(square_estimate, label, math.log(prob) if prob > 0 else -1e9)
    return square_estimate


def build_square_params(llm_method, annotated_image_path, model, prompt_version=1, reasoning="none", effort="medium"):
    """Request params for one square classification (method iii/iv). `annotated_image_path` is the
    marked-up crop the square prompt refers to (the `_annotated` PNG)."""
    schema = SQUARE_LABEL_SCHEMA if llm_method == "square_label" else SQUARE_LOGITS_SCHEMA
    return build_message_params(
        model, [annotated_image_path], PROMPTS[llm_method][prompt_version], schema,
        reasoning=reasoning, max_tokens=SQUARE_MAX_TOKENS[llm_method], effort=effort,
    )


def build_move_params(warped_image_path, previous_fen, corner_map, model, prompt_version=1, reasoning="none", effort="medium"):
    """Request params for method i (ordered candidate moves). Score the response with
    `collect_slots(parsed, "move")` + `first_legal_and_stats(kind="move")`."""
    prompt = previous_position_prompt(PROMPTS["move"][prompt_version], previous_fen, corner_map)
    return build_message_params(model, [warped_image_path], prompt, MOVE_SCHEMA, reasoning=reasoning, effort=effort)


def build_board_params(warped_image_path, previous_fen, corner_map, model, prompt_version=1, reasoning="none", effort="medium"):
    """Request params for method ii (ordered candidate positions). Score with
    `collect_slots(parsed, "board")` + `first_legal_and_stats(kind="board")`."""
    prompt = previous_position_prompt(PROMPTS["board"][prompt_version], previous_fen, corner_map)
    return build_message_params(model, [warped_image_path], prompt, BOARD_SCHEMA, reasoning=reasoning, effort=effort)


def build_fen_whole_params(image_path, corner_map, model, prompt_version=1, reasoning="none", effort="medium"):
    """Request params for the whole-board FEN baseline. Score with `parsed["fen_board"]`."""
    prompt = PROMPTS["fen_whole"][prompt_version].format(orientation=orientation_sentence(corner_map))
    return build_message_params(model, [image_path], prompt, FEN_WHOLE_SCHEMA, reasoning=reasoning, effort=effort)
