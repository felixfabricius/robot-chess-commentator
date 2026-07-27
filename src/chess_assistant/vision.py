"""
Reading a chess move off the camera image, several ways, so the CNN can be benchmarked against
the strongest VLM approach.

`BoardEstimator` is the square-level path: it runs over the per-square cutouts produced by
image_processing.cutout() and fills a `BoardEstimate` (64 `SquareEstimate`s, each holding a score
for all 13 labels: the 12 pieces plus "empty"). Three square-level backends:
- "CNN" (production): the trained SquareClassifierMultiHead. One forward pass per square; the
  three heads are recombined into 13-way log-probabilities.
- "LLM" with llm_method="square_label" (iii): one Claude call per square returning a hard label
  (a one-hot reading).
- "LLM" with llm_method="square_logits" (iv): one Claude call per square returning a score for all
  13 labels, converted to log-probabilities.
The square-level backends never pick a move; they emit logit-like scores that game.estimate_move()
scores every legal move against, so a square the model is unsure about costs a candidate move some
likelihood rather than corrupting the position outright.

Two whole-image VLM strategies work at a coarser granularity and can hallucinate an illegal
result (handled by `first_legal_and_stats`):
- `estimate_move_llm` (i): given the position before the move + the warped image after, return an
  ordered list of candidate moves (UCI).
- `estimate_board_after_llm` (ii): same inputs, return an ordered list of candidate positions
  after the move (FEN board strings), from which an implied legal move is derived.

`infer_fen_from_image` is the older, cruder baseline still kept for comparison: one LLM call for
the whole board, returning a FEN board string directly.

All Claude calls (except the legacy `infer_fen_from_image`) go through `_call_claude`, which uses
structured outputs so parsing is robust, and honours a `reasoning` knob: "none" (answer only),
"thinking" (adaptive thinking, hidden) or "text" (a visible reasoning field before the answer).
"""
import base64
import json
import math
import time
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, make_dataclass
import anthropic
import chess
import torch
from torchvision import tv_tensors
from safetensors.torch import load_file

import numpy as np


from omegaconf import OmegaConf, DictConfig

from chess_assistant.config import SQUARES, PIECES
from chess_assistant.model.config import TARGET_MAP, reconstruct_13way_logprobs, TOP_LEFT_OHE_MAP
from chess_assistant.model.data import EVAL_TRANSFORM
from chess_assistant.model.model import SquareClassifierMultiHead

load_dotenv()

# Prompts are keyed by method then version, so each strategy can be tuned independently and a
# `prompt_version` recorded alongside its results. The `{...}` placeholders are filled per board
# by the move/board strategies (previous position + image orientation); the square prompts need no
# substitution. A trailing reasoning instruction is appended by `_call_claude` when reasoning
# == "text", so the prompts here describe only the task and the answer format.

# Shared by the two square-level strategies (iii, iv) so they see identical visual guidance and a
# iii-vs-iv comparison reflects the label-vs-scores format, not prompt wording. Describes exactly
# what the annotated crop (image_processing._cutout_v2 / _cutout_global) draws: red dots on the
# target square's base corners, and -- on v2 crops -- a green convex-hull outline of the column of
# space above the square.
_SQUARE_TASK_PREAMBLE = (
    "You are classifying one square from a chessboard image.\n"
    "The target square's base corners are marked with red dots. Green lines, when shown, outline "
    "the convex hull of the space above the square -- the column a piece standing there occupies "
    "as it leans toward the camera -- which helps you tell this square's piece from neighbours "
    "leaning in.\n"
    "Other pieces and neighbouring squares may be visible because the crop includes padding.\n"
    "Classify only the chess piece whose BASE sits on the highlighted target square; ignore all "
    "other visible pieces. The square is empty unless a piece's base is on it, even if a "
    "neighbouring piece overhangs it.\n"
)

PROMPTS = {
    "fen_whole": {
        0: (
            "You are looking at a physical chess board."
            "Return only the board position as a FEN board string, "
            "not the full FEN. Example format (if in starting position): "
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR. "
            "Do not include side to move, castling rights, move counters, or explanation."
            "CAREFULLY inspect each of the 64 squares individually to identify which piece - if any - "
            "is located there."
        ),
    },
    "square_label": {
        1: (
            _SQUARE_TASK_PREAMBLE
            + "Return exactly one label from:\n"
            "empty, K, Q, R, B, N, P, k, q, r, b, n, p,\n"
            "where the letter is the piece in FEN notation (e.g. K is the white king; uppercase = "
            "White, lowercase = Black)."
        ),
    },
    "square_logits": {
        1: (
            _SQUARE_TASK_PREAMBLE
            + "Return a score from 0 to 100 for EVERY one of the 13 labels (empty, K, Q, R, B, N, "
            "P, k, q, r, b, n, p), reflecting how likely that label is for the target square. "
            "Uppercase letters are White pieces, lowercase are Black, in FEN notation. Give higher "
            "scores to more likely labels; they need not sum to 100."
        ),
    },
    "move": {
        1: (
            "You are a chess move detector. You are given the position BEFORE a move and a "
            "top-down (rectified) photo of the physical board AFTER exactly one legal move was made.\n"
            "Position before the move (piece-placement FEN, standard orientation with a8 at the top-left):\n"
            "{fen_board}\n"
            "{ascii_board}\n"
            "{orientation}\n"
            "Identify the single legal move that was played. Return an ordered list of candidate moves "
            "in UCI notation (e.g. 'e2e4', or 'e7e8q' for a promotion), your best guess first. "
            "If you are not certain your first guess is correct, append additional plausible moves in "
            "decreasing order of likelihood; return as many as you need."
        ),
    },
    "board": {
        1: (
            "You are a chess position reader. You are given the position BEFORE a move and a "
            "top-down (rectified) photo of the physical board AFTER exactly one legal move was made.\n"
            "Position before the move (piece-placement FEN, standard orientation with a8 at the top-left):\n"
            "{fen_board}\n"
            "{ascii_board}\n"
            "{orientation}\n"
            "Report the position AFTER the move. Return an ordered list of candidate positions, each as a "
            "piece-placement FEN string in standard orientation (a8 top-left), e.g. "
            "'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', your best guess first. "
            "If you are not certain your first guess is correct, append additional plausible positions in "
            "decreasing order of likelihood; return as many as you need."
        ),
    },
}

# Structured-output json_schemas, one per method. Structured outputs guarantee the response is a
# JSON object matching the schema, so parsing never has to scrape free-form text. (json_schema
# does not support numeric/length/array-size constraints, so "return at least one, best first" is
# expressed in the prompt, not the schema.)
_SQUARE_LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(PIECES)}},
    "required": ["label"],
    "additionalProperties": False,
}
_SQUARE_LOGITS_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {label: {"type": "number"} for label in PIECES},
            "required": list(PIECES),
            "additionalProperties": False,
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}
_MOVE_SCHEMA = {
    "type": "object",
    "properties": {"moves": {"type": "array", "items": {"type": "string"}}},
    "required": ["moves"],
    "additionalProperties": False,
}
_BOARD_SCHEMA = {
    "type": "object",
    "properties": {"fen_boards": {"type": "array", "items": {"type": "string"}}},
    "required": ["fen_boards"],
    "additionalProperties": False,
}
_FEN_WHOLE_SCHEMA = {
    "type": "object",
    "properties": {"fen_board": {"type": "string"}},
    "required": ["fen_board"],
    "additionalProperties": False,
}


@dataclass
class SquareEstimate:
    image_path: Path | None = None
    copied: bool = False
    copied_from: Path | None = None
    K: float = 0
    Q: float = 0
    R: float = 0
    B: float = 0
    N: float = 0
    P: float = 0
    k: float = 0
    q: float = 0
    r: float = 0
    b: float = 0
    n: float = 0
    p: float = 0
    empty: float = 0

BoardEstimate = make_dataclass(
    "BoardEstimate",
    [(square, SquareEstimate | None, None) for square in SQUARES]
)

def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def infer_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    if suffix not in types:
        raise ValueError(f"Unsupported image type: {suffix}")
    return types[suffix]


def _image_block(image_path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": infer_media_type(image_path),
            "data": encode_image_base64(image_path),
        },
    }


def _with_reasoning_field(schema: dict) -> dict:
    """A copy of `schema` with a leading `reasoning` string property (also required), so a
    reasoning="text" call writes its chain of thought into the structured output before the answer
    fields."""
    properties = {"reasoning": {"type": "string", "description": "Step-by-step reasoning, written before the answer."}}
    properties.update(schema.get("properties", {}))
    return {
        "type": "object",
        "properties": properties,
        "required": ["reasoning"] + list(schema.get("required", [])),
        "additionalProperties": False,
    }


def _call_claude(client, model, image_paths, prompt, answer_schema, reasoning="none", max_tokens=1024):
    """One Claude call with a structured (json_schema) answer. Returns (parsed, usage, elapsed).

    `parsed` is the JSON object matching `answer_schema` (plus a `reasoning` key when
    reasoning == "text"). `usage` is the response's token usage (for cost); `elapsed` is wall-clock
    seconds for this single call (for timing). `reasoning`:
      - "none":     answer only.
      - "text":     a visible `reasoning` field is added to the schema and filled before the answer.
      - "thinking": adaptive thinking (hidden), with the same structured final answer.
    """
    assert reasoning in ("none", "text", "thinking")
    schema = _with_reasoning_field(answer_schema) if reasoning == "text" else answer_schema
    if reasoning == "text":
        prompt = prompt + "\nFirst reason step by step in the `reasoning` field, then fill in the answer."

    content = [_image_block(p) for p in image_paths] + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    output_config = {"format": {"type": "json_schema", "schema": schema}}

    start = time.monotonic()
    if reasoning == "thinking":
        # Thinking tokens count against max_tokens, so give generous headroom and stream to avoid
        # the SDK's non-streaming timeout guard at large max_tokens.
        with client.messages.stream(
            model=model,
            max_tokens=max(max_tokens, 8192),
            thinking={"type": "adaptive"},
            output_config=output_config,
            messages=messages,
        ) as stream:
            message = stream.get_final_message()
    else:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            output_config=output_config,
            messages=messages,
        )
    elapsed = time.monotonic() - start

    # output_config json_schema guarantees the first text block is JSON valid against the schema.
    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text), message.usage, elapsed


def infer_fen_from_image(image_path: Path, model: str = "claude-opus-4-8", prompt_version: int = 0) -> str:
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

def _orientation_sentence(corner_map: dict) -> str:
    """Describe the image's orientation from the setup's corner_map so the model can read the
    photo in its own frame yet still answer in canonical (a8-top-left) coordinates."""
    return (
        "In the photo, the square at the top-left is {tl}, top-right is {tr}, "
        "bottom-right is {br}, bottom-left is {bl}."
    ).format(**{k: corner_map[k] for k in ("tl", "tr", "br", "bl")})


def _previous_position_prompt(prompt_template: str, previous_fen: str, corner_map: dict) -> str:
    board = chess.Board(previous_fen)
    return prompt_template.format(
        fen_board=board.board_fen(),
        ascii_board=str(board),
        orientation=_orientation_sentence(corner_map),
    )


def estimate_move_llm(client, warped_image_path, previous_fen, corner_map,
                      model="claude-opus-4-8", prompt_version=1, reasoning="none", max_tokens=2048):
    """Method i: given the position before the move and the warped image after, return an ordered
    list of candidate moves (UCI), best guess first. Pair with first_legal_and_stats(kind="move")
    to get the prediction and legality stats."""
    prompt = _previous_position_prompt(PROMPTS["move"][prompt_version], previous_fen, corner_map)
    parsed, usage, elapsed = _call_claude(
        client, model, [warped_image_path], prompt, _MOVE_SCHEMA,
        reasoning=reasoning, max_tokens=max_tokens,
    )
    return {
        "moves": [str(m).strip() for m in parsed.get("moves", [])],
        "reasoning": parsed.get("reasoning"),
        "usage": usage,
        "elapsed": elapsed,
    }


def estimate_board_after_llm(client, warped_image_path, previous_fen, corner_map,
                             model="claude-opus-4-8", prompt_version=1, reasoning="none", max_tokens=2048):
    """Method ii: given the position before the move and the warped image after, return an ordered
    list of candidate positions after the move (FEN board strings), best guess first. Pair with
    first_legal_and_stats(kind="board")."""
    prompt = _previous_position_prompt(PROMPTS["board"][prompt_version], previous_fen, corner_map)
    parsed, usage, elapsed = _call_claude(
        client, model, [warped_image_path], prompt, _BOARD_SCHEMA,
        reasoning=reasoning, max_tokens=max_tokens,
    )
    return {
        "fen_boards": [str(b).strip() for b in parsed.get("fen_boards", [])],
        "reasoning": parsed.get("reasoning"),
        "usage": usage,
        "elapsed": elapsed,
    }


def estimate_fen_whole_llm(client, image_path, model="claude-opus-4-8", prompt_version=0,
                           reasoning="none", max_tokens=1024):
    """The crude whole-board baseline: one call reading the entire board straight to a FEN board
    string (no previous position). Structured-output twin of `infer_fen_from_image`, but reporting
    token usage and timing so it can be compared on cost like the other strategies."""
    parsed, usage, elapsed = _call_claude(
        client, model, [image_path], PROMPTS["fen_whole"][prompt_version], _FEN_WHOLE_SCHEMA,
        reasoning=reasoning, max_tokens=max_tokens,
    )
    return {
        "fen_board": str(parsed.get("fen_board", "")).strip(),
        "reasoning": parsed.get("reasoning"),
        "usage": usage,
        "elapsed": elapsed,
    }


class BoardEstimator:
    def __init__(self, model_type: str = "CNN", config: DictConfig | None = None, calibration_metadata_path: Path | None = None, model_weights_path = None, device = None, model = None, prior_correction: bool = True,
                 model_version: str | None = None, prompt_version: int | None = None, llm_method: str = "square_label", reasoning: str = "none"):
        """
        Build the estimator and hold the board estimate it keeps overwriting, one board
        position per call to estimate_board().

        Args:
            model_type: "CNN" for the trained classifier, "LLM" for a per-square Claude backend.
            config: the loaded config.yaml. Only read by the LLM path, and only as a fallback for
                model_version / prompt_version; the CNN path takes its weights explicitly.
            calibration_metadata_path: calibration_metadata.json of the setup the squares
                come from. CNN only, and required: it carries the one piece of metadata the
                model is fed alongside the image (which board corner is top-left).
            model_weights_path: safetensors checkpoint to load into a fresh
                SquareClassifierMultiHead. CNN only.
            device: "cpu" or "cuda" (default "cpu"). CNN only.
            model: an already-constructed model, used instead of model_weights_path. This is
                how evaluation and the tests hand in a model they have in memory, without a
                round trip through disk.
            prior_correction: subtract the model's training log-prior from the 13-way
                log-probabilities (Bayesian prior correction; see reconstruct_13way_logprobs).
                CNN only. Defaults to True, which is what live inference wants: the robot should
                always use the best available decision rule, and the buffer is all-zeros whenever
                the prior is unknown (legacy checkpoints), where subtracting it is a no-op. Eval
                passes this explicitly instead, so a run can be scored both ways.
            model_version, prompt_version: LLM only. Override the Claude model id and prompt
                version directly (so evaluation can sweep them); fall back to `config.vision` then
                to defaults when not given.
            llm_method: LLM only. "square_label" (iii; a hard one-hot label) or "square_logits"
                (iv; a score per label, stored as log-probabilities).
            reasoning: LLM only. "none" / "text" / "thinking" (see _call_claude). Defaults to
                "none" because a square method makes 64 calls per board.
        """
        assert model_type in ["CNN", "LLM"]
        self.board_estimate = BoardEstimate()
        if model_type == "LLM":
            assert llm_method in ("square_label", "square_logits")
            cfg_vision = config.vision if config is not None else {}
            self.model_version = model_version or cfg_vision.get("model_version", "claude-opus-4-8")
            self.prompt_version = prompt_version if prompt_version is not None else cfg_vision.get("prompt_version", 1)
            self.llm_method = llm_method
            self.reasoning = reasoning
            self.client = anthropic.Anthropic()
        else:
            assert calibration_metadata_path is not None
            assert model_weights_path is not None or model is not None
            if model is None:
                model = SquareClassifierMultiHead()
                state_dict = load_file(model_weights_path, device="cpu")
                # strict=False tolerates exactly one thing: weights saved before `log_prior` (the
                # Bayesian prior-correction buffer) existed, which simply leaves it at all-zeros
                # = no correction. Everything else is still an error, so a genuinely mismatched
                # checkpoint cannot slip through silently.
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                assert not unexpected, f"Unexpected keys in {model_weights_path}: {unexpected}"
                assert set(missing) <= {"log_prior"}, f"Missing keys in {model_weights_path}: {missing}"
            assert device in ["cpu", "cuda", None, torch.device("cpu"), torch.device("cuda")]
            self.device = torch.device(device) if device is not None else torch.device("cpu")
            with open(calibration_metadata_path, "r", encoding="utf-8") as f:
                calibration_metadata = json.load(f)
            # The model's only metadata: which board corner is top-left in the image.
            self.top_left_corner = calibration_metadata["camera_natural_orientation"]["order"]["tl"]
            model.eval()
            self.model = model.to(self.device)
            # Resolved once here rather than per square. getattr keeps this working for any model
            # handed in that predates the buffer; None means "no correction".
            self.log_prior = getattr(model, "log_prior", None) if prior_correction else None
        self.model_type = model_type

    def estimate_square(self, image_path: Path) -> SquareEstimate:
        """
        Classify one square, given the path of its cutout (.../squares/e4/e4.png).

        The two backends read different files from that directory: the LLM gets the annotated
        PNG (the crop with red markers on the target square's corners, which the prompt refers
        to), the CNN gets the 4-channel masked .npy the crop was saved alongside.

        Returns a SquareEstimate whose 13 label fields hold logit-like scores. square_label gives
        a hard one-hot; square_logits gives log-probabilities; the CNN gives log-probabilities.

        LLM calls record their token usage and wall-clock time in `self.last_usage` /
        `self.last_elapsed` so the evaluation harness can accumulate cost and timing per board.
        """
        if self.model_type == "LLM":
            image_path = image_path.parent / (image_path.stem + "_annotated" + image_path.suffix)
            square_estimate = SquareEstimate(image_path=image_path, copied=False, copied_from=None)

            if self.llm_method == "square_label":
                parsed, usage, elapsed = _call_claude(
                    self.client, self.model_version, [image_path],
                    PROMPTS["square_label"][self.prompt_version], _SQUARE_LABEL_SCHEMA,
                    reasoning=self.reasoning, max_tokens=512,
                )
                # One-hot: the chosen label scores 1.0, the rest stay 0. Fed to estimate_move's
                # CrossEntropyLoss this ranks moves the same as the old squared-error path did.
                setattr(square_estimate, parsed["label"], 1.0)
            else:  # square_logits (iv)
                parsed, usage, elapsed = _call_claude(
                    self.client, self.model_version, [image_path],
                    PROMPTS["square_logits"][self.prompt_version], _SQUARE_LOGITS_SCHEMA,
                    reasoning=self.reasoning, max_tokens=1024,
                )
                # Normalise the per-label scores into a distribution and store its log, so the
                # values are log-probabilities like the CNN's (a drop-in for estimate_move). A
                # degenerate all-zero/negative response falls back to a uniform prior.
                scores = {label: max(float(parsed["scores"].get(label, 0.0)), 0.0) for label in PIECES}
                total = sum(scores.values())
                for label in PIECES:
                    prob = (scores[label] / total) if total > 0 else (1.0 / len(PIECES))
                    setattr(square_estimate, label, math.log(prob) if prob > 0 else -1e9)

            self.last_usage = usage
            self.last_elapsed = elapsed
            return square_estimate

        else:
            # TODO: the square name is recovered from the filename, so this breaks if the
            # cutout naming convention in image_processing.py ever changes.
            square = image_path.stem
            square_dir = image_path.parent

            # Metadata: one-hot of which board corner is top-left in the image. Must match
            # training (model/data.py) - both use TOP_LEFT_OHE_MAP.
            metadata = torch.zeros(1, 4, dtype=torch.float32)
            metadata[0, TOP_LEFT_OHE_MAP[self.top_left_corner]] = 1
            metadata = metadata.to(self.device)
            assert metadata.shape == (1, 4)

            # The image: RGB plus the square's mask as a 4th channel, transformed exactly as in
            # training (EVAL_TRANSFORM, i.e. no augmentation).
            image = np.load(square_dir / f"{square}_masked.npy")
            rgb = image[..., :3]
            mask = tv_tensors.Mask(image[..., 3])
            rgb = EVAL_TRANSFORM(rgb)
            mask = EVAL_TRANSFORM(mask).unsqueeze(dim=0)
            image = torch.cat([rgb, mask]).unsqueeze(dim=0).to(self.device)
            assert image.shape == (1, 4, 144, 144)
            
            # Whatever the model's output shape, what gets stored is logit-like values (raw
            # logits or reconstructed log-probabilities), which game.py feeds into its own
            # CrossEntropyLoss.
            square_estimate = SquareEstimate(
                image_path=image_path,
                copied=False,
                copied_from=None
            )

            # Multi-head model (model 3): recombine the three heads into 13-way
            # log-probabilities. softmax(log p) == p and the reconstructed probs sum to
            # 1, so log p is a drop-in for the old single-head logits under the softmax
            # game.py re-applies downstream.
            with torch.no_grad():
                logit_empty, logits_color, logits_type = self.model(image, metadata)
            logprobs = reconstruct_13way_logprobs(
                logit_empty.squeeze(0),
                logits_color.squeeze(0),
                logits_type.squeeze(0),
                log_prior=self.log_prior,
            )
            for label, idx in TARGET_MAP.items():
                setattr(square_estimate, label, logprobs[idx].item())
            return square_estimate

    def estimate_board(self, squares_dir):
        """
        Classify all 64 squares of one board image and return the resulting BoardEstimate.

        `squares_dir` is the directory image_processing.cutout() wrote, i.e. one subdirectory
        per square. The estimate is stored on the estimator (self.board_estimate) as well as
        returned; it is overwritten on every call, so callers must not hold on to it across
        board positions.

        For the LLM backends the 64 per-square calls' token usage and wall-clock time are summed
        into `self.board_input_tokens` / `self.board_output_tokens` / `self.board_elapsed`, so the
        evaluation harness can read a board's cost and timing after one estimate_board() call.
        """
        is_llm = self.model_type == "LLM"
        self.board_input_tokens = 0
        self.board_output_tokens = 0
        self.board_elapsed = 0.0
        for square in SQUARES:
            image_path = squares_dir / square / f"{square}.png"
            square_estimate = self.estimate_square(image_path)
            setattr(self.board_estimate, square, square_estimate)
            if is_llm:
                self.board_input_tokens += self.last_usage.input_tokens
                self.board_output_tokens += self.last_usage.output_tokens
                self.board_elapsed += self.last_elapsed

        return self.board_estimate


if __name__ == "__main__":
    # Debug tool: run the trained classifier over a squares directory and print what it thinks
    # is standing on each square. Useful when the board estimate disagrees with reality and you
    # want to see whether the model is confidently wrong or merely unsure.
    #
    #   uv run python -m chess_assistant.vision \
    #       data/generated/2026-07-01_175334/board_2026-07-01_175602/squares --squares a1 e4
    import argparse
    import math

    parser = argparse.ArgumentParser(description="Print the model's prediction for each square.")
    parser.add_argument(
        "squares_dir",
        type=Path,
        help="A .../<board>/squares directory, as written by image_processing.cutout().",
    )
    parser.add_argument(
        "--squares",
        nargs="+",
        default=SQUARES,
        metavar="SQUARE",
        help="Squares to classify, e.g. --squares a1 e4. Defaults to all 64.",
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--calibration-metadata",
        type=Path,
        default=None,
        help="Defaults to calibration_metadata.json in the setup dir two levels up from squares_dir.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=3, help="How many labels to print per square.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    calibration_metadata_path = (
        args.calibration_metadata
        if args.calibration_metadata is not None
        else args.squares_dir.parent.parent / "calibration_metadata.json"
    )

    board_estimator = BoardEstimator(
        "CNN",
        config,
        calibration_metadata_path=calibration_metadata_path,
        model_weights_path=Path(config.vision.model_weights_path),
        device=args.device,
    )

    for square in args.squares:
        square_estimate = board_estimator.estimate_square(args.squares_dir / square / f"{square}.png")
        # The stored scores are log-probabilities; exp() turns them back into probabilities.
        ranked = sorted(
            ((getattr(square_estimate, label), label) for label in TARGET_MAP),
            reverse=True,
        )[: args.top_k]
        predictions = "  ".join(f"{label}: {math.exp(logprob):.3f}" for logprob, label in ranked)
        print(f"{square}  {predictions}")

