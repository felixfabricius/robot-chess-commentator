"""The task text and answer schemas for every image-reading method, plus the helpers that fill
placeholders in them.

Prompts are keyed by method then version, so each strategy can be tuned independently and a
`prompt_version` recorded alongside its results. The `{...}` placeholders are filled per board by
the move/board strategies (previous position + image orientation); the square prompts need no
substitution. A trailing reasoning instruction is appended by `client.build_message_params` when
reasoning == "text", so the prompts here describe only the task and the answer format.
"""
import chess

from chess_commentator.board import PIECES

# Shared by the two square-level strategies (iii, iv) so they see identical visual guidance and a
# iii-vs-iv comparison reflects the label-vs-scores format, not prompt wording. Describes exactly
# what the annotated crop (image_processing._cutout_v2 / _cutout_global) draws: red dots on the
# target square's base corners, and -- on v2 crops -- a green convex-hull outline of the column of
# space above the square.
SQUARE_TASK_PREAMBLE = (
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
        1: (
            "You are looking at a top-down (rectified) photo of a physical chess board.\n"
            "{orientation}\n"
            "Return only the board position as a FEN board string (piece placement only, not the full "
            "FEN), in STANDARD orientation with a8 at the top-left -- use the orientation note above to "
            "map the photo's corners to real squares. Example format (starting position): "
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR. "
            "Do not include side to move, castling rights, move counters, or explanation. "
            "Inspect each of the 64 squares to identify which piece - if any - is on it. "
            "Output EXACTLY one FEN board string: exactly 8 rank groups separated by '/', at most 71 "
            "characters. Do not repeat rank groups or characters, and stop immediately after the 8th rank."
        ),
    },
    "square_label": {
        1: (
            SQUARE_TASK_PREAMBLE
            + "Return exactly one label from:\n"
            "empty, K, Q, R, B, N, P, k, q, r, b, n, p,\n"
            "where the letter is the piece in FEN notation (e.g. K is the white king; uppercase = "
            "White, lowercase = Black)."
        ),
    },
    "square_logits": {
        1: (
            SQUARE_TASK_PREAMBLE
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
            "Identify the single legal move that was played. Put your single best guess in `move_1` as a "
            "UCI move (e.g. 'e2e4', or 'e7e8q' for a promotion). If -- and only if -- you are genuinely "
            "uncertain, put distinct alternatives in `move_2`, `move_3`, ... (most likely first); leave the "
            "remaining slots unset. Never repeat a move and never pad the slots -- typically just move_1, "
            "rarely more than three. Fill move_1 even if you are unsure."
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
            "Report the position AFTER the move. Put your single best guess in `board_1` as a piece-placement "
            "FEN in standard orientation (a8 top-left, exactly 8 rank groups separated by '/'), e.g. "
            "'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'. If -- and only if -- you are genuinely uncertain, "
            "put distinct alternatives in `board_2`, `board_3`, ... (most likely first); leave the remaining "
            "slots unset. Never repeat a position and never pad the slots -- typically just board_1, rarely "
            "more than three. Fill board_1 even if you are unsure."
        ),
    },
}

# Structured-output json_schemas, one per method. Structured outputs guarantee the response is a
# JSON object matching the schema, so parsing never has to scrape free-form text. (json_schema
# does not support numeric/length/array-size constraints, so "return at least one, best first" is
# expressed in the prompt, not the schema.)
SQUARE_LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(PIECES)}},
    "required": ["label"],
    "additionalProperties": False,
}
SQUARE_LOGITS_SCHEMA = {
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

# Move/board candidates use a FIXED set of named slots rather than an unbounded array. json_schema
# structured outputs can't cap an array's length, and an open array is where the model runs away
# (it keeps emitting elements -- usually repeats -- until it hits max_tokens and the object never
# closes). Bounding the slots means even a degenerate/repetitive model emits a short, COMPLETE
# object; downstream dedup (first_legal_and_stats) collapses any repeats. Only the first slot is
# required; the rest are optional and read in order.
MAX_CANDIDATES = 8


def ordered_slots_schema(prefix: str) -> dict:
    return {
        "type": "object",
        "properties": {f"{prefix}_{i}": {"type": "string"} for i in range(1, MAX_CANDIDATES + 1)},
        "required": [f"{prefix}_1"],
        "additionalProperties": False,
    }


def collect_slots(parsed: dict, prefix: str) -> list[str]:
    """Ordered, non-empty values from prefix_1..N (the model fills as many slots as it has guesses)."""
    values = []
    for i in range(1, MAX_CANDIDATES + 1):
        value = parsed.get(f"{prefix}_{i}")
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


MOVE_SCHEMA = ordered_slots_schema("move")
BOARD_SCHEMA = ordered_slots_schema("board")
FEN_WHOLE_SCHEMA = {
    "type": "object",
    "properties": {"fen_board": {"type": "string"}},
    "required": ["fen_board"],
    "additionalProperties": False,
}

# max_tokens the two square strategies request synchronously; reused so batch requests match.
SQUARE_MAX_TOKENS = {"square_label": 512, "square_logits": 1024}


def orientation_sentence(corner_map: dict) -> str:
    """Describe the image's orientation from the setup's corner_map so the model can read the
    photo in its own frame yet still answer in canonical (a8-top-left) coordinates."""
    return (
        "In the photo, the square at the top-left is {tl}, top-right is {tr}, "
        "bottom-right is {br}, bottom-left is {bl}."
    ).format(**{k: corner_map[k] for k in ("tl", "tr", "br", "bl")})


def previous_position_prompt(prompt_template: str, previous_fen: str, corner_map: dict) -> str:
    board = chess.Board(previous_fen)
    return prompt_template.format(
        fen_board=board.board_fen(),
        ascii_board=str(board),
        orientation=orientation_sentence(corner_map),
    )
