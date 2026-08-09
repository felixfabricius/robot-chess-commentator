"""
Board vocabulary shared across the project: the 64 square names, the 13 square labels
(12 pieces in FEN notation, plus "empty"), and the shape of a *reading* of a board.

The estimate types live here rather than next to the estimator that fills them because both
`perception` (the CNN backend) and `vlm` (the Claude backends) produce them, and a type owned by
either one would force an import cycle between the two. They are also literally derived from the
vocabulary above: `SquareEstimate` has one score field per entry of PIECES, and `BoardEstimate`
has one slot per entry of SQUARES.
"""
from dataclasses import dataclass, make_dataclass
from pathlib import Path

FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = [str(i) for i in range(1, 9)]
SQUARES = [file + rank for file in FILES for rank in RANKS]

PIECES = ["empty", "K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]

# Human-readable name for each label, for printing. White pieces are Capitalised and tagged
# "(w)", black pieces are lowercased and tagged "(b)", and "empty" stays "empty" -- e.g.
# "R" -> "Rook (w)", "r" -> "rook (b)".
_PIECE_TYPE_NAMES = {"K": "King", "Q": "Queen", "R": "Rook", "B": "Bishop", "N": "Knight", "P": "Pawn"}


def _piece_display(symbol: str) -> str:
    if symbol == "empty":
        return "empty"
    name = _PIECE_TYPE_NAMES[symbol.upper()]
    return f"{name} (w)" if symbol.isupper() else f"{name.lower()} (b)"


PIECE_DISPLAY = {symbol: _piece_display(symbol) for symbol in PIECES}


@dataclass
class SquareEstimate:
    """One square's reading: a logit-like score for each of the 13 labels.

    The scores are deliberately not normalised to any one convention -- the CNN and square_logits
    store log-probabilities, square_label stores a hard one-hot -- because game.estimate_move()
    re-applies its own softmax over them. What matters is that they are comparable within a square.
    """
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

# One field per square, so a board estimate is addressed as `estimate.e4` rather than by index.
BoardEstimate = make_dataclass(
    "BoardEstimate",
    [(square, SquareEstimate | None, None) for square in SQUARES]
)