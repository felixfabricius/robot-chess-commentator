"""
Board vocabulary shared across the project: the 64 square names and the 13 square labels
(12 pieces in FEN notation, plus "empty").
"""
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