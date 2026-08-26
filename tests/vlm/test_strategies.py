"""Pure-helper tests for vlm/strategies.py -- no API calls, no data files.

These cover the parsing + legality logic that turns a VLM's raw output (method i's UCI list,
method ii's FEN list) into a prediction and the illegality statistics, plus the FEN/label
plumbing they rely on.
"""
import chess
import pytest

from chess_commentator.vlm.strategies import (
    board_to_labels,
    fen_board_to_labels,
    first_legal_and_stats,
    implied_move,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _board_after(*ucis):
    board = chess.Board(START_FEN)
    for uci in ucis:
        board.push_uci(uci)
    return board


def test_fen_board_to_labels_reads_placement():
    labels = fen_board_to_labels("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    assert labels["e2"] == "P"
    assert labels["e7"] == "p"
    assert labels["e4"] == "empty"
    assert labels["a1"] == "R"


def test_fen_board_to_labels_accepts_full_fen():
    # A full FEN (with side-to-move etc.) is accepted; only the placement field is used.
    labels = fen_board_to_labels(START_FEN)
    assert labels == board_to_labels(chess.Board(START_FEN))


@pytest.mark.parametrize("bad", ["", "   ", "not a fen", "rnbqkbnr/9/8/8/8/8/8/8", 123, None])
def test_fen_board_to_labels_rejects_malformed(bad):
    assert fen_board_to_labels(bad) is None


def test_implied_move_finds_the_legal_move():
    after = _board_after("e2e4")
    assert implied_move(START_FEN, board_to_labels(after)) == "e2e4"


def test_implied_move_none_when_unreachable():
    # Two pawns moved -> no single legal move reproduces this placement.
    board = chess.Board(START_FEN)
    board.remove_piece_at(chess.E2)
    board.set_piece_at(chess.E4, chess.Piece.from_symbol("P"))
    board.remove_piece_at(chess.D2)
    board.set_piece_at(chess.D4, chess.Piece.from_symbol("P"))
    assert implied_move(START_FEN, board_to_labels(board)) is None


def test_first_legal_and_stats_move_skips_leading_illegal():
    stats = first_legal_and_stats(START_FEN, ["e2e5", "e2e4", "g1f3"], "move")
    assert stats["first_output_illegal"] is True   # e2e5 is illegal
    assert stats["first_legal"] == "e2e4"          # first legal one is the prediction
    assert stats["ordered_legal"] == ["e2e4", "g1f3"]
    assert stats["none_legal"] is False


def test_first_legal_and_stats_move_first_is_legal():
    stats = first_legal_and_stats(START_FEN, ["e2e4"], "move")
    assert stats["first_output_illegal"] is False
    assert stats["first_legal"] == "e2e4"


def test_first_legal_and_stats_move_none_legal():
    stats = first_legal_and_stats(START_FEN, ["z9z9", "e2e5"], "move")
    assert stats["none_legal"] is True
    assert stats["first_legal"] is None
    assert stats["first_output_illegal"] is True


def test_first_legal_and_stats_accepts_san():
    stats = first_legal_and_stats(START_FEN, ["e4", "Nf3"], "move")
    assert stats["first_legal"] == "e2e4"
    assert stats["ordered_legal"] == ["e2e4", "g1f3"]


def test_first_legal_and_stats_board_kind():
    good = _board_after("e2e4").board_fen()
    stats = first_legal_and_stats(START_FEN, [good], "board")
    assert stats["first_output_illegal"] is False
    assert stats["first_legal"] == "e2e4"


def test_first_legal_and_stats_board_dedupes_ordered_legal():
    good = _board_after("e2e4").board_fen()
    stats = first_legal_and_stats(START_FEN, [good, good], "board")
    assert stats["ordered_legal"] == ["e2e4"]  # duplicate implied move collapsed


def test_first_legal_and_stats_empty_candidates():
    stats = first_legal_and_stats(START_FEN, [], "move")
    assert stats["first_legal"] is None
    assert stats["none_legal"] is True
    assert stats["first_output_illegal"] is None  # no first output to judge
