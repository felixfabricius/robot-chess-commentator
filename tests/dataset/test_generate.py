"""Unit tests for the pure helpers and state logic in ``data_generation``.

These deliberately avoid the pygame GUI and the robot capture pipeline, which
require a display / hardware. Everything tested here is import-light.
"""

import csv

import chess

from chess_commentator.board import SQUARES
from chess_commentator.dataset.generate import (
    CSV_COLUMNS,
    DataGenerationSession,
    append_rows_to_csv,
    board_to_piece_map,
    build_square_rows,
    create_setup,
    piece_label_at,
    square_annotated_image_path,
    square_image_path,
)


# -- labels -----------------------------------------------------------------


def test_piece_label_at_starting_position():
    board = chess.Board()
    assert piece_label_at(board, "e1") == "K"
    assert piece_label_at(board, "e8") == "k"
    assert piece_label_at(board, "a1") == "R"
    assert piece_label_at(board, "a7") == "p"
    assert piece_label_at(board, "e4") == "empty"


def test_board_to_piece_map_covers_all_squares():
    board = chess.Board()
    piece_map = board_to_piece_map(board)
    assert len(piece_map) == 64
    assert set(piece_map) == set(SQUARES)
    # 32 pieces at the start, 32 empty squares.
    assert sum(1 for label in piece_map.values() if label != "empty") == 32
    assert piece_map["e2"] == "P"
    assert piece_map["d7"] == "p"


# -- paths ------------------------------------------------------------------


def test_square_image_paths_are_nested_correctly(tmp_path):
    squares_dir = tmp_path / "squares"
    assert square_image_path(squares_dir, "e4") == squares_dir / "e4" / "e4_annotated.png"
    assert (
        square_annotated_image_path(squares_dir, "e4")
        == squares_dir / "e4" / "e4_annotated.png"
    )


# -- row building -----------------------------------------------------------


def test_build_square_rows_shape_and_labels(tmp_path):
    board = chess.Board()
    piece_map = board_to_piece_map(board)
    squares_dir = tmp_path / "squares"
    rows = build_square_rows(
        setup_id="setup1",
        setup_split="train",
        image_id="board_x",
        squares_dir=squares_dir,
        full_image_path=tmp_path / "image.png",
        calibration_metadata_path=tmp_path / "calibration_metadata.json",
        piece_map=piece_map,
        valid_game_position=True,
        board_fen=board.fen(),
        previous_board_fen=None,
        move_uci=None,
        created_at="2026-07-01T10:00:00",
    )
    assert len(rows) == 64
    assert list(rows[0].keys()) == CSV_COLUMNS

    by_square = {row["square"]: row for row in rows}
    assert by_square["e1"]["label"] == "K"
    assert by_square["e4"]["label"] == "empty"
    # Paths are serialised POSIX-style (.as_posix()) so the CSV is portable across OSes.
    assert by_square["e1"]["square_image_path"] == (
        squares_dir / "e1" / "e1_annotated.png"
    ).as_posix()
    # None values are serialised as empty strings for the CSV.
    assert by_square["e1"]["previous_board_fen"] == ""
    assert by_square["e1"]["move_uci"] == ""


# -- CSV append -------------------------------------------------------------


def test_append_rows_writes_header_once_and_appends(tmp_path):
    csv_path = tmp_path / "nested" / "data.csv"
    row = {col: "x" for col in CSV_COLUMNS}

    append_rows_to_csv(csv_path, [row, row])
    append_rows_to_csv(csv_path, [row])

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    # 1 header + 3 data rows, header only once.
    assert reader[0] == CSV_COLUMNS
    assert len(reader) == 4
    assert all(line[0] != "setup_id" for line in reader[1:])


# -- setup dir --------------------------------------------------------------


def test_create_setup_makes_dir_with_given_timestamp(tmp_path):
    setup_id, setup_dir, setup_split = create_setup(tmp_path, timestamp="2026-07-01_120000")
    assert setup_id == "2026-07-01_120000"
    assert setup_dir == tmp_path / "2026-07-01_120000"
    assert setup_dir.is_dir()
    assert setup_split in ["train", "val", "test"]


# -- legal-move mode --------------------------------------------------------


def _session(tmp_path):
    # mini is stored but never touched by the non-hardware code paths these tests exercise.
    return DataGenerationSession(object(), config_path="config.yaml", data_root=tmp_path)


def test_legal_move_tracks_fens(tmp_path):
    session = _session(tmp_path)
    start_fen = session.board.fen()
    result = session.apply_legal_move("e2", "e4")
    assert result == "ok"
    assert session.previous_board_fen == start_fen
    assert session.board_fen != start_fen
    assert session.move_uci == "e2e4"
    assert session.valid_game_position is True


def test_illegal_move_rejected(tmp_path):
    session = _session(tmp_path)
    assert session.apply_legal_move("e2", "e5") == "illegal"
    assert session.move_uci is None


def test_promotion_requires_explicit_choice(tmp_path):
    session = _session(tmp_path)
    session.board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    # Without a promotion piece the move must not silently default to queen.
    assert session.apply_legal_move("a7", "a8") == "need_promotion"
    assert session.move_uci is None
    # With an explicit choice it applies.
    assert session.apply_legal_move("a7", "a8", promotion=chess.KNIGHT) == "ok"
    assert session.move_uci == "a7a8n"


# -- free-placement mode ----------------------------------------------------


def test_place_piece_sets_invalid_and_labels(tmp_path):
    session = _session(tmp_path)
    session.toggle_mode()
    assert session.legal_mode is False
    session.place_piece("e4", "Q")
    assert session.valid_game_position is False
    assert piece_label_at(session.board, "e4") == "Q"
    assert board_to_piece_map(session.board)["e4"] == "Q"


def test_remove_piece_clears_square(tmp_path):
    session = _session(tmp_path)
    session.toggle_mode()
    session.remove_piece("e2")
    assert piece_label_at(session.board, "e2") == "empty"
    assert session.valid_game_position is False


def test_move_piece_free_ignores_legality(tmp_path):
    session = _session(tmp_path)
    session.toggle_mode()
    assert session.move_piece_free("a1", "d4") is True
    assert piece_label_at(session.board, "d4") == "R"
    assert piece_label_at(session.board, "a1") == "empty"
    assert session.valid_game_position is False
    # Moving from an empty square returns False.
    assert session.move_piece_free("a1", "e5") is False


def test_new_game_resets_state(tmp_path):
    session = _session(tmp_path)
    session.toggle_mode()
    session.place_piece("e4", "Q")
    session.new_game()
    assert session.legal_mode is True
    assert session.valid_game_position is True
    assert session.previous_board_fen is None
    assert session.move_uci is None
    assert session.board.fen() == chess.Board().fen()


# -- recalibrate / new setup ------------------------------------------------


def _patch_hardware(monkeypatch):
    """Stub out the robot-facing calibrate/Processor so setup logic is testable.

    calibrate is stubbed with create_autospec so that a mismatched call -- e.g. if its
    signature drifts again -- fails loudly here rather than being silently swallowed by a
    ``lambda *a, **k`` that accepts anything.
    """
    import chess_commentator.dataset.generate as dg
    from unittest.mock import create_autospec

    monkeypatch.setattr(
        dg, "calibrate", create_autospec(dg.calibrate, return_value={"stub": True})
    )
    monkeypatch.setattr(dg, "Processor", lambda *args, **kwargs: object())


def test_recalibrate_keeps_position_by_default(tmp_path, monkeypatch):
    session = _session(tmp_path)
    # Put the board into an edited free-placement state.
    session.toggle_mode()
    session.place_piece("e4", "Q")
    fen_before = session.board.fen()
    assert session.valid_game_position is False

    _patch_hardware(monkeypatch)
    assert session.start_new_setup() is True

    # A new setup exists, but the position and all its state are preserved.
    assert session.setup_id is not None
    assert session.processor is not None
    assert session.board.fen() == fen_before
    assert session.valid_game_position is False
    assert session.legal_mode is False


def test_recalibrate_reset_board_true_starts_fresh(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.toggle_mode()
    session.place_piece("e4", "Q")

    _patch_hardware(monkeypatch)
    assert session.start_new_setup(reset_board=True) is True

    assert session.board.fen() == chess.Board().fen()
    assert session.valid_game_position is True
    assert session.legal_mode is True


def test_failed_calibration_does_not_change_board(tmp_path, monkeypatch):
    import chess_commentator.dataset.generate as dg

    session = _session(tmp_path)
    session.apply_legal_move("e2", "e4")
    fen_before = session.board.fen()

    monkeypatch.setattr(dg, "calibrate", lambda *a, **k: None)  # aborted
    assert session.start_new_setup() is False
    # Nothing was reset or half-applied.
    assert session.processor is None
    assert session.board.fen() == fen_before


def test_annotate_center_flag_passed_to_calibrate(tmp_path, monkeypatch):
    import chess_commentator.dataset.generate as dg

    calls = []
    monkeypatch.setattr(dg, "calibrate", lambda *a, **k: (calls.append((a, k)), {"stub": True})[1])
    monkeypatch.setattr(dg, "Processor", lambda *a, **k: object())

    session = DataGenerationSession(
        object(), config_path="config.yaml", data_root=tmp_path, annotate_center=True
    )
    assert session.start_new_setup() is True
    # calibrate is called as calibrate(mini, setup_dir, config_path, annotate_center),
    # so annotate_center is now the 4th positional arg.
    args, _kwargs = calls[0]
    assert args[3] is True
