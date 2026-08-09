import pytest

from chess_commentator.game import ChessGame
from chess_commentator.perception.board_estimator import SquareEstimate, BoardEstimate
from chess_commentator.board import SQUARES
from chess_commentator.labels import TARGET_MAP


@pytest.fixture
def make_game():
    """Build ChessGames and make sure their Stockfish processes are reaped."""
    games = []

    def _make(**kwargs):
        game = ChessGame(**kwargs)
        games.append(game)
        return game

    yield _make
    for game in games:
        game.close()


def create_board_estimate(square_occupants: dict[str, str]):
    """A BoardEstimate in which every square is, by default, equally likely to hold any piece
    (all 13 scores left at 0), except the squares named in square_occupants, which are given an
    overwhelming score for the piece they map to. Just enough signal for estimate_move to have
    exactly one sensible answer.
    """
    for piece in square_occupants.values():
        assert piece in TARGET_MAP.keys() # "empty", "K", "Q", ...

    board_estimate = BoardEstimate()
    for square in SQUARES:
        if square not in square_occupants.keys():
            setattr(board_estimate, square, SquareEstimate())
        else:
            square_estimate = SquareEstimate()
            setattr(square_estimate, square_occupants[square], 100)
            setattr(
                board_estimate, 
                square,
                square_estimate
            )
    return board_estimate

### Test that estimate move works
@pytest.mark.parametrize(
    "move_uci, square_occupants",
    [
        ("e2e4", {"e2": "empty", "e4": "P"}),
        ("b1c3", {"b1": "empty", "c3": "N"})
    ]
)
def test_move_estimation(move_uci, square_occupants, make_game):
    game = make_game()
    board_estimate = create_board_estimate(square_occupants)
    move_estimate = game.estimate_move(board_estimate)
    assert move_estimate[0]["move"] == move_uci


def test_estimate_move_attaches_move_info(make_game):
    """Speaker and main.py both index candidate["move_info"], so estimate_move must emit it."""
    game = make_game()
    candidates = game.estimate_move(create_board_estimate({"e2": "empty", "e4": "P"}))
    top = candidates[0]
    assert top["move"] == "e2e4"
    assert top["move_info"]["move"] == "e2e4"
    assert top["move_info"]["san"] == "e4"
    assert top["move_info"]["moved_piece"] == "Pawn"


### describe_move: the flags a commentator needs, read off the pre-move board

# Each case: a position, a move, and the move_info fields that must come out of it.
DESCRIBE_CASES = [
    pytest.param(
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "e4d5",
        {"san": "exd5", "moved_piece": "Pawn", "turn": "white", "capture": True,
         "captured_piece": "Pawn", "castle": None, "en_passant": False,
         "promotion": None, "check": False, "checkmate": False},
        id="normal_capture",
    ),
    pytest.param(
        # The captured pawn stands on d5, NOT on the destination square d6, so a naive
        # piece_at(to_square) would report captured_piece=None here.
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "e5d6",
        {"san": "exd6", "moved_piece": "Pawn", "capture": True,
         "captured_piece": "Pawn", "en_passant": True, "checkmate": False},
        id="en_passant",
    ),
    pytest.param(
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "e1g1",
        {"san": "O-O", "moved_piece": "King", "capture": False,
         "captured_piece": None, "castle": "kingside", "check": False},
        id="kingside_castle",
    ),
    pytest.param(
        "r3kbnr/pppqpppp/2np4/8/8/2NPB3/PPPQPPPP/R3KBNR w KQkq - 6 5",
        "e1c1",
        {"san": "O-O-O", "moved_piece": "King", "castle": "queenside"},
        id="queenside_castle",
    ),
    pytest.param(
        "8/P7/8/8/8/8/8/K6k w - - 0 1",
        "a7a8q",
        {"san": "a8=Q+", "moved_piece": "Pawn", "capture": False,
         "promotion": "Queen", "check": True, "checkmate": False},
        id="promotion",
    ),
    pytest.param(
        "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
        "a1a8",
        {"san": "Ra8+", "moved_piece": "Rook", "check": True, "checkmate": False},
        id="check_not_mate",
    ),
    pytest.param(
        # Scholar's mate.
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "f3f7",
        {"san": "Qxf7#", "moved_piece": "Queen", "capture": True,
         "captured_piece": "Pawn", "check": True, "checkmate": True},
        id="checkmate",
    ),
    pytest.param(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "e7e5",
        {"san": "e5", "turn": "black", "move_number": 1, "capture": False,
         "captured_piece": None, "castle": None, "en_passant": False,
         "promotion": None, "check": False, "checkmate": False},
        id="black_quiet_move",
    ),
]


@pytest.mark.parametrize("fen, move_uci, expected", DESCRIBE_CASES)
def test_describe_move(fen, move_uci, expected, make_game):
    game = make_game(fen=fen)
    # Not the fen literal above: python-chess drops an en-passant square from the FEN when
    # no legal en-passant capture actually exists, so the round-trip is not byte-identical.
    before = game.fen()

    move_info = game.describe_move(move_uci)

    assert move_info["move"] == move_uci
    for key, value in expected.items():
        assert move_info[key] == value, f"{key}: {move_info[key]!r} != {value!r}"

    # describe_move must not mutate the board it read from.
    assert game.fen() == before


### cp_loss_for: the sign must follow the side that MOVES, not the side to move afterwards

def test_cp_loss_punishes_a_blunder(make_game):
    """1. f3 e5 2. g4?? hangs mate-in-one (Qh4#).

    The old rate_move() read board.turn *after* pushing -- i.e. the opponent's colour --
    which flipped the sign and clamped every blunder to 0. This is the regression guard.
    """
    game = make_game()
    game.apply_move("f2f3")
    game.apply_move("e7e5")

    cp_loss, _ = game.cp_loss_for("g2g4")
    assert cp_loss > 500


def test_cp_loss_forgives_a_good_move(make_game):
    game = make_game()
    cp_loss, _ = game.cp_loss_for("e2e4")
    assert cp_loss < 60


def test_cp_loss_for_does_not_mutate(make_game):
    game = make_game()
    before = game.fen()
    score_before = game.recent_position_score

    game.cp_loss_for("e2e4")

    assert game.fen() == before
    assert game.recent_position_score == score_before
    assert game.move_log == []


def test_apply_move_reuses_supplied_rating(make_game):
    """main.py hands apply_move the cp_loss the worker already computed, so it must be
    recorded verbatim rather than recomputed."""
    game = make_game()
    move_info = game.describe_move("e2e4")

    game.apply_move("e2e4", move_info=move_info, cp_loss=42, new_score=123)

    assert game.recent_position_score == 123
    assert game.cp_losses["white"] == [42]
    assert game.move_log[-1]["san"] == "e4"


### history accessors

def test_history_streaks_and_averages(make_game):
    game = make_game()
    # Hand-build a log rather than playing 6 moves through Stockfish.
    game.move_log = [
        {"uci": "e2e4", "san": "e4", "turn": "white", "capture": False, "cp_loss": 10},
        {"uci": "d7d5", "san": "d5", "turn": "black", "capture": False, "cp_loss": 20},
        {"uci": "e4d5", "san": "exd5", "turn": "white", "capture": True, "cp_loss": 0},
        {"uci": "d8d5", "san": "Qxd5", "turn": "black", "capture": True, "cp_loss": 300},
    ]
    game.cp_losses = {"white": [10, 0], "black": [20, 300]}

    assert game.capture_streak() == 2
    assert game.quiet_streak() == 0
    assert game.recent_moves(3) == ["d7d5", "e4d5", "d8d5"]
    assert game.average_cp_loss() == {"white": 5.0, "black": 160.0}
    assert game.last_cp_losses(1) == {"white": [0], "black": [300]}


def test_quiet_streak_is_the_other_side_of_the_coin(make_game):
    game = make_game()
    game.move_log = [
        {"uci": "e4d5", "san": "exd5", "turn": "white", "capture": True, "cp_loss": 0},
        {"uci": "g8f6", "san": "Nf6", "turn": "black", "capture": False, "cp_loss": 5},
        {"uci": "b1c3", "san": "Nc3", "turn": "white", "capture": False, "cp_loss": 5},
    ]
    assert game.quiet_streak() == 2
    assert game.capture_streak() == 0


def test_history_snapshot_of_a_fresh_game(make_game):
    game = make_game()
    snapshot = game.history_snapshot()
    assert snapshot["recent_moves"] == []
    assert snapshot["capture_streak"] == 0
    assert snapshot["quiet_streak"] == 0
    assert snapshot["average_cp_loss"] == {"white": 0.0, "black": 0.0}
    assert snapshot["last_cp_losses"] == {"white": [], "black": []}


### end-of-game summary, as consumed by the closing roast

SCHOLARS_MATE = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]
# Black to move, no legal move, not in check.
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"


def play(game, ucis):
    """Push moves without an engine: apply_move only reaches Stockfish if cp_loss is None."""
    for uci in ucis:
        game.apply_move(uci, cp_loss=0, new_score=0)


def test_outcome_summary_of_a_checkmate(make_game):
    game = make_game()
    play(game, SCHOLARS_MATE)

    assert game.outcome_summary() == {
        "result": "1-0", "termination": "CHECKMATE", "winner": "white",
    }


def test_outcome_summary_names_black_as_the_winner(make_game):
    """The winner is read off the outcome, not off whose turn it is -- getting that backwards
    is invisible in a white win."""
    game = make_game()
    play(game, FOOLS_MATE)

    assert game.outcome_summary() == {
        "result": "0-1", "termination": "CHECKMATE", "winner": "black",
    }


def test_outcome_summary_of_a_stalemate_has_no_winner(make_game):
    """winner None means a draw, and must stay distinguishable from a win."""
    game = make_game(fen=STALEMATE_FEN)

    assert game.outcome_summary() == {
        "result": "1/2-1/2", "termination": "STALEMATE", "winner": None,
    }


def test_outcome_summary_of_a_game_still_being_played(make_game):
    """board.outcome() is None mid-game. Reporting that must not raise."""
    game = make_game()
    play(game, ["e2e4"])

    assert game.outcome_summary() == {
        "result": "*", "termination": "UNFINISHED", "winner": None,
    }


def test_worst_blunder_is_the_costliest_move(make_game):
    game = make_game()
    game.move_log = [
        {"uci": "e2e4", "san": "e4", "turn": "white", "capture": False, "cp_loss": 10},
        {"uci": "d8h4", "san": "Qh4", "turn": "black", "capture": False, "cp_loss": 412},
        {"uci": "b1c3", "san": "Nc3", "turn": "white", "capture": False, "cp_loss": 20},
    ]

    assert game.worst_blunder()["san"] == "Qh4"


def test_worst_blunder_of_a_game_with_no_moves(make_game):
    assert make_game().worst_blunder() is None


def test_final_snapshot_summarizes_the_whole_game(make_game):
    game = make_game()
    play(game, SCHOLARS_MATE)

    snapshot = game.final_snapshot()

    assert snapshot["result"] == "1-0"
    assert snapshot["winner"] == "white"
    assert snapshot["total_plies"] == 7
    assert snapshot["captures"] == 1  # Qxf7#
    assert [entry["san"] for entry in snapshot["moves"]] == [
        "e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#",
    ]
    # The move number is what lets the roast line a move up with the comment it made on it.
    assert [entry["move_number"] for entry in snapshot["moves"]] == [1, 1, 2, 2, 3, 3, 4]


def test_final_snapshot_does_not_alias_the_live_move_log(make_game):
    """Same contract as history_snapshot: a worker reads this while the main thread may
    still be mutating move_log."""
    game = make_game()
    play(game, ["e2e4"])

    snapshot = game.final_snapshot()
    play(game, ["e7e5"])

    assert len(snapshot["moves"]) == 1


def test_final_snapshot_tolerates_a_move_log_without_move_numbers(make_game):
    """move_number was added to move_log after the fact, and it is read back with .get()."""
    game = make_game()
    game.move_log = [
        {"uci": "e2e4", "san": "e4", "turn": "white", "capture": False, "cp_loss": 10},
    ]

    snapshot = game.final_snapshot()

    assert snapshot["moves"][0]["move_number"] is None
    assert snapshot["worst_blunder"]["move_number"] is None
