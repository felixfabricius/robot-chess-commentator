"""
The game itself: board state, inferring the played move from a board estimate, and rating
moves with Stockfish.

Finding the engine is the fiddly part (see resolve_stockfish_path), so it is deliberately
kept optional: nothing in the board-reading path spawns an engine, and ChessGame only
launches one the first time somebody asks for an evaluation.
"""
import os
import shutil
import threading
from pathlib import Path

import chess
import chess.engine
import torch
from torch import nn

from chess_commentator.board import SQUARES
from chess_commentator.labels import TARGET_MAP, INVERSE_TARGET_MAP


def _is_executable(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name == "nt":
        # X_OK is meaningless on Windows (it returns True for any readable file), so go by
        # extension instead -- otherwise a stray stockfish.txt on PATH would be "found".
        return path.suffix.lower() in {".exe", ".com", ".bat", ".cmd"}
    return os.access(path, os.X_OK)


def resolve_stockfish_path(explicit: str | None = None) -> str:
    """Locate a Stockfish binary: explicit argument, then $STOCKFISH_PATH, then $PATH.

    A path that was named explicitly but does not exist is an error rather than a reason
    to fall through -- that is a typo in the config, and silently analysing with some
    other binary found on PATH would hide it.
    """
    for candidate in (explicit, os.environ.get("STOCKFISH_PATH")):
        if candidate:
            if Path(candidate).is_file():
                return str(candidate)
            raise FileNotFoundError(f"No Stockfish binary at {str(candidate)!r}.")

    found = shutil.which("stockfish")
    if found:
        return found

    # winget installs the binary under an arch-tagged name (stockfish-windows-x86-64-avx2.exe)
    # and creates no `stockfish` shim, so which() misses it even though its directory is on
    # PATH. Sweep PATH for anything that looks like a Stockfish executable before giving up.
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            matches = sorted(Path(entry).glob("stockfish*"))
        except OSError:  # unreadable or malformed PATH entry
            continue
        for match in matches:
            if _is_executable(match):
                return str(match)

    raise FileNotFoundError(
        "Stockfish not found. Install it and put it on PATH, set STOCKFISH_PATH, or set "
        "engine.stockfish_path in config.yaml.\n"
        "  Windows:  winget install Stockfish.Stockfish\n"
        "  macOS:    brew install stockfish\n"
        "  Linux:    apt install stockfish"
    )


class ChessGame:
    """
    The game state, and everything the rest of the system asks of it.

    Two jobs, deliberately kept independent:

    1. Turning a board estimate into a move. estimate_move() scores every legal move by how
       well the position it leads to explains what the vision model saw, and returns them all,
       ranked. It never consults Stockfish. The whole list is returned rather than just the
       argmax because the players confirm or reject the top suggestion out loud (see main.py):
       when the reading is wrong, the move the players actually made is usually second or third.

    2. Judging a move once it is played: centipawn loss, capture/quiet streaks, average
       accuracy per side. This is what the commentary prompt is built from, and it is the only
       part that needs an engine.
    """
    def __init__(
        self,
        fen: str | None = None,
        depth=16,
        stockfish_path: str | None = None,
    ):
        # A FEN can be passed in to continue the game from an arbitrary prior position, which is
        # what evaluation does: it replays recorded board positions one at a time.
        fen = fen if fen is not None else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.board = chess.Board(fen=fen)
        # estimate_move() scores candidate moves with cross-entropy against the vision model's
        # per-square scores. This works for every backend: log-probabilities (CNN, model iii/iv
        # logits) and hard one-hot readings alike -- CrossEntropyLoss soft-maxes its input, so a
        # 0/1 one-hot fed in as "logits" is well-defined and ranks moves identically to the old
        # squared-error path it replaced (see estimate_move).
        self.loss_fn = nn.CrossEntropyLoss()
        self.depth = depth  # engine search depth for move rating; maybe increase to 18 eventually
        # Stockfish is spawned on first use, not here -- see the engine property. estimate_move()
        # never touches it, so board reading and move ranking run with no engine installed at all.
        self._stockfish_path = stockfish_path
        self._engine = None
        self._recent_position_score = None
        # Speaker's background thread calls cp_loss_for() while the main thread may still be
        # inside estimate_move(). One Stockfish process, so serialise access to it.
        self.engine_lock = threading.Lock()

        # Game history, appended to by apply_move(). Feeds the commentary prompt.
        self.move_log = []  # one dict per played move: uci, san, turn, capture, cp_loss
        self.cp_losses = {"white": [], "black": []}

    @property
    def engine(self):
        """The Stockfish process, launched on first read.

        Every read goes through eval_position(), which holds engine_lock, so two threads
        racing to be the first user cannot spawn two processes.
        """
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                resolve_stockfish_path(self._stockfish_path)
            )
        return self._engine

    @property
    def recent_position_score(self):
        """Engine score of the current position. Evaluated on first read rather than in
        __init__, which used to spend a depth-16 search on every ChessGame ever built --
        including the ones that only rank moves and never ask for a score."""
        if self._recent_position_score is None:
            self._recent_position_score = self.eval_position()
        return self._recent_position_score

    @recent_position_score.setter
    def recent_position_score(self, value):
        self._recent_position_score = value

    def fen(self):
        return self.board.fen()

    def eval_position(self, board=None):
        board = self.board if board is None else board
        with self.engine_lock:
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].white().score(mate_score=10000)
        return score  # centipawns, from White's perspective

    def estimate_move(self, board_estimate):
        """
        Rank every legal move by how well it explains the vision model's board estimate.

        For each legal move we ask: if this move was played, how badly would the resulting
        position disagree with what the model saw? The disagreement is a cross-entropy loss summed
        over the 64 squares, against whatever per-square scores the vision model produced (CNN
        log-probabilities, LLM one-hot readings, or LLM per-label scores -- all fed to the same
        CrossEntropyLoss, which soft-maxes them).

        Returns a list of {"move", "loss", "move_info"} dicts sorted by ascending loss, i.e.
        most plausible move first. Never touches the engine.
        """
        # Loss of the *current* position under the estimate. This is a constant term added to
        # every candidate's score, so it has no effect on the ranking whatsoever -- it is kept
        # only because it makes the absolute loss values comparable across board positions,
        # which is useful as an evaluation metric.
        initial_loss = 0
        for square in SQUARES:
            square_estimate = getattr(board_estimate, square)
            piece_at_square = self.board.piece_at(chess.parse_square(square))
            piece_at_square = "empty" if piece_at_square is None else piece_at_square.symbol()
            # Cross-entropy against the vision model's per-square scores. The 13 scores have to be
            # laid out in TARGET_MAP index order, which is what INVERSE_TARGET_MAP is for.
            initial_loss += self.loss_fn(
                torch.tensor(
                    [
                        getattr(square_estimate, INVERSE_TARGET_MAP[target])
                        for target in range(13)
                    ],
                    dtype=torch.float32
                ),
                torch.tensor(TARGET_MAP[piece_at_square])
            )

        scored_moves = []
        for move in self.board.legal_moves:
            loss_increment = 0

            before = self.board.copy()
            after = self.board.copy()
            after.push(move)

            move_info = self.describe_move(move, after=after)

            changed_squares = []

            # Diff all 64 squares rather than just the two squares readable from the move in UCI
            # notation, because there are edge cases in which more than 2 squares change:
            #   - Castling changes four squares: king from/to and rook from/to.
            #   - En passant changes three squares: pawn from, pawn to, and captured pawn square.
            #   - Promotion capture still works out for the destination square, but the captured
            #     piece disappears from square_to.
            # TODO: could be sped up by handling those edge cases explicitly, so that we don't
            # loop over all 64 squares for every legal move.
            for square in chess.SQUARES:
                before_piece = before.piece_at(square)
                after_piece = after.piece_at(square)

                before_symbol = before_piece.symbol() if before_piece else "empty"
                after_symbol = after_piece.symbol() if after_piece else "empty"

                if before_symbol != after_symbol:
                    changed_squares.append((
                        chess.square_name(square),
                        before_symbol,
                        after_symbol,
                    ))

            # Only the squares the move changes can change the loss, so the candidate's score is
            # initial_loss plus a delta over those few squares: drop each changed square's old
            # contribution, add its new one.
            #
            # CAREFUL: getattr does not support nested attribute access, so the square estimate
            # has to be pulled out first; getattr(board_estimate, f"{square}.{piece}") does not
            # work.
            for square, old_piece, new_piece in changed_squares:
                square_estimate = getattr(board_estimate, square)
                square_pred_tensor = torch.tensor(
                    [
                        getattr(square_estimate, INVERSE_TARGET_MAP[target])
                        for target in range(13)
                    ],
                    dtype=torch.float32
                )
                loss_increment += self.loss_fn(square_pred_tensor, torch.tensor(TARGET_MAP[new_piece]))
                loss_increment -= self.loss_fn(square_pred_tensor, torch.tensor(TARGET_MAP[old_piece]))

            scored_moves.append({
                "move": move.uci(),
                "loss": initial_loss + loss_increment,
                "move_info": move_info,
            })

        # Sort candidate moves by likelihood (descending), which means sorting in ascending
        # order by how badly they disagree with the board estimate.
        scored_moves.sort(key = lambda x: x["loss"])
        return scored_moves

    def describe_move(self, move, after=None):
        """Everything a commentator would want to know about `move`, read off the
        *pre-move* board. Nothing here mutates state, so a candidate move can be
        described without ever being played.

        `after` is the board with `move` already pushed. estimate_move() builds one
        anyway, so it passes it in; otherwise we make our own.
        """
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        board = self.board
        if after is None:
            after = board.copy()
            after.push(move)

        piece = board.piece_at(move.from_square)
        assert piece is not None  # otherwise the move could not have been legal

        en_passant = board.is_en_passant(move)
        if en_passant:
            # The captured pawn does not stand on the destination square, so
            # piece_at(to_square) is None here. It is always a pawn.
            captured_piece = "Pawn"
        else:
            captured = board.piece_at(move.to_square)
            captured_piece = _piece_name(captured.piece_type) if captured else None

        if board.is_castling(move):
            castle = "kingside" if board.is_kingside_castling(move) else "queenside"
        else:
            castle = None

        return {
            "move": move.uci(),
            "san": board.san(move),
            "moved_piece": _piece_name(piece.piece_type),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "move_number": board.fullmove_number,
            "capture": board.is_capture(move),
            "captured_piece": captured_piece,
            "castle": castle,
            "en_passant": en_passant,
            "promotion": _piece_name(move.promotion) if move.promotion else None,
            "check": board.gives_check(move),
            "checkmate": after.is_checkmate(),
        }

    def evaluate_after(self, move):
        """White-perspective centipawn score of the position `move` would lead to.
        Does not touch self.board, so it is safe to call from a worker thread on a
        move that has not been (and may never be) played."""
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        board = self.board.copy()
        board.push(move)
        return self.eval_position(board)

    def cp_loss_for(self, move):
        """How much `move` costs the side playing it, in centipawns. Returns
        (cp_loss, new_score) so the caller can reuse the evaluation when it commits
        the move instead of paying for a second engine analysis.

        The mover's colour is read off the board *before* any push. Doing this after
        pushing would read the opponent's colour and invert the sign, which is what
        the old rate_move() did.
        """
        new_score = self.evaluate_after(move)
        if self.board.turn == chess.WHITE:
            cp_loss = self.recent_position_score - new_score
        else:
            cp_loss = new_score - self.recent_position_score
        # A negative loss would mean the move beat the engine's own best line -> clamp.
        return max(0, cp_loss), new_score

    def apply_move(self, move_uci, move_info=None, cp_loss=None, new_score=None):
        move = chess.Move.from_uci(move_uci)

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")

        if move_info is None:
            move_info = self.describe_move(move)
        if cp_loss is None or new_score is None:
            cp_loss, new_score = self.cp_loss_for(move)

        self.board.push(move)
        self.recent_position_score = new_score

        self.move_log.append({
            "uci": move_info["move"],
            "san": move_info["san"],
            "turn": move_info["turn"],
            "move_number": move_info["move_number"],
            "capture": move_info["capture"],
            "cp_loss": cp_loss,
        })
        self.cp_losses[move_info["turn"]].append(cp_loss)

    # --- history, as consumed by the commentary prompt ---

    def recent_moves(self, n=6):
        return [entry["uci"] for entry in self.move_log[-n:]]

    def capture_streak(self):
        """How many moves in a row, ending at the last one played, were captures."""
        return _trailing_streak(self.move_log, capture=True)

    def quiet_streak(self):
        """How many moves in a row, ending at the last one played, were not captures."""
        return _trailing_streak(self.move_log, capture=False)

    def average_cp_loss(self):
        return {
            side: (sum(losses) / len(losses)) if losses else 0.0
            for side, losses in self.cp_losses.items()
        }

    def last_cp_losses(self, n=5):
        return {side: losses[-n:] for side, losses in self.cp_losses.items()}

    def history_snapshot(self, recent_moves=6, recent_cp_losses=5):
        """A self-contained copy of the game history, safe to hand to a worker thread
        (the main thread keeps mutating move_log/cp_losses as the game goes on)."""
        return {
            "recent_moves": self.recent_moves(recent_moves),
            "capture_streak": self.capture_streak(),
            "quiet_streak": self.quiet_streak(),
            "average_cp_loss": self.average_cp_loss(),
            "last_cp_losses": self.last_cp_losses(recent_cp_losses),
        }

    # --- end-of-game summary, as consumed by the closing roast ---

    def worst_blunder(self):
        """The single costliest move of the game, or None if nothing was played.

        Ties go to the earliest move: max() keeps the first maximum it sees, and the
        first time a player threw the game away is the more interesting one.
        """
        if not self.move_log:
            return None
        return _summarize_move(max(self.move_log, key=lambda entry: entry["cp_loss"]))

    def outcome_summary(self):
        """How the game ended: {"result", "termination", "winner"}.

        Termination is python-chess's own reason ("CHECKMATE", "STALEMATE",
        "THREEFOLD_REPETITION", ...). A game that is still running reports "*" /
        "UNFINISHED" rather than raising, so this is safe to call at any time.
        """
        outcome = self.board.outcome()
        if outcome is None:
            return {"result": "*", "termination": "UNFINISHED", "winner": None}

        # outcome.winner is None on a draw, which is distinct from "white".
        winner = None
        if outcome.winner is not None:
            winner = "white" if outcome.winner == chess.WHITE else "black"

        return {
            "result": outcome.result(),
            "termination": outcome.termination.name,
            "winner": winner,
        }

    def final_snapshot(self):
        """A self-contained summary of the whole game, safe to hand to a worker thread.

        Same contract as history_snapshot(), but for the closing roast: it wants the
        entire game rather than a recent window, so the move list is deliberately
        unbounded (a 60-move game is a few hundred tokens).
        """
        return {
            **self.outcome_summary(),
            "total_moves": self.board.fullmove_number,
            "total_plies": len(self.move_log),
            "captures": sum(1 for entry in self.move_log if entry["capture"]),
            "average_cp_loss": self.average_cp_loss(),
            "worst_blunder": self.worst_blunder(),
            "moves": [_summarize_move(entry) for entry in self.move_log],
        }

    def print_board(self):
        print(self.board)

    def close(self):
        # self._engine, not self.engine: reading the property would spawn a Stockfish process
        # purely in order to quit it. Idempotent, so an explicit close() inside a with-block
        # (or a second call) is harmless.
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _summarize_move(entry):
    """One move_log entry, copied into the shape the closing roast reads.

    move_number is read with .get(): it was added to move_log after the fact, so an entry
    built before it existed reports None rather than blowing up.
    """
    return {
        "move_number": entry.get("move_number"),
        "san": entry["san"],
        "uci": entry["uci"],
        "turn": entry["turn"],
        "cp_loss": entry["cp_loss"],
    }


def _piece_name(piece_type):
    return chess.piece_name(piece_type).capitalize()


def _trailing_streak(move_log, capture):
    streak = 0
    for entry in reversed(move_log):
        if entry["capture"] != capture:
            break
        streak += 1
    return streak
