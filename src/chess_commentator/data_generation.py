"""Data-generation tool for the chess vision classifier.

Run with ``uv run python -m chess_assistant.data_generation [--center]``. Needs the robot and a
display; calibration starts automatically on launch.

Workflow:

1. A ``pygame`` window shows a virtual chessboard (White at the bottom).
2. You make a move / edit the position on the virtual board.
3. You make the *same* change on the real physical board.
4. You press Space; the robot captures a photo.
5. The photo is warped and cut into 64 square cutouts by the existing
   :class:`~chess_assistant.image_processing.Processor`.
6. The virtual board is the source of truth for the label of every square.
7. Labels are written into each square's metadata JSON, a per-image
   ``metadata.json`` is saved, and 64 rows are appended to the master CSV.

Keybindings:

===========================  ===================================================================
Key                          Action
===========================  ===================================================================
Space                        Capture: photo -> warp -> cutout -> labels -> 64 CSV rows
click, then click            Select the source square, then the target (moves the piece)
right-click                  Cancel the current selection / armed spawn piece
f                            Toggle legal-move <-> free-placement mode
Esc                          Quit (cancels a pending promotion instead, if one is pending)
legal mode: n / r / q        New game / recalibrate (new setup) / quit
free mode: PNBRQK / pnbrqk   Arm a white / black piece to spawn, then click a square
free mode: x                 Remove the piece under the cursor
promotion: q / r / b / n     Choose the promotion piece (it never silently defaults to a queen)
===========================  ===================================================================

In free-placement mode the letters spawn pieces, so ``n`` / ``r`` / ``q`` are *not* commands
there (they would collide with the black knight / rook / queen). Press ``f`` to return to legal
mode first, or quit with Esc.

Data layout::

    data/generated/
      data.csv                          # master CSV, appended across every session
      <setup_id>/                       # timestamp; one per calibration
        calibration_metadata.json       # written by calibrate()
        setup_metadata.json
        board_<timestamp>/              # == image_id; the dir capture_image() creates
          image.png                     # the raw frame
          image_warped.png              # Processor.warp() output
          metadata.json                 # per-image record, includes the full piece_map
          squares/<square>/
            <square>_annotated.png      # the cutout the CSV points at
            <square>_masked.npy         # the model input
            <square>_metadata.json      # Processor's metadata, plus the "label" we add

CSV schema: one row per square, so 64 rows per captured image. The columns are ``CSV_COLUMNS``
(``setup_id``, ``setup_split``, ``image_id``, ``square``, ``label``, ``square_image_path``,
``full_image_path``, ``calibration_metadata_path``, ``valid_game_position``, ``board_fen``,
``previous_board_fen``, ``move_uci``, ``created_at``).

Labels are 13-class: ``"empty"``, or a python-chess piece symbol (``P N B R Q K`` for White,
``p n b r q k`` for Black). ``valid_game_position`` is True only while the position has stayed in
legal-move mode since the last new game; any free-placement edit flips it to False.

The pure helper functions and :class:`DataGenerationSession` contain no
``pygame`` / robot code so they can be unit tested without a display.
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from reachy_mini import ReachyMini

import chess

from chess_assistant.calibration import calibrate
from chess_assistant.camera import capture_image
from chess_assistant.config import SQUARES
from chess_assistant.image_processing import Processor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root under which everything is stored, grouped by physical setup.
DATA_ROOT = Path("data") / "generated"
CSV_NAME = "data.csv"

# One row per square cutout. Kept as a stable, ordered schema so the CSV can be
# appended to over many sessions without ever changing column order.
CSV_COLUMNS: list[str] = [
    "setup_id",
    "setup_split",
    "image_id",
    "square",
    "label",
    "square_image_path",
    "full_image_path",
    "calibration_metadata_path",
    "valid_game_position",
    "board_fen",
    "previous_board_fen",
    "move_uci",
    "created_at",
]

# 13-class label for an empty square.
EMPTY_LABEL = "empty"

# Piece letters accepted for spawning in free-placement mode (uppercase = white,
# lowercase = black), and the promotion choices in legal-move mode.
PIECE_SYMBOLS = set("PNBRQKpnbrqk")
PROMOTION_MAP = {
    "q": chess.QUEEN,
    "r": chess.ROOK,
    "b": chess.BISHOP,
    "n": chess.KNIGHT,
}


# ---------------------------------------------------------------------------
# Pure helpers (no pygame / robot dependencies -> easy to unit test)
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return the current time as an ISO-8601 string (seconds resolution)."""
    return datetime.now().isoformat(timespec="seconds")


def now_stamp() -> str:
    """Return a filesystem-friendly timestamp, e.g. ``2026-07-01_143512``."""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def save_json(path: Path, data: dict) -> None:
    """Write ``data`` as indented UTF-8 JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def piece_label_at(board: chess.Board, square: str) -> str:
    """Return the 13-class label of ``square`` (e.g. ``"e4"``).

    ``"empty"`` if there is no piece, otherwise the piece symbol
    (``"P"``, ``"n"``, ...). The virtual board is the source of truth.
    """
    piece = board.piece_at(chess.parse_square(square))
    return piece.symbol() if piece is not None else EMPTY_LABEL


def board_to_piece_map(board: chess.Board) -> dict[str, str]:
    """Map every one of the 64 squares to its 13-class label."""
    return {square: piece_label_at(board, square) for square in SQUARES}


def square_image_path(squares_dir: Path, square: str) -> Path:
    """Path recorded in the CSV for a square, e.g. ``squares/e4/e4_annotated.png``.

    Points at the annotated cutout (the plain ``<square>.png`` is no longer written). The model
    input — ``<square>_masked.npy`` — lives in the same per-square directory and is derived from
    this path's parent by the training loader.
    """
    return Path(squares_dir) / square / f"{square}_annotated.png"


def square_annotated_image_path(squares_dir: Path, square: str) -> Path:
    """Path to the annotated cutout, e.g. ``squares/e4/e4_annotated.png``."""
    return Path(squares_dir) / square / f"{square}_annotated.png"


def build_square_rows(
    *,
    setup_id: str,
    setup_split: str,
    image_id: str,
    squares_dir: Path,
    full_image_path: Path,
    calibration_metadata_path: Path,
    piece_map: dict[str, str],
    valid_game_position: bool,
    board_fen: str | None,
    previous_board_fen: str | None,
    move_uci: str | None,
    created_at: str,
    squares: list[str] = SQUARES,
) -> list[dict]:
    """Build the 64 CSV rows (one per square) for a single captured image."""
    rows: list[dict] = []
    for square in squares:
        rows.append(
            {
                "setup_id": setup_id,
                "setup_split": setup_split,
                "image_id": image_id,
                "square": square,
                "label": piece_map[square],
                "square_image_path": square_image_path(squares_dir, square).as_posix(),
                "full_image_path": Path(full_image_path).as_posix(),
                "calibration_metadata_path": Path(calibration_metadata_path).as_posix(),
                "valid_game_position": valid_game_position,
                "board_fen": board_fen or "",
                "previous_board_fen": previous_board_fen or "",
                "move_uci": move_uci or "",
                "created_at": created_at,
            }
        )
    return rows


def append_rows_to_csv(
    csv_path: Path,
    rows: list[dict],
    columns: list[str] = CSV_COLUMNS,
) -> None:
    """Append ``rows`` to the master CSV, writing a header if the file is new.

    Uses append mode so existing data is never lost and a crash mid-write can
    at worst leave a truncated final row rather than corrupting the whole file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def create_setup(data_root: Path, timestamp: str | None = None) -> tuple[str, Path, str]:
    """Create (and return) a new timestamped setup directory.

    Returns ``(setup_id, setup_dir, setup_split)``. ``timestamp`` can be injected for tests.
    """
    setup_id = timestamp or now_stamp()
    setup_dir = Path(data_root) / setup_id
    setup_dir.mkdir(parents=True, exist_ok=True)
    # The train/val/test split is drawn once PER SETUP, not per image: every frame of a setup
    # shares one lighting, one board and one camera pose, so splitting per image would put
    # near-identical frames on both sides of the split and let the model memorise a setup
    # instead of learning the pieces. Whole setups are therefore held out together.
    setup_split = random.choices(["train", "val", "test"], [0.6, 0.2, 0.2])[0]
    return setup_id, setup_dir, setup_split


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class DataGenerationSession:
    """Holds all mutable state for a data-generation run.

    Intentionally free of any ``pygame`` / OpenCV window code; the GUI
    (:class:`BoardUI`) drives this object, and the robot pipeline
    (calibrate / capture / warp / cutout) is called from here.
    """

    def __init__(
        self,
        mini: ReachyMini,
        config_path: Path,
        data_root: Path = DATA_ROOT,
        annotate_center: bool = False,
    ) -> None:
        self.mini = mini
        self.config_path = Path(config_path)
        self.data_root = Path(data_root)
        self.csv_path = self.data_root / CSV_NAME
        # When True, calibration measures the extended centre point (click it); otherwise the
        # centre is interpolated from the 4 corners. See chess_assistant.calibration.CalibrationUI.
        self.annotate_center = annotate_center

        # Set once a setup has been calibrated.
        self.setup_id: str | None = None
        self.setup_dir: Path | None = None
        self.setup_split: str | None = None
        self.calibration_metadata_path: Path | None = None
        self.processor: Processor | None = None

        # Virtual-board state.
        self.board = chess.Board()
        self.legal_mode = True
        self.valid_game_position = True
        self.previous_board_fen: str | None = None
        self.board_fen: str | None = self.board.fen()
        self.move_uci: str | None = None

        self.image_count = 0

    # -- setup / calibration ------------------------------------------------

    def start_new_setup(self, reset_board: bool = False) -> bool:
        """Create a new setup dir, calibrate, and (re)initialise the Processor.

        The current virtual-board position and all its state
        (``valid_game_position``, previous/current FEN, ``move_uci``,
        legal/free mode) are **kept** by default, so you can recapture the same
        position under a new setup to build robustness across setups. Pass
        ``reset_board=True`` to start from a fresh game instead. Returns
        ``True`` on success.
        """
        setup_id, setup_dir, setup_split = create_setup(self.data_root)

        calibration_data = calibrate(self.mini, setup_dir, self.config_path, self.annotate_center)
        if not calibration_data:
            print("Calibration was aborted or failed. Setup not created.")
            return False

        calibration_metadata_path = setup_dir / "calibration_metadata.json"
        try:
            processor = Processor(calibration_metadata_path, self.config_path)
        except Exception as exc:  # noqa: BLE001 - surface any init failure
            print(f"Failed to initialise Processor: {exc}")
            return False

        self.setup_id = setup_id
        self.setup_dir = setup_dir
        self.setup_split = setup_split
        self.calibration_metadata_path = calibration_metadata_path
        self.processor = processor

        save_json(
            setup_dir / "setup_metadata.json",
            {
                "setup_id": setup_id,
                "created_at": now_iso(),
                "config_path": Path(self.config_path).as_posix(),
                "calibration_metadata_path": Path(calibration_metadata_path).as_posix(),
            },
        )

        if reset_board:
            self.new_game()
        print(f"New setup ready: {setup_id}")
        return True

    # -- virtual board: legal-move mode ------------------------------------

    def new_game(self) -> None:
        """Reset to the standard starting position and legal-move mode."""
        self.board = chess.Board()
        self.legal_mode = True
        self.valid_game_position = True
        self.previous_board_fen = None
        self.board_fen = self.board.fen()
        self.move_uci = None

    def apply_legal_move(
        self,
        from_square: str,
        to_square: str,
        promotion: int | None = None,
    ) -> str:
        """Attempt a legal move from ``from_square`` to ``to_square``.

        Returns one of:

        - ``"ok"`` -- move applied; FEN tracking updated.
        - ``"illegal"`` -- no such legal move.
        - ``"need_promotion"`` -- the move is a pawn promotion; call again with ``promotion``
          set to a piece type.
        """
        from_idx = chess.parse_square(from_square)
        to_idx = chess.parse_square(to_square)

        try:
            candidates = [
                m
                for m in self.board.legal_moves
                if m.from_square == from_idx and m.to_square == to_idx
            ]
        except Exception:  # noqa: BLE001 - board may be odd after free edits
            return "illegal"

        if not candidates:
            return "illegal"

        if promotion is None:
            needs_promotion = any(m.promotion is not None for m in candidates)
            if needs_promotion and len(candidates) > 1:
                return "need_promotion"
            move = candidates[0]
        else:
            move = next((m for m in candidates if m.promotion == promotion), None)
            if move is None:
                return "illegal"

        self.previous_board_fen = self.board.fen()
        self.board.push(move)
        self.board_fen = self.board.fen()
        self.move_uci = move.uci()
        return "ok"

    # -- virtual board: free-placement mode --------------------------------

    def toggle_mode(self) -> None:
        """Toggle between legal-move and free-placement mode."""
        self.legal_mode = not self.legal_mode

    def _mark_edited(self) -> None:
        """Record that the position was edited freely (no longer a valid game)."""
        self.valid_game_position = False
        self.move_uci = None
        self.previous_board_fen = None
        self.board_fen = self._current_fen()

    def place_piece(self, square: str, symbol: str) -> None:
        """Place ``symbol`` (e.g. ``"Q"``/``"q"``) on ``square``, ignoring legality."""
        self.board.set_piece_at(
            chess.parse_square(square), chess.Piece.from_symbol(symbol)
        )
        self._mark_edited()

    def remove_piece(self, square: str) -> None:
        """Remove any piece from ``square``."""
        self.board.remove_piece_at(chess.parse_square(square))
        self._mark_edited()

    def move_piece_free(self, from_square: str, to_square: str) -> bool:
        """Move whatever is on ``from_square`` to ``to_square``, ignoring legality."""
        from_idx = chess.parse_square(from_square)
        piece = self.board.piece_at(from_idx)
        if piece is None:
            return False
        self.board.remove_piece_at(from_idx)
        self.board.set_piece_at(chess.parse_square(to_square), piece)
        self._mark_edited()
        return True

    def _current_fen(self) -> str | None:
        """Best-effort FEN; ``None`` if the position cannot be represented."""
        try:
            return self.board.fen()
        except Exception:  # noqa: BLE001
            return None

    # -- capture ------------------------------------------------------------

    def capture(self) -> bool:
        """Capture a photo, warp+cutout it, write labels and append CSV rows.

        Returns ``True`` only if the full pipeline succeeded. On any failure
        (capture, warp, cutout) nothing is appended to the CSV.
        """
        if self.processor is None or self.setup_dir is None:
            print("No active setup. Press 'r' to calibrate a setup first.")
            return False

        created_at = now_iso()

        # 1. Capture. capture_image() creates its own board_<timestamp> dir.
        try:
            image_dir = capture_image(self.mini, self.setup_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Image capture failed: {exc}")
            return False

        full_image_path = image_dir / "image.png"

        # 2. Warp + cutout using the existing Processor.
        try:
            warped_image_path = self.processor.warp(full_image_path)
            squares_dir = self.processor.cutout(warped_image_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Warp/cutout failed: {exc}. No rows written.")
            return False

        image_id = image_dir.name
        piece_map = board_to_piece_map(self.board)
        board_fen = self.board_fen

        # 3. Append the label to each square's existing metadata JSON.
        for square in SQUARES:
            meta_path = squares_dir / square / f"{square}_metadata.json"
            existing: dict = {}
            if meta_path.exists():
                try:
                    with meta_path.open(encoding="utf-8") as f:
                        existing = json.load(f)
                except (OSError, json.JSONDecodeError):
                    existing = {}
            existing["label"] = piece_map[square]
            save_json(meta_path, existing)

        # 4. Append the 64 CSV rows (built first, then written in one go).
        rows = build_square_rows(
            setup_id=self.setup_id,
            setup_split=self.setup_split,
            image_id=image_id,
            squares_dir=squares_dir,
            full_image_path=full_image_path,
            calibration_metadata_path=self.calibration_metadata_path,
            piece_map=piece_map,
            valid_game_position=self.valid_game_position,
            board_fen=board_fen,
            previous_board_fen=self.previous_board_fen,
            move_uci=self.move_uci,
            created_at=created_at,
        )
        append_rows_to_csv(self.csv_path, rows)

        # 5. Save the per-image metadata.
        save_json(
            image_dir / "metadata.json",
            {
                "setup_id": self.setup_id,
                "setup_split": self.setup_split,
                "image_id": image_id,
                "created_at": created_at,
                "valid_game_position": self.valid_game_position,
                "legal_move_mode": self.legal_mode,
                "board_fen": board_fen,
                "previous_board_fen": self.previous_board_fen,
                "move_uci": self.move_uci,
                "piece_map": piece_map,
                "full_image_path": Path(full_image_path).as_posix(),
                "warped_image_path": Path(warped_image_path).as_posix(),
                "squares_dir": Path(squares_dir).as_posix(),
                "calibration_metadata_path": Path(self.calibration_metadata_path).as_posix(),
            },
        )

        self.image_count += 1
        print(f"Captured {image_id}: appended {len(rows)} rows to {self.csv_path}")
        return True


# ---------------------------------------------------------------------------
# Pygame GUI (lazily imports pygame so the helpers above stay import-light)
# ---------------------------------------------------------------------------


class BoardUI:
    """A deliberately simple pygame board for driving the capture session."""

    SQUARE = 80
    MARGIN = 40
    PANEL = 320

    LIGHT = (240, 217, 181)
    DARK = (181, 136, 99)
    SELECT = (246, 246, 105)
    BG = (30, 30, 34)
    TEXT = (230, 230, 230)

    def __init__(self, session: DataGenerationSession) -> None:
        import pygame  # local import: not needed for the helpers/tests

        self.pg = pygame
        self.session = session

        pygame.init()
        pygame.display.set_caption("Chess data generation")
        board_px = self.SQUARE * 8
        self.width = self.MARGIN * 2 + board_px + self.PANEL
        self.height = self.MARGIN * 2 + board_px
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.label_font = pygame.font.SysFont("consolas", 16)
        self.panel_font = pygame.font.SysFont("consolas", 18)
        self.piece_images = self._load_piece_images()

        # Interaction state.
        self.selected: str | None = None          # source square for a move
        self.spawn: str | None = None              # piece selected to place
        self.promotion: tuple[str, str] | None = None  # (from, to) awaiting choice
        self.message = "Press 'r' to calibrate a setup, then Space to capture."
        self.running = True

    # -- piece assets --------------------------------------------------------

    def _load_piece_images(self) -> dict:
        """Rasterise the 12 piece SVGs to pygame surfaces, once, at startup.

        Uses the Cburnett set bundled with python-chess, sized to fit a square.
        """
        import io
        import chess.svg

        images = {}
        svg_size = self.SQUARE - 10  # small margin so pieces don't touch grid lines
        for symbol in PIECE_SYMBOLS:
            piece = chess.Piece.from_symbol(symbol)
            svg_data = chess.svg.piece(piece, size=svg_size)
            surface = self.pg.image.load(io.BytesIO(svg_data.encode("utf-8")), f"{symbol}.svg")
            images[symbol] = surface.convert_alpha()
        return images

    # -- coordinate helpers -------------------------------------------------

    def _square_rect(self, square: str):
        file_idx = ord(square[0]) - ord("a")
        rank = int(square[1])
        col = file_idx
        row = 8 - rank  # White at the bottom
        x = self.MARGIN + col * self.SQUARE
        y = self.MARGIN + row * self.SQUARE
        return self.pg.Rect(x, y, self.SQUARE, self.SQUARE)

    def _square_at_pixel(self, pos) -> str | None:
        x, y = pos
        col = (x - self.MARGIN) // self.SQUARE
        row = (y - self.MARGIN) // self.SQUARE
        if 0 <= col < 8 and 0 <= row < 8:
            file_char = chr(ord("a") + int(col))
            rank = 8 - int(row)
            return f"{file_char}{rank}"
        return None

    # -- drawing ------------------------------------------------------------

    def _draw_piece(self, symbol: str, rect) -> None:
        image = self.piece_images[symbol]
        self.screen.blit(image, image.get_rect(center=rect.center))

    def _draw_board(self) -> None:
        for square in SQUARES:
            rect = self._square_rect(square)
            file_idx = ord(square[0]) - ord("a")
            rank = int(square[1])
            is_light = (file_idx + rank) % 2 == 0
            self.pg.draw.rect(self.screen, self.LIGHT if is_light else self.DARK, rect)
            if square == self.selected:
                self.pg.draw.rect(self.screen, self.SELECT, rect, 5)
            if self.promotion and square in self.promotion:
                self.pg.draw.rect(self.screen, (200, 60, 60), rect, 5)

            piece = self.session.board.piece_at(chess.parse_square(square))
            if piece is not None:
                self._draw_piece(piece.symbol(), rect)

    def _draw_panel(self) -> None:
        x = self.MARGIN * 2 + self.SQUARE * 8
        mode = "LEGAL" if self.session.legal_mode else "FREE-PLACEMENT"
        lines = [
            f"Setup: {self.session.setup_id or '(none)'}",
            f"Images captured: {self.session.image_count}",
            f"Mode: {mode}",
            f"valid_game_position: {self.session.valid_game_position}",
            f"Last move: {self.session.move_uci or '-'}",
            f"Spawn piece: {self.spawn or '-'}",
            "",
        ]
        if self.promotion:
            lines.append("PROMOTION: press q / r / b / n")
            lines.append("")
        lines += [
            "-- Controls --",
            "Space: capture",
            "r: recalibrate / new setup",
            "n: new game (legal mode)",
            "f: toggle legal/free mode",
            "click: select then move",
            "right-click: cancel",
            "",
            "Free mode:",
            "  P N B R Q K = white spawn",
            "  p n b r q k = black spawn",
            "  x: remove under cursor",
            "",
            "Esc: quit  (q also quits",
            "     in legal mode)",
            "",
            f"> {self.message}",
        ]
        y = self.MARGIN
        for line in lines:
            surface = self.panel_font.render(line, True, self.TEXT)
            self.screen.blit(surface, (x, y))
            y += 22

    def _draw(self) -> None:
        self.screen.fill(self.BG)
        self._draw_board()
        self._draw_panel()
        self.pg.display.flip()

    # -- event handling -----------------------------------------------------

    def _on_key(self, event) -> None:
        pg = self.pg
        key = event.key
        char = event.unicode

        if key == pg.K_ESCAPE:
            if self.promotion:
                self.promotion = None
                self.message = "Promotion cancelled."
            else:
                self.running = False
            return

        if key == pg.K_SPACE:
            self._do_capture()
            return

        # Promotion choice takes priority while pending.
        if self.promotion:
            if char in PROMOTION_MAP:
                self._finish_promotion(char)
            else:
                self.message = "Choose promotion: q / r / b / n (Esc cancels)."
            return

        if char == "f":
            self.session.toggle_mode()
            self.selected = None
            self.spawn = None
            self.message = f"Mode: {'LEGAL' if self.session.legal_mode else 'FREE'}"
            return

        if self.session.legal_mode:
            # In legal mode the letters are commands, not piece spawns.
            if char == "n":
                self.session.new_game()
                self.selected = None
                self.message = "New game."
            elif char == "r":
                self._do_recalibrate()
            elif char == "q":
                self.running = False
            return

        # Free-placement mode: letters spawn pieces, 'x' removes.
        if char == "x":
            square = self._square_at_pixel(pg.mouse.get_pos())
            if square:
                self.session.remove_piece(square)
                self.message = f"Removed piece on {square}."
            else:
                self.message = "Hover a square, then press x to remove."
        elif char in PIECE_SYMBOLS:
            self.spawn = char
            self.selected = None
            self.message = f"Spawn '{char}': click a square to place it."

    def _on_click(self, event) -> None:
        if event.button == 3:  # right-click cancels current selection
            self.selected = None
            self.spawn = None
            self.message = "Selection cancelled."
            return
        if event.button != 1:
            return

        square = self._square_at_pixel(event.pos)
        if square is None:
            return

        # Free-placement: if a spawn piece is armed, place it.
        if not self.session.legal_mode and self.spawn:
            self.session.place_piece(square, self.spawn)
            self.message = f"Placed '{self.spawn}' on {square}."
            return

        # Otherwise: click source, then target.
        if self.selected is None:
            self.selected = square
            self.message = f"Selected {square}."
            return

        source = self.selected
        self.selected = None
        if source == square:
            self.message = "Deselected."
            return

        if self.session.legal_mode:
            result = self.session.apply_legal_move(source, square)
            if result == "ok":
                self.message = f"Move {self.session.move_uci}."
            elif result == "need_promotion":
                self.promotion = (source, square)
                self.message = "Promotion: press q / r / b / n."
            else:
                self.message = f"Illegal move {source}{square}."
        else:
            if self.session.move_piece_free(source, square):
                self.message = f"Moved {source} -> {square}."
            else:
                self.message = f"No piece on {source}."

    def _finish_promotion(self, char: str) -> None:
        source, target = self.promotion
        self.promotion = None
        result = self.session.apply_legal_move(
            source, target, promotion=PROMOTION_MAP[char]
        )
        if result == "ok":
            self.message = f"Promoted: {self.session.move_uci}."
        else:
            self.message = "Promotion failed."

    def _do_capture(self) -> None:
        self.message = "Capturing..."
        self._draw()
        ok = self.session.capture()
        self.message = "Captured." if ok else "Capture failed (see console)."

    def _do_recalibrate(self) -> None:
        self.message = "Calibrating in the OpenCV window..."
        self._draw()
        ok = self.session.start_new_setup()
        if ok:
            self.selected = None
            self.spawn = None
            self.promotion = None
            self.message = f"New setup {self.session.setup_id} (position kept)."
        else:
            self.message = "Setup failed (see console). Press 'r' to retry."

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        # Try to establish an initial setup up front.
        if self.session.processor is None:
            self._draw()
            self.session.start_new_setup()

        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.KEYDOWN:
                    self._on_key(event)
                elif event.type == pg.MOUSEBUTTONDOWN:
                    self._on_click(event)
            self._draw()
            clock.tick(30)

        pg.quit()


def generate_data(
    config_path: Path = Path("config.yaml"),
    data_root: Path = DATA_ROOT,
    annotate_center: bool = False,
) -> None:
    """Entry point: open the virtual board and run the capture loop.

    ``annotate_center`` (default False) is passed through to calibration for every setup /
    recalibration in this session: True clicks the extended centre point, False interpolates it
    from the corners.
    """
    with ReachyMini(media_backend="default") as mini:
        session = DataGenerationSession(
            mini,
            config_path=Path(config_path),
            data_root=Path(data_root),
            annotate_center=annotate_center,
        )
        ui = BoardUI(session)
        ui.run()


if __name__ == "__main__":
    # Usage: uv run python -m chess_assistant.data_generation [--center]
    #
    # Opens the virtual board against the default config (config.yaml) and data root
    # (data/generated), and calibrates a first setup straight away. Pass --center to click the
    # extended centre point during every calibration instead of interpolating it from the four
    # corners. Requires the robot and a display.
    import sys

    generate_data(annotate_center="--center" in sys.argv[1:])
