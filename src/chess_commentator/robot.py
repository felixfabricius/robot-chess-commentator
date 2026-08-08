import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import chess
import anthropic
from scipy.signal import resample
from kokoro import KPipeline
from reachy_mini import ReachyMini

from dotenv import load_dotenv

from chess_commentator.speech_clips import (
    CLIP_CACHE_DIR,
    DEFAULT_SPLICE_GAP_MS,
    KOKORO_SAMPLE_RATE,
    announcement_parts,
    clip_texts,
    concat,
    load_clips,
)

load_dotenv()

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 64
DEFAULT_VOICE = "bm_george"

# The closing roast is a few sentences rather than a one-liner, so it cannot share
# DEFAULT_MAX_TOKENS -- 64 would cut it off mid-sentence.
DEFAULT_OUTRO_MAX_TOKENS = 200
DEFAULT_OUTRO_MAX_WORDS = 60

SYSTEM_PROMPT = (
    "You are a funny chess commentator. I will provide you with context about a chess move and want you to return a one-liner. "
    "Goal: entertainment. If possible, make the comment funny. But DON'T FORCE IT. Sometimes simple, neutral comments might be better. "
    "References to chess culture (Magnus Carlsen, Hikaru as successful player examples); 'Botez Gambit' when blundering a good piece; "
    "names of openings; are welcome! "
    "I will provide you with a move, the moved piece, the move turn (e.g. if 1st turn: can comment on game starting), whether it was "
    "white or black's turn, the centipawn loss associated with that move "
    "(higher means move was worse; best is zero; anything above 100 is poor, anything above 300 a clear blunder), "
    "and potential extra move indicators (e.g. capture? castle? etc). "
    "I will also provide you with some info about the recent move history within the game: actual recent moves, number of subsequent moves with "
    "(aggressive playing) / without (potential pacifism comment) "
    "capture, and average and recent centipawn loss (can joke about it being very high (poor chess game), or low). "
    "Additionally, I will provide a list of the recent comments in this game, ordered by turn (earlier comments earlier), and by whose "
    "turn it was during that comment. There might be potential for a funny callback; try avoid unintentionally repeating comments / phrases, though. "
    "Also DO NOT FORCE callbacks, or refer to the same theme TOO often. "
)
PROMPT_END = (
    "Based on this, return a one-liner. Applaud solid moves, and roast bad ones. "
    "Do NOT explicitly mention the centipawn loss number or centipawn loss concept. Listeners might not understand. "
    # Kokoro synthesis runs at roughly 0.6x realtime, so every spoken second costs ~0.6s of
    # synthesis. Keeping the comment short is what keeps it inside the move-review window --
    # and a rambling 'one-liner' is not a one-liner anyway.
    "Hard limit: at most 20 words, one sentence. Be punchy; cut any word that isn't earning its place. "
    "Return only the comment, with no preamble."
)

OUTRO_SYSTEM_PROMPT = (
    "You are a funny British chess commentator, delivering the closing summary of a chess game "
    "that has just ended. I will give you the final result, the full move list with the centipawn "
    "loss of every move, the average centipawn loss per side, the worst blunder of the game, and "
    "every comment you made during the game. "
    "Write a short closing roast of the game as a whole. It must: "
    "(1) refer to at least one specific thing that actually happened in THIS game -- a named move, "
    "the worst blunder, the opening, or a callback to one of your own earlier comments; "
    "(2) judge the overall quality of play from the average centipawn loss (higher is worse; under "
    "30 is decent, over 100 is a bad game, a single move over 300 is a catastrophe) -- but read "
    "those numbers silently and translate the verdict into plain English; "
    "(3) end on the note that you are relieved this is finally over and would rather not be made to "
    "do this again. Weary, deadpan, affectionate -- exhausted rather than cruel. "
    "References to chess culture (Magnus Carlsen, Hikaru, the 'Botez Gambit', opening names) are "
    "welcome. Do not repeat your earlier comments verbatim. "
    # The numbers are in the prompt and the model reaches for them unprompted -- "a 412-point
    # catastrophe" is not a sentence anyone says out loud. Say the move was terrible instead.
    "Never say a centipawn number aloud, and never use the phrase 'centipawn' or refer to the "
    "concept: this is spoken to people in a room, and they have no idea what it means. Say 'a "
    "catastrophe' or 'a howler', not 'a 412-point blunder'."
)
OUTRO_PROMPT_END = (
    "Based on this, deliver the closing summary. "
    # Same arithmetic as PROMPT_END, with a longer budget: this one is allowed to be a few
    # sentences, but it is still spoken aloud at ~0.6x realtime while the robot dances.
    "Hard limit: at most {max_words} words, three or four sentences. "
    "The final sentence must land the 'thank god that is over, please do not make me do this again' "
    "note. Return only the summary, with no preamble."
)


def format_uci_for_speech(uci_move: str) -> str:
    """Turn 'e2e4' into 'E2 to E4'.

    Spelled out, the move is said clearly; fed in raw it gets mangled into one
    run-together token.
    """
    origin, destination = uci_move[:2], uci_move[2:4]
    return f"{origin.upper()} to {destination.upper()}"


def synthesize(pipeline, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> np.ndarray:
    """Run text through Kokoro and return one concatenated float32 waveform."""
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio.numpy() if hasattr(audio, "numpy") else audio)
    return np.concatenate(chunks)


def play(mini: ReachyMini, audio: np.ndarray, source_rate: int = KOKORO_SAMPLE_RATE):
    """Play an already-synthesized waveform through Reachy Mini's speaker.

    Kept separate from synthesize() so a worker thread can do the (slow) synthesis
    while playback stays on the main thread -- mini.media must only ever be touched
    from one thread.
    """
    target_rate = mini.media.get_output_audio_samplerate()
    if target_rate != source_rate:
        audio = resample(audio, int(len(audio) * target_rate / source_rate))

    mini.media.start_playing()
    mini.media.push_audio_sample(audio.astype(np.float32))
    time.sleep(len(audio) / target_rate)
    mini.media.stop_playing()


def say(mini: ReachyMini, pipeline, text: str, voice: str = DEFAULT_VOICE):
    """Synthesize text with Kokoro and play it through Reachy Mini's speaker."""
    play(mini, synthesize(pipeline, text, voice=voice))


def _yes_no(value) -> str:
    return "yes" if value else "no"


def _render_comment_history(comment_history: list, empty: str) -> list:
    """The "Turn; side; move; comment" lines both prompts ask for, or `empty` if there
    are none -- a bare 'None' or a blank section must never reach the model."""
    if not comment_history:
        return [empty]
    return [
        f"Turn {entry['turn']}; {entry['side']}; {entry['move']}; \"{entry['comment']}\""
        for entry in comment_history
    ]


def build_prompt(move_info: dict, cp_loss: int, history: dict, comment_history: list) -> str:
    """Assemble the user turn: just the facts. The instructions live in SYSTEM_PROMPT."""
    capture = "no"
    if move_info["capture"]:
        captured = move_info.get("captured_piece") or "a piece"
        capture = f"yes - took a {captured}"
        if move_info["en_passant"]:
            capture += " (en passant)"

    lines = [
        f"Move: {move_info['san']} ({move_info['move']})",
        f"Moved piece: {move_info['moved_piece']}",
        f"Turn: {move_info['move_number']}, {move_info['turn']} to move",
        f"Centipawn loss: {cp_loss}",
        f"Capture: {capture}",
        f"Castle: {move_info['castle'] or 'no'}",
        f"En passant: {_yes_no(move_info['en_passant'])}",
        f"Promotion: {move_info['promotion'] or 'no'}",
        f"Check: {_yes_no(move_info['check'])}",
        f"Checkmate: {_yes_no(move_info['checkmate'])}",
    ]

    recent_moves = history["recent_moves"]
    averages = history["average_cp_loss"]
    last_losses = history["last_cp_losses"]
    lines += [
        "",
        "Game history:",
        f"Recent moves: {' '.join(recent_moves) if recent_moves else '(none - this is the first move)'}",
        f"Consecutive captures: {history['capture_streak']}",
        f"Consecutive quiet moves: {history['quiet_streak']}",
        f"Average centipawn loss - white: {averages['white']:.1f}, black: {averages['black']:.1f}",
        f"Last centipawn losses - white: {last_losses['white']}, black: {last_losses['black']}",
    ]

    lines += ["", "Comment history:"]
    lines += _render_comment_history(
        comment_history, "(none yet - this is the first comment of the game)"
    )

    lines += ["", PROMPT_END]
    return "\n".join(lines)


def build_outro_prompt(
    summary: dict, comment_history: list, max_words: int = DEFAULT_OUTRO_MAX_WORDS
) -> str:
    """Assemble the user turn for the closing roast: just the facts, like build_prompt().

    `summary` is ChessGame.final_snapshot(). Unlike the per-move prompt this gets the whole
    game rather than a recent window -- the roast is meant to look back over all of it.
    """
    averages = summary["average_cp_loss"]
    winner = summary["winner"] or "nobody - it was a draw"

    lines = [
        "Game over.",
        f"Result: {summary['result']} ({summary['termination']})",
        f"Winner: {winner}",
        f"Total moves: {summary['total_moves']} ({summary['total_plies']} plies)",
        f"Captures: {summary['captures']}",
        f"Average centipawn loss - white: {averages['white']:.1f}, black: {averages['black']:.1f}",
    ]

    blunder = summary["worst_blunder"]
    if blunder:
        lines.append(
            f"Worst blunder: turn {blunder.get('move_number')}, {blunder['turn']}, "
            f"{blunder['san']}, {blunder['cp_loss']} centipawns lost"
        )
    else:
        lines.append("Worst blunder: (none - no moves were played)")

    lines += ["", "Full move list (turn; side; move; centipawn loss):"]
    if summary["moves"]:
        for entry in summary["moves"]:
            lines.append(
                f"{entry.get('move_number')}; {entry['turn']}; {entry['san']}; {entry['cp_loss']}"
            )
    else:
        lines.append("(none - no moves were played)")

    lines += ["", "Comment history:"]
    lines += _render_comment_history(
        comment_history, "(none - you did not comment on this game)"
    )

    lines += ["", OUTRO_PROMPT_END.format(max_words=max_words)]
    return "\n".join(lines)


class Speaker:
    """The robot's voice: Claude writes the commentary, Kokoro speaks it.

    Also owns the pregeneration pool. A comment for a candidate move is generated on a worker
    thread while the players are still deciding whether to accept that move. This reduces / eliminate
    wait times due to engine, Claude and Kokoro latency.
    """

    def __init__(self, mini, config=None):
        speaker_config = config.get("speaker", {}) if config is not None else {}
        outro_config = speaker_config.get("outro", {}) or {}

        self.mini = mini
        self.model = speaker_config.get("model", DEFAULT_MODEL)
        self.max_tokens = speaker_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.voice = speaker_config.get("voice", DEFAULT_VOICE)
        self.n_recent_moves = speaker_config.get("recent_moves", 6)
        self.n_recent_cp_losses = speaker_config.get("recent_cp_losses", 5)
        self.splice_gap_ms = speaker_config.get("splice_gap_ms", DEFAULT_SPLICE_GAP_MS)
        self.outro_max_tokens = outro_config.get("max_tokens", DEFAULT_OUTRO_MAX_TOKENS)
        self.outro_max_words = outro_config.get("max_words", DEFAULT_OUTRO_MAX_WORDS)

        # Kokoro's convention is that a voice's first letter is its language: "bm_george" is
        # British, "am_michael" American. Deriving it keeps the two in step -- hardcoding "b"
        # while leaving voice configurable buys you an American voice read by a British G2P.
        lang_code = self.voice[0]
        self.pipeline = KPipeline(lang_code=lang_code)
        self.client = anthropic.Anthropic()

        # Turn-ordered so the "Turn; side; move; comment" lines the prompt asks for fall
        # straight out of it.
        self.comment_history = []

        # move_uci -> Future[{"comment", "audio", "cp_loss", "new_score"}], populated
        # during the move-review window and consumed the moment a move is accepted.
        self.pregenerated_comments = {}
        self.turn = 0

        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pregen")
        # KPipeline is a torch model with no thread-safety guarantees, and a worker may be
        # synthesizing a comment while the main thread falls back to live synthesis.
        self.pipeline_lock = threading.Lock()

        # Read-only: generating these is a setup step (see pregenerate_speech), not something
        # to discover eight minutes into starting a game. Every move suggestion is spliced from
        # them, so nothing is synthesized on the critical path -- which leaves the lock above
        # free for the comment worker all through the review window, the runway main.py banks on.
        self.clips = load_clips(CLIP_CACHE_DIR, self.voice, lang_code)
        self._warn_about_missing_clips()

    def _warn_about_missing_clips(self) -> None:
        """Say so, once, at startup. A cold cache is not fatal -- it just costs ~2.2s of Kokoro
        per suggestion, exactly as it did before the clips existed."""
        expected = clip_texts()
        missing = set(expected) - set(self.clips)
        if not missing:
            return
        print(f"speech: {len(missing)}/{len(expected)} clips missing for {self.voice}.")
        print("speech: falling back to live synthesis (~2.2s per suggestion). To remove that "
              "latency, run:")
        print(f"speech:   uv run python -m chess_commentator.pregenerate_speech --voice {self.voice}")

    def _synthesize(self, text: str) -> np.ndarray:
        with self.pipeline_lock:
            return synthesize(self.pipeline, text, voice=self.voice)

    def suggest_move(self, move, move_info=None):
        """Speak a candidate move aloud, e.g. "E2 to E4?".

        `move_info` settles what the UCI cannot: "e1g1" is castling, or a queen walking from
        e1 to g1. Without it, the four castling UCIs are assumed to be castles.
        """
        parts = announcement_parts(move, move_info)
        if any(key not in self.clips for key in parts):
            # Already reported once at startup, so don't repeat it every move: self.clips never
            # shrinks, which makes that warning complete and authoritative.
            play(self.mini, self._synthesize(f"{format_uci_for_speech(move)}?"))
            return
        play(self.mini, concat([self.clips[key] for key in parts], self.splice_gap_ms))

    def pregenerate_comment(self, move: dict, turn: int, game) -> None:
        """Kick off comment generation for a candidate move and return immediately.

        Called right after the candidate is spoken aloud, so the engine analysis, the
        Claude call and the Kokoro synthesis all happen while the players are using the
        review window to accept or reject the suggestion.
        """
        if turn > self.turn:  # new turn -> last turn's candidates are dead
            self.pregenerated_comments = {}
            self.turn = turn

        move_uci = move["move"]
        if move_uci in self.pregenerated_comments:
            return  # already generated (or generating) for this candidate

        move_info = move["move_info"]
        # Snapshot on the main thread: the worker must not read move_log / comment_history
        # while a later move is being appended to them.
        history = game.history_snapshot(
            recent_moves=self.n_recent_moves,
            recent_cp_losses=self.n_recent_cp_losses,
        )
        comment_history = list(self.comment_history)

        self.pregenerated_comments[move_uci] = self.executor.submit(
            self._generate, game, move_uci, move_info, history, comment_history
        )

    def _generate(self, game, move_uci, move_info, history, comment_history):
        """Runs on a worker thread. Never touches mini.media."""
        cp_loss, new_score = game.cp_loss_for(move_uci)

        prompt = build_prompt(move_info, cp_loss, history, comment_history)
        message = self.client.messages.create(
            model="claude-sonnet-5",
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        comment = next(block.text for block in message.content if block.type == "text")

        return {
            "comment": comment,
            "audio": self._synthesize(comment),
            "cp_loss": cp_loss,
            "new_score": new_score,
        }

    def comment_on_move(self, move_uci: str, move_info: dict, game):
        """Speak the comment for an accepted move and record it.

        Returns (cp_loss, new_score) so the caller can pass them to game.apply_move()
        rather than paying for a second Stockfish analysis of the same position.
        """
        future = self.pregenerated_comments.get(move_uci)
        if future is None:
            # Never pregenerated (shouldn't normally happen) - generate inline.
            history = game.history_snapshot(
                recent_moves=self.n_recent_moves,
                recent_cp_losses=self.n_recent_cp_losses,
            )
            result = self._generate(
                game, move_uci, move_info, history, list(self.comment_history)
            )
        else:
            # Blocks only if the worker hasn't finished yet, i.e. no worse than the old
            # fully-synchronous path.
            result = future.result()

        play(self.mini, result["audio"])

        self.comment_history.append({
            "turn": move_info["move_number"],
            "side": move_info["turn"],
            "move": move_info["move"],
            "comment": result["comment"],
        })

        return result["cp_loss"], result["new_score"]

    def exclaim_win(self, game):
        winner = "black" if game.board.turn == chess.WHITE else "white"
        play(self.mini, self._synthesize(f"{winner} won!!!"))

    def pregenerate_outro(self, game):
        """Kick off the closing roast and return its Future immediately.

        Called the moment the game ends, so Claude and Kokoro work while the robot is
        still exclaiming the win and dancing -- see outro.finale(), which uses this
        future's done() as the signal to stop dancing.
        """
        # The last turn's rejected candidates may still be sitting in the queue, and there
        # are only two workers. Nothing is going to consume those comments now -- the game
        # is over -- so drop them rather than let the outro wait behind a Kokoro synthesis.
        for future in self.pregenerated_comments.values():
            future.cancel()
        self.pregenerated_comments = {}

        # Snapshot on the main thread, as ever: the worker must not read move_log or
        # comment_history directly.
        summary = game.final_snapshot()
        comment_history = list(self.comment_history)

        return self.executor.submit(self._generate_outro, summary, comment_history)

    def _generate_outro(self, summary, comment_history):
        """Runs on a worker thread. Never touches mini.media."""
        prompt = build_outro_prompt(summary, comment_history, self.outro_max_words)
        message = self.client.messages.create(
            model="claude-sonnet-5",
            # Not self.max_tokens: that is the one-liner cap, and it would guillotine the
            # roast mid-sentence.
            max_tokens=self.outro_max_tokens,
            system=OUTRO_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        comment = next(block.text for block in message.content if block.type == "text")

        return {"comment": comment, "audio": self._synthesize(comment)}

    def speak_outro(self, future):
        """Speak the closing roast. Blocks only if the worker is not done yet."""
        result = future.result()
        play(self.mini, result["audio"])
        return result["comment"]

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
