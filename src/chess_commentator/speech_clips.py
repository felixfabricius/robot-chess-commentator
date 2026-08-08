"""
Pregenerated speech for the move suggestions: bake a small library of clips once, then
splice them together instead of running Kokoro on the critical path.

Everything the robot says when suggesting a move is drawn from a fixed vocabulary -- 64
squares, four promotion pieces, two castling sides -- so all of it can be synthesized ahead
of time. "E2 to E4?" is then two array lookups and a concatenate rather than ~2.2s of
synthesis while the players wait.

Reading and writing the cache are separate on purpose. bake_clips() synthesizes and is a setup
step (see pregenerate_speech); load_clips() only reads, takes no synthesizer, and is what the
game uses -- so a cold cache costs a warning and the old live-synthesis latency, never a
five-minute stall in the middle of starting a game.

The module deliberately imports neither kokoro nor reachy_mini: synthesis is injected into
bake_clips() as a plain `synth(text) -> waveform` callable. That is what lets the audition CLI
and the tests use this without a Speaker, a robot, or a torch model.

Assembly returns clip *keys*, never audio (announcement_parts -> ["origin/e2", "dest/e4"]).
Keeping that boundary is what makes the inventory a data decision: if spliced prosody ever
disappoints, moving to whole-phrase clips is a change to clip_texts() and
announcement_parts() and nothing else.
"""
import json
import os
from pathlib import Path

import numpy as np

from chess_assistant.config import SQUARES

KOKORO_SAMPLE_RATE = 24000

# Bump when a change to this module alters the *waveforms* rather than the texts (the per-clip
# text map below already catches text edits).
CACHE_VERSION = 1
CLIP_CACHE_DIR = Path(".cache/speech")

DEFAULT_SPLICE_GAP_MS = 60

# Silence trimming and joins. These are applied at load time, not bake time, so they are free
# to tune by ear without re-synthesizing anything -- see _finish().
_FRAME = 512      # 21ms analysis window
_HOP = 128        # 5.3ms resolution
_TOP_DB = 40.0    # keep every frame within 40dB of the loudest one
_PAD_MS = 20.0    # context kept either side, so onsets/offsets aren't clipped
_FADE_MS = 5.0
_SILENCE_FLOOR = 1e-4  # peak amplitude below this and the clip is nothing but noise

_PROMO_LETTERS = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}
_CASTLE_UCIS = {"e1g1": "kingside", "e1c1": "queenside", "e8g8": "kingside", "e8c8": "queenside"}

# Promotions only ever land on the back ranks.
PROMOTION_SQUARES = [square for square in SQUARES if square[1] in ("1", "8")]


def clip_texts() -> dict[str, str]:
    """The whole vocabulary: cache key -> exactly the text Kokoro is asked to say.

    The text doubles as the per-clip cache key, so editing one string here re-bakes that one
    clip and leaves the other 149 alone.
    """
    texts = {}
    for square in SQUARES:
        # Trailing comma, not a bare "E2 to": Kokoro conditions prosody on the whole input, so
        # a complete utterance gets a terminal fall, which is the wrong contour for a phrase
        # that carries on into the destination. A comma asks for a continuation instead. Its
        # own pause lands at the very end and is trimmed straight back off.
        texts[f"origin/{square}"] = f"{square.upper()} to,"
        texts[f"dest/{square}"] = f"{square.upper()}?"
    for square in PROMOTION_SQUARES:
        # Promotions put the question mark on the promotion piece instead, so the destination
        # needs a non-final variant.
        texts[f"dest_plain/{square}"] = f"{square.upper()},"
    for piece in _PROMO_LETTERS.values():
        texts[f"promo/{piece}"] = f"promoting to {piece.capitalize()}?"
    for side in ("kingside", "queenside"):
        texts[f"castle/{side}"] = f"Castle {side}?"
    return texts


def announcement_parts(uci: str, move_info: dict | None = None) -> list[str]:
    """Clip keys, in order, that spell out the spoken suggestion of `uci`."""
    origin, destination = uci[:2], uci[2:4]

    # Straight off the UCI: a length-5 move *is* a promotion and uci[4] *is* the piece. Reading
    # it from move_info instead would add a "Queen" -> "queen" normalisation and a way for the
    # two sources to disagree, and buy nothing.
    promotion = _PROMO_LETTERS.get(uci[4]) if len(uci) == 5 else None

    # Castling is the one thing the UCI genuinely cannot settle: "e1g1" is O-O, or a queen on
    # e1 walking to g1. move_info knows which; the UCI table is a best-effort guess for callers
    # that have no move_info (the audition CLI).
    castle = move_info.get("castle") if move_info is not None else _CASTLE_UCIS.get(uci)

    if castle:
        return [f"castle/{castle}"]
    if promotion:
        return [f"origin/{origin}", f"dest_plain/{destination}", f"promo/{promotion}"]
    return [f"origin/{origin}", f"dest/{destination}"]


def _frame_rms(audio: np.ndarray, frame: int = _FRAME, hop: int = _HOP) -> np.ndarray:
    if len(audio) < frame:
        return np.array([np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12)])
    # Slice the stride view before squaring: sliding_window_view itself is free, but squaring
    # the unsliced (len - frame + 1, frame) view would materialise ~70x more data than we read.
    windows = np.lib.stride_tricks.sliding_window_view(audio, frame)[::hop]
    return np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1) + 1e-12)


def trim(
    audio: np.ndarray,
    sample_rate: int = KOKORO_SAMPLE_RATE,
    top_db: float = _TOP_DB,
    pad_ms: float = _PAD_MS,
) -> np.ndarray:
    """Drop leading and trailing silence, keeping `pad_ms` of context either side.

    Frames rather than samples: voiced speech crosses zero a couple of hundred times a second,
    so any single-sample amplitude test flickers on and off through the middle of a word.
    Returns an empty array if the clip is silence throughout -- Kokoro producing nothing is a
    real (if rare) failure, and the caller says so rather than this raising from inside a bake.

    Detection is fuzzy by up to one frame either side: a window straddling the boundary still
    holds a fraction of the signal's energy, and a hundredth of it is only -20dB. That errs
    towards keeping audio, which is the right way to be wrong.
    """
    audio = np.asarray(audio, dtype=np.float32)
    # Checked in absolute terms, before anything relative to the peak: the peak of pure silence
    # is zero, so "within 40dB of the peak" is a condition every frame of it satisfies.
    if not len(audio) or float(np.max(np.abs(audio))) < _SILENCE_FLOOR:
        return audio[:0]

    rms = _frame_rms(audio)
    loud = np.flatnonzero(rms > rms.max() * 10.0 ** (-top_db / 20.0))
    pad = int(pad_ms * sample_rate / 1000)
    start = max(0, loud[0] * _HOP - pad)
    end = min(len(audio), loud[-1] * _HOP + _FRAME + pad)
    return audio[start:end]


def fade(
    audio: np.ndarray, sample_rate: int = KOKORO_SAMPLE_RATE, fade_ms: float = _FADE_MS
) -> np.ndarray:
    """Raised-cosine fade in and out, so a clip can be butted against silence without a click.

    Also guarantees every buffer starts and ends at zero, which play() quietly depends on:
    scipy.signal.resample is FFT-based and therefore treats the buffer as periodic, so nonzero
    edges ring.
    """
    audio = np.array(audio, dtype=np.float32, copy=True)
    n = min(int(fade_ms * sample_rate / 1000), len(audio) // 2)
    if n < 2:
        return audio
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))).astype(np.float32)
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


def concat(
    clips, gap_ms: float = DEFAULT_SPLICE_GAP_MS, sample_rate: int = KOKORO_SAMPLE_RATE
) -> np.ndarray:
    """Butt clips together with `gap_ms` of silence between them -- never after the last."""
    clips = [np.asarray(clip, dtype=np.float32) for clip in clips]
    if not clips:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(gap_ms * sample_rate / 1000), dtype=np.float32)
    pieces = [clips[0]]
    for clip in clips[1:]:
        pieces += [gap, clip]
    return np.concatenate(pieces)


def _read_manifest(voice_dir: Path) -> dict:
    try:
        with open(voice_dir / "manifest.json") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        # A manifest the user cannot reasonably be expected to reason about must never stand
        # between them and a game. Re-bake instead.
        return {}


def _write_manifest(voice_dir: Path, header: dict, texts: dict[str, str]) -> None:
    tmp = voice_dir / "manifest.json.tmp"
    with open(tmp, "w") as fh:
        json.dump({**header, "texts": texts}, fh, indent=2)
    os.replace(tmp, voice_dir / "manifest.json")


def _save_clip(path: Path, audio: np.ndarray) -> None:
    tmp = path.parent / (path.name + ".tmp")
    # A file object, not a path: np.save appends ".npy" to any *path* that doesn't already end
    # in it, so np.save(".../e2.npy.tmp", a) would quietly write ".../e2.npy.tmp.npy".
    with open(tmp, "wb") as fh:
        np.save(fh, audio.astype(np.float32))
    os.replace(tmp, path)


def _kokoro_versions() -> str:
    """Identify the synthesizer, so a library upgrade doesn't leave half the clips in the old
    voice and half in the new one -- a silent prosody mismatch at a join, which is far worse to
    debug than a crash."""
    try:
        import kokoro
        import misaki

        return f"kokoro={kokoro.__version__},misaki={misaki.__version__}"
    except (ImportError, AttributeError):
        return "unknown"


def cache_header(voice: str, lang_code: str, speed: float = 1.0) -> dict:
    """Everything that invalidates the whole cache at once, as opposed to a single clip."""
    return {
        "cache_version": CACHE_VERSION,
        "voice": voice,
        "lang_code": lang_code,
        "sample_rate": KOKORO_SAMPLE_RATE,
        "speed": speed,
        "kokoro_versions": _kokoro_versions(),
    }


def _cache_state(cache_dir, voice: str, lang_code: str, texts: dict[str, str]):
    """Read the cache: what's usable already, and what still needs synthesizing.

    Deliberately does not create anything -- load_clips() reads through here, and a read must
    not conjure up the directory it is reading. A cache_dir that isn't there simply yields
    nothing usable and everything stale.
    """
    voice_dir = Path(cache_dir) / voice

    header = cache_header(voice, lang_code)
    manifest = _read_manifest(voice_dir)
    fresh = all(manifest.get(key) == value for key, value in header.items())
    cached = dict(manifest.get("texts", {})) if fresh else {}

    raw, stale = {}, []
    for key, text in texts.items():
        path = voice_dir / f"{key}.npy"
        if cached.get(key) == text and path.exists():
            try:
                raw[key] = np.load(path).astype(np.float32)
                continue
            except (ValueError, OSError):
                pass  # truncated or half-written -- just bake it again
        stale.append((key, text, path))
    return voice_dir, header, cached, raw, stale


def _finish(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Raw Kokoro output -> playable clips.

    Kept out of the bake so what lands on disk stays raw, which is what makes top_db / pad_ms /
    fade_ms tunable by ear with no re-bake and no manifest fields -- for ~10MB of disk and
    ~100ms of startup.
    """
    clips = {}
    for key, audio in raw.items():
        clip = fade(trim(audio))
        if not len(clip):
            print(f"speech: WARNING {key!r} trimmed to nothing -- Kokoro produced silence")
        clips[key] = clip
    return clips


def load_clips(
    cache_dir, voice: str, lang_code: str, texts: dict[str, str] | None = None
) -> dict[str, np.ndarray]:
    """Return {clip key: playable waveform} for whatever is already cached.

    Takes no `synth`, so it cannot synthesize: a clip that isn't cached is simply absent from
    the result, and the caller decides what that means. This is the path the game uses --
    baking 150 clips is a setup step (see pregenerate_speech), not something to discover
    halfway through starting a game.
    """
    texts = texts if texts is not None else clip_texts()
    _, _, _, raw, _ = _cache_state(cache_dir, voice, lang_code, texts)
    return _finish(raw)


def bake_clips(
    cache_dir, voice: str, lang_code: str, synth, texts: dict[str, str] | None = None
) -> dict[str, np.ndarray]:
    """Synthesize whatever isn't cached, then return the full library.

    `synth` is a `text -> float32 waveform` callable, injected so this module never has to
    import kokoro. Slow by nature (~5-8 minutes for all 150 clips) but done once per voice,
    ever, and resumable: a crash keeps everything already baked.
    """
    texts = texts if texts is not None else clip_texts()
    voice_dir = Path(cache_dir) / voice
    voice_dir.mkdir(parents=True, exist_ok=True)

    _, header, cached, raw, stale = _cache_state(cache_dir, voice, lang_code, texts)

    if stale:
        print(f"speech: baking {len(stale)} clip(s) for {voice} -- the first run takes a few minutes")
        # Drop keys we no longer know about, so a renamed clip doesn't leave its manifest entry
        # behind to confuse the next reader.
        cached = {key: value for key, value in cached.items() if key in texts}
        for i, (key, text, path) in enumerate(stale, 1):
            audio = np.asarray(synth(text), dtype=np.float32)
            if not len(audio):
                raise RuntimeError(f"Kokoro returned no audio for {key!r} ({text!r})")
            path.parent.mkdir(parents=True, exist_ok=True)
            _save_clip(path, audio)
            raw[key] = audio
            cached[key] = text
            # Per clip, not once at the end: a crash on clip 90 must not throw away the 89
            # before it.
            _write_manifest(voice_dir, header, cached)
            print(f"speech: [{i}/{len(stale)}] {key} -- {text!r}")

    return _finish(raw)
