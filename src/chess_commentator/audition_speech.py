"""
Bake the move-suggestion clip library and write spliced samples to WAV, so the splice gap can
be tuned by ear without the robot.

    uv run python -m chess_assistant.audition_speech --sweep
    uv run python -m chess_assistant.audition_speech --gaps 0,40,60,80,120 --moves e2e4,e1g1
    uv run python -m chess_assistant.audition_speech --raw --moves e2e4

Needs Kokoro, but no robot and no Anthropic key. On a cold cache the first run bakes all 150
clips (~5-8 minutes); after that it starts instantly and writes WAVs into .cache/speech/audition.

Tune --top-db before the gap. The two trade off against each other -- too high leaves breath
noise in and the gap sounds long and mushy, too low eats the tail of "E2 to," and it sounds
chopped -- and chasing both at once gets you nowhere.
"""
import argparse

import numpy as np
from scipy.io import wavfile

from chess_assistant.robot import DEFAULT_VOICE, synthesize
from chess_assistant.speech_clips import (
    CLIP_CACHE_DIR,
    DEFAULT_SPLICE_GAP_MS,
    KOKORO_SAMPLE_RATE,
    announcement_parts,
    bake_clips,
    concat,
    fade,
    trim,
)

DEFAULT_MOVES = ["e2e4", "g1f3", "e1g1", "e8c8", "e7e8q", "a2a1n"]


def write_wav(path, audio: np.ndarray) -> None:
    """int16, not float32: float WAV is valid but not every player copes with it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, KOKORO_SAMPLE_RATE, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--moves", default=",".join(DEFAULT_MOVES), help="comma-separated UCI moves")
    parser.add_argument("--gaps", default="0,40,60,80,120", help="comma-separated gaps in ms")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="one file per move holding every gap back-to-back. A/B-ing across separate files "
             "is hopeless; hearing them in sequence makes the answer obvious in one listen.",
    )
    parser.add_argument("--top-db", type=float, default=40.0, help="silence-trim threshold")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="also dump each clip untrimmed, to tell 'top-db ate the tail' apart from 'Kokoro's "
             "prosody is wrong' -- two hypotheses with different fixes",
    )
    args = parser.parse_args()

    moves = [move.strip() for move in args.moves.split(",") if move.strip()]
    gaps = [float(gap) for gap in args.gaps.split(",") if gap.strip()]
    lang_code = args.voice[0]

    from kokoro import KPipeline  # imported late: the arg parse above shouldn't wait on torch

    pipeline = KPipeline(lang_code=lang_code)
    clips = bake_clips(
        CLIP_CACHE_DIR, args.voice, lang_code, lambda text: synthesize(pipeline, text, voice=args.voice)
    )
    # Re-trim at the requested threshold: bake_clips caches raw audio and trims at its own
    # default, so --top-db needs to be applied over the top rather than through it.
    if args.top_db != 40.0:
        voice_dir = CLIP_CACHE_DIR / args.voice
        clips = {
            key: fade(trim(np.load(voice_dir / f"{key}.npy"), top_db=args.top_db))
            for key in clips
        }

    out = CLIP_CACHE_DIR / "audition"
    silence = np.zeros(KOKORO_SAMPLE_RATE, dtype=np.float32)  # 1s between sweep variants

    for move in moves:
        parts = announcement_parts(move)
        # A clip Kokoro botched shows up instantly as an outlier in a column of durations.
        print(f"\n{move}: {' + '.join(parts)}")
        for key in parts:
            print(f"  {key:<18} {len(clips[key]) / KOKORO_SAMPLE_RATE:.3f}s")

        if args.sweep:
            pieces = []
            for gap in gaps:
                pieces += [concat([clips[key] for key in parts], gap), silence]
            write_wav(out / f"{move}_sweep.wav", np.concatenate(pieces))
            print(f"  -> {move}_sweep.wav (gaps: {', '.join(str(g) for g in gaps)}ms, in order)")
        else:
            for gap in gaps:
                write_wav(out / f"{move}_gap{gap:g}.wav", concat([clips[key] for key in parts], gap))
            print(f"  -> {move}_gap*.wav")

        if args.raw:
            for key in parts:
                path = CLIP_CACHE_DIR / args.voice / f"{key}.npy"
                write_wav(out / f"raw_{key.replace('/', '_')}.wav", np.load(path))
            print("  -> raw_*.wav (untrimmed)")

    print(f"\nWrote to {out.as_posix()}")
    print(f"Pick a gap, then set speaker.splice_gap_ms in config.yaml (currently defaults to "
          f"{DEFAULT_SPLICE_GAP_MS}).")


if __name__ == "__main__":
    main()
