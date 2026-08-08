"""
Generate the clips the robot speaks when it suggests a move, and cache them under .cache/speech.

    uv run python -m chess_assistant.pregenerate_speech
    uv run python -m chess_assistant.pregenerate_speech --voice am_michael

Run this once, before the first game. It synthesizes the ~150 fragments every move suggestion
is spliced from ("E2 to,", "E4?", "Castle kingside?", ...) so the game loop never has to run
Kokoro on the critical path.

Optional: skip it and the robot still plays. It just synthesizes each suggestion live instead,
which costs ~2.2s per suggested move. Takes 5-8 minutes, and is resumable -- a crash keeps
whatever it already baked, and re-running only fills the gaps. Re-run it after changing
speaker.voice, which invalidates the cache.
"""
import argparse

from omegaconf import OmegaConf

from chess_assistant.robot import DEFAULT_VOICE, synthesize
from chess_assistant.speech_clips import (
    CLIP_CACHE_DIR,
    KOKORO_SAMPLE_RATE,
    bake_clips,
    clip_texts,
    load_clips,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--voice", default=None, help="overrides speaker.voice from --config")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    voice = args.voice or config.get("speaker", {}).get("voice", DEFAULT_VOICE)
    # Kokoro's convention is that a voice's first letter is its language. Derived the same way
    # Speaker does it, so the cache we write is the cache the game reads.
    lang_code = voice[0]

    texts = clip_texts()
    already = len(load_clips(CLIP_CACHE_DIR, voice, lang_code, texts))
    print(f"speech: {voice} ({lang_code}) -- {already}/{len(texts)} clips already cached")
    if already == len(texts):
        print(f"speech: nothing to do. Cache: {(CLIP_CACHE_DIR / voice).as_posix()}")
        return

    # Imported late so --help doesn't wait on torch.
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=lang_code)
    clips = bake_clips(
        CLIP_CACHE_DIR, voice, lang_code, lambda text: synthesize(pipeline, text, voice=voice), texts
    )

    seconds = sum(len(clip) for clip in clips.values()) / KOKORO_SAMPLE_RATE
    print(
        f"\nspeech: {len(clips)} clips ({seconds:.1f}s of audio, {len(texts) - already} newly baked)"
    )
    print(f"speech: cached in {(CLIP_CACHE_DIR / voice).as_posix()}")
    empty = [key for key, clip in clips.items() if not len(clip)]
    if empty:
        print(f"speech: WARNING {len(empty)} clip(s) are silent and will fall back to live "
              f"synthesis: {', '.join(sorted(empty))}")


if __name__ == "__main__":
    main()
