import json

import numpy as np
import pytest

from chess_commentator.config import SQUARES
from chess_commentator.speech_clips import (
    KOKORO_SAMPLE_RATE,
    PROMOTION_SQUARES,
    announcement_parts,
    bake_clips,
    clip_texts,
    concat,
    fade,
    load_clips,
    trim,
)


def tone(seconds=0.1, hz=180.0, amplitude=0.5):
    t = np.arange(int(seconds * KOKORO_SAMPLE_RATE), dtype=np.float32) / KOKORO_SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


### Inventory and assembly

def test_every_key_assembly_can_emit_exists_in_the_inventory():
    """The one test standing between a typo in a key string and a silent fallback to live
    synthesis in the middle of a game. Cheap: 4k iterations of a pure function."""
    inventory = set(clip_texts())

    emitted = set()
    for origin in SQUARES:
        for destination in SQUARES:
            uci = origin + destination
            emitted.update(announcement_parts(uci))  # UCI-only: castling UCIs read as castles
            emitted.update(announcement_parts(uci, {"castle": None}))
            emitted.update(announcement_parts(uci, {"castle": "kingside"}))
            emitted.update(announcement_parts(uci, {"castle": "queenside"}))
    for destination in PROMOTION_SQUARES:
        for piece in "qrbn":
            emitted.update(announcement_parts(f"e7{destination}{piece}", {"castle": None}))

    assert emitted <= inventory


def test_inventory_is_exactly_the_150_clips_we_expect():
    texts = clip_texts()
    assert len(texts) == 64 + 64 + 16 + 4 + 2
    assert texts["origin/e2"] == "E2 to,"
    assert texts["dest/e4"] == "E4?"
    assert texts["dest_plain/e8"] == "E8,"
    assert texts["promo/queen"] == "promoting to Queen?"
    assert texts["castle/kingside"] == "Castle kingside?"


def test_promotion_squares_are_the_back_ranks_only():
    assert len(PROMOTION_SQUARES) == 16
    assert all(square[1] in ("1", "8") for square in PROMOTION_SQUARES)


def test_normal_move_splices_origin_and_destination():
    assert announcement_parts("e2e4", {"castle": None}) == ["origin/e2", "dest/e4"]


def test_castling_is_announced_as_castling():
    assert announcement_parts("e1g1", {"castle": "kingside"}) == ["castle/kingside"]
    assert announcement_parts("e8c8", {"castle": "queenside"}) == ["castle/queenside"]


def test_castle_detection_prefers_move_info_over_the_uci():
    """The one ambiguity the whole move_info plumbing exists to resolve: "e1g1" is O-O, or a
    queen on e1 walking to g1. Get this wrong and a queen move announces "Castle kingside?"."""
    assert announcement_parts("e1g1", {"castle": None}) == ["origin/e1", "dest/g1"]
    assert announcement_parts("e1g1") == ["castle/kingside"]  # no move_info -- best-effort guess


def test_promotion_names_the_piece():
    """e7e8q and e7e8n must not sound identical -- the players are being asked to confirm one."""
    assert announcement_parts("e7e8q") == ["origin/e7", "dest_plain/e8", "promo/queen"]
    assert announcement_parts("e7e8n") == ["origin/e7", "dest_plain/e8", "promo/knight"]
    assert announcement_parts("a2a1r") == ["origin/a2", "dest_plain/a1", "promo/rook"]


def test_promotion_is_read_off_the_uci_without_move_info():
    """A length-5 UCI *is* a promotion and uci[4] *is* the piece, so move_info adds nothing."""
    assert announcement_parts("e7e8b", {"castle": None}) == announcement_parts("e7e8b")


### Trim, fade, concat

def test_trim_recovers_the_signal_from_surrounding_silence():
    silence = np.zeros(int(0.5 * KOKORO_SAMPLE_RATE), dtype=np.float32)
    signal = tone(seconds=1.0)
    trimmed = trim(np.concatenate([silence, signal, silence]))

    # Never less than the signal, and never more than the 20ms pad plus a frame of straddle
    # either side. Erring long is the right way to be wrong -- erring short clips a consonant.
    pad, frame = int(20.0 * KOKORO_SAMPLE_RATE / 1000), 512
    assert len(signal) <= len(trimmed) <= len(signal) + 2 * (pad + frame)


def test_trim_is_stable_under_reapplication():
    """Not exactly idempotent -- re-trimming shifts the analysis grid, so the edges move by a
    hop or so. What matters is that it converges instead of nibbling the clip away."""
    padded = np.concatenate([np.zeros(1000, dtype=np.float32), tone(), np.zeros(1000, dtype=np.float32)])
    once = trim(padded)
    twice = trim(once)

    assert 0 <= len(once) - len(twice) <= 128  # one hop


def test_trim_of_pure_silence_is_empty_rather_than_an_exception():
    """Kokoro producing nothing is rare but real. _finish() warns; it must not blow up."""
    assert len(trim(np.zeros(4800, dtype=np.float32))) == 0


def test_fade_zeroes_the_endpoints_and_leaves_the_interior_alone():
    faded = fade(tone(seconds=0.5))
    assert abs(faded[0]) < 1e-6
    assert abs(faded[-1]) < 1e-6

    n = int(5.0 * KOKORO_SAMPLE_RATE / 1000)
    assert np.array_equal(faded[n:-n], tone(seconds=0.5)[n:-n])


def test_concat_puts_the_gap_between_clips():
    a, b = tone(), tone()
    joined = concat([a, b], gap_ms=60)
    gap_samples = int(60 * KOKORO_SAMPLE_RATE / 1000)

    assert len(joined) == len(a) + len(b) + gap_samples
    assert np.array_equal(joined[len(a):len(a) + gap_samples], np.zeros(gap_samples, dtype=np.float32))


def test_concat_of_a_single_clip_adds_no_trailing_gap():
    """The gap goes *between* clips, never after the last one -- castling is a single clip, and
    it must not end in dead air."""
    a = tone()
    assert len(concat([a], gap_ms=60)) == len(a)


def test_concat_with_no_gap_is_a_plain_join():
    a, b = tone(), tone()
    assert len(concat([a, b], gap_ms=0)) == len(a) + len(b)


def test_concat_of_nothing_is_empty():
    assert len(concat([], gap_ms=60)) == 0


### Cache

class CountingSynth:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return tone()


TEXTS = {"origin/e2": "E2 to,", "dest/e4": "E4?", "castle/kingside": "Castle kingside?"}


@pytest.fixture
def synth():
    return CountingSynth()


def bake(cache_dir, synth, voice="bm_george", texts=None):
    return bake_clips(cache_dir, voice, "b", synth, texts=texts if texts is not None else TEXTS)


def load(cache_dir, voice="bm_george", texts=None):
    return load_clips(cache_dir, voice, "b", texts=texts if texts is not None else TEXTS)


def test_bake_synthesizes_every_clip_and_returns_playable_audio(tmp_path, synth):
    clips = bake(tmp_path, synth)

    assert sorted(synth.calls) == sorted(TEXTS.values())
    assert set(clips) == set(TEXTS)
    assert all(len(clip) > 0 for clip in clips.values())


def test_a_warm_cache_synthesizes_nothing(tmp_path, synth):
    bake(tmp_path, synth)
    synth.calls.clear()

    bake(tmp_path, synth)
    assert synth.calls == []


def test_changing_one_text_rebakes_exactly_that_clip(tmp_path, synth):
    bake(tmp_path, synth)
    synth.calls.clear()

    bake(tmp_path, synth, texts={**TEXTS, "dest/e4": "E4, please?"})
    assert synth.calls == ["E4, please?"]


def test_a_different_voice_gets_its_own_cache(tmp_path, synth):
    bake(tmp_path, synth)
    synth.calls.clear()

    bake(tmp_path, synth, voice="am_michael")
    assert sorted(synth.calls) == sorted(TEXTS.values())


def test_a_deleted_clip_is_rebaked_on_its_own(tmp_path, synth):
    bake(tmp_path, synth)
    synth.calls.clear()

    (tmp_path / "bm_george" / "dest" / "e4.npy").unlink()
    bake(tmp_path, synth)
    assert synth.calls == ["E4?"]


def test_a_truncated_clip_is_rebaked_rather_than_raising(tmp_path, synth):
    bake(tmp_path, synth)
    synth.calls.clear()

    (tmp_path / "bm_george" / "dest" / "e4.npy").write_bytes(b"not an npy file")
    bake(tmp_path, synth)
    assert synth.calls == ["E4?"]


def test_a_corrupt_manifest_rebakes_everything_without_raising(tmp_path, synth):
    """A cache file the user cannot reasonably reason about must never block a game."""
    bake(tmp_path, synth)
    synth.calls.clear()

    (tmp_path / "bm_george" / "manifest.json").write_text("{ this is not json")
    bake(tmp_path, synth)
    assert sorted(synth.calls) == sorted(TEXTS.values())


def test_the_manifest_survives_a_crash_partway_through_a_bake(tmp_path):
    """Written per clip, not once at the end: a crash on clip 3 must not discard clips 1 and 2."""
    class ExplodingSynth(CountingSynth):
        def __call__(self, text):
            if text == "Castle kingside?":
                raise RuntimeError("boom")
            return super().__call__(text)

    with pytest.raises(RuntimeError):
        bake(tmp_path, ExplodingSynth())

    manifest = json.loads((tmp_path / "bm_george" / "manifest.json").read_text())
    assert "Castle kingside?" not in manifest["texts"].values()

    survivor = CountingSynth()
    bake(tmp_path, survivor)
    assert survivor.calls == ["Castle kingside?"]  # only the clip that never landed


def test_bake_raises_if_the_synthesizer_returns_no_audio(tmp_path):
    with pytest.raises(RuntimeError, match="no audio"):
        bake_clips(tmp_path, "bm_george", "b", lambda text: np.zeros(0, dtype=np.float32), texts=TEXTS)


### load_clips: the read-only path the game uses

def test_load_clips_returns_nothing_for_a_cache_that_was_never_baked(tmp_path):
    assert load(tmp_path / "nope") == {}


def test_load_clips_does_not_create_the_cache_directory(tmp_path):
    """A read must not conjure up the thing it is reading -- otherwise a game started on a cold
    cache leaves an empty .cache/speech/<voice>/ behind to confuse the next person."""
    cache = tmp_path / "nope"
    load(cache)

    assert not cache.exists()


def test_load_clips_returns_everything_after_a_bake(tmp_path, synth):
    baked = bake(tmp_path, synth)
    loaded = load(tmp_path)

    assert set(loaded) == set(baked)
    assert all(np.array_equal(loaded[key], baked[key]) for key in baked)


def test_load_clips_ignores_a_cache_baked_for_a_different_voice(tmp_path, synth):
    bake(tmp_path, synth, voice="bm_george")
    assert load(tmp_path, voice="am_michael") == {}


def test_load_clips_ignores_a_stale_manifest(tmp_path, synth):
    """Stale means a different kokoro version, speed or cache_version. Handing back audio baked
    under the old one would splice two voices together at a join -- a silent prosody mismatch,
    which is far worse to debug than a miss."""
    bake(tmp_path, synth)
    manifest_path = tmp_path / "bm_george" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["kokoro_versions"] = "kokoro=0.0.1,misaki=0.0.1"
    manifest_path.write_text(json.dumps(manifest))

    assert load(tmp_path) == {}


def test_load_clips_skips_only_the_clip_that_is_missing(tmp_path, synth):
    bake(tmp_path, synth)
    (tmp_path / "bm_george" / "dest" / "e4.npy").unlink()

    loaded = load(tmp_path)
    assert set(loaded) == set(TEXTS) - {"dest/e4"}
