from types import SimpleNamespace

import numpy as np
import pytest

from chess_commentator import robot
from chess_commentator.robot import (
    OUTRO_PROMPT_END,
    PROMPT_END,
    Speaker,
    build_outro_prompt,
    build_prompt,
    format_uci_for_speech,
)
from chess_commentator.speech_clips import KOKORO_SAMPLE_RATE, bake_clips, clip_texts


def move_info(**overrides):
    base = {
        "move": "f3f7",
        "san": "Qxf7#",
        "moved_piece": "Queen",
        "turn": "white",
        "move_number": 4,
        "capture": True,
        "captured_piece": "Pawn",
        "castle": None,
        "en_passant": False,
        "promotion": None,
        "check": True,
        "checkmate": True,
    }
    return {**base, **overrides}


def history(**overrides):
    base = {
        "recent_moves": ["e2e4", "e7e5", "f1c4", "b8c6", "d1f3"],
        "capture_streak": 0,
        "quiet_streak": 5,
        "average_cp_loss": {"white": 12.5, "black": 88.0},
        "last_cp_losses": {"white": [0, 25], "black": [4, 172]},
    }
    return {**base, **overrides}


def test_prompt_contains_every_move_field():
    prompt = build_prompt(move_info(), cp_loss=0, history=history(), comment_history=[])

    assert "Move: Qxf7# (f3f7)" in prompt
    assert "Moved piece: Queen" in prompt
    assert "Turn: 4, white to move" in prompt
    assert "Centipawn loss: 0" in prompt
    assert "Capture: yes - took a Pawn" in prompt
    assert "Castle: no" in prompt
    assert "Check: yes" in prompt
    assert "Checkmate: yes" in prompt
    assert prompt.rstrip().endswith(PROMPT_END)


def test_prompt_contains_game_history():
    prompt = build_prompt(move_info(), cp_loss=240, history=history(), comment_history=[])

    assert "Recent moves: e2e4 e7e5 f1c4 b8c6 d1f3" in prompt
    assert "Consecutive captures: 0" in prompt
    assert "Consecutive quiet moves: 5" in prompt
    assert "Average centipawn loss - white: 12.5, black: 88.0" in prompt
    assert "Last centipawn losses - white: [0, 25], black: [4, 172]" in prompt


def test_prompt_renders_comment_history_in_turn_order():
    comments = [
        {"turn": 1, "side": "white", "move": "e2e4", "comment": "A bold start."},
        {"turn": 2, "side": "black", "move": "e7e5", "comment": "Mirror, mirror."},
    ]
    prompt = build_prompt(move_info(), cp_loss=0, history=history(), comment_history=comments)

    first = prompt.index('Turn 1; white; e2e4; "A bold start."')
    second = prompt.index('Turn 2; black; e7e5; "Mirror, mirror."')
    assert first < second


def test_castle_and_promotion_render_as_words_not_booleans():
    prompt = build_prompt(
        move_info(castle="queenside", promotion="Knight", capture=False, captured_piece=None),
        cp_loss=15,
        history=history(),
        comment_history=[],
    )
    assert "Castle: queenside" in prompt
    assert "Promotion: Knight" in prompt
    assert "Capture: no" in prompt


def test_en_passant_capture_is_spelled_out():
    prompt = build_prompt(
        move_info(capture=True, captured_piece="Pawn", en_passant=True),
        cp_loss=0,
        history=history(),
        comment_history=[],
    )
    assert "Capture: yes - took a Pawn (en passant)" in prompt
    assert "En passant: yes" in prompt


def test_empty_history_does_not_leak_none_into_the_prompt():
    """First move of the game: every history section is empty. The model should see
    an explanation, not a bare 'None' or an empty line."""
    empty = history(
        recent_moves=[],
        capture_streak=0,
        quiet_streak=0,
        average_cp_loss={"white": 0.0, "black": 0.0},
        last_cp_losses={"white": [], "black": []},
    )
    prompt = build_prompt(
        move_info(move_number=1, capture=False, captured_piece=None, check=False, checkmate=False),
        cp_loss=0,
        history=empty,
        comment_history=[],
    )

    assert "None" not in prompt
    assert "this is the first move" in prompt
    assert "this is the first comment of the game" in prompt


@pytest.mark.parametrize("uci, spoken", [("e2e4", "E2 to E4"), ("g1f3", "G1 to F3")])
def test_format_uci_for_speech(uci, spoken):
    assert format_uci_for_speech(uci) == spoken


### The closing roast prompt

def summary(**overrides):
    """A ChessGame.final_snapshot() for a Scholar's mate."""
    base = {
        "result": "1-0",
        "termination": "CHECKMATE",
        "winner": "white",
        "total_moves": 4,
        "total_plies": 7,
        "captures": 2,
        "average_cp_loss": {"white": 12.5, "black": 88.0},
        "worst_blunder": {
            "move_number": 3, "turn": "black", "san": "Ke7", "uci": "e8e7", "cp_loss": 412,
        },
        "moves": [
            {"move_number": 1, "san": "e4", "uci": "e2e4", "turn": "white", "cp_loss": 0},
            {"move_number": 1, "san": "e5", "uci": "e7e5", "turn": "black", "cp_loss": 12},
            {"move_number": 3, "san": "Ke7", "uci": "e8e7", "turn": "black", "cp_loss": 412},
        ],
        }
    return {**base, **overrides}


def test_outro_prompt_contains_the_result_and_the_stats():
    prompt = build_outro_prompt(summary(), comment_history=[])

    assert "Result: 1-0 (CHECKMATE)" in prompt
    assert "Winner: white" in prompt
    assert "Total moves: 4 (7 plies)" in prompt
    assert "Captures: 2" in prompt
    assert "Average centipawn loss - white: 12.5, black: 88.0" in prompt
    assert "Worst blunder: turn 3, black, Ke7, 412 centipawns lost" in prompt
    assert prompt.rstrip().endswith(OUTRO_PROMPT_END.format(max_words=60))


def test_outro_prompt_lists_every_move_in_order():
    prompt = build_outro_prompt(summary(), comment_history=[])

    assert "1; white; e4; 0" in prompt
    assert "1; black; e5; 12" in prompt
    assert prompt.index("1; white; e4; 0") < prompt.index("3; black; Ke7; 412")


def test_outro_prompt_renders_comment_history_for_callbacks():
    comments = [
        {"turn": 1, "side": "white", "move": "e2e4", "comment": "A bold start."},
        {"turn": 2, "side": "black", "move": "e7e5", "comment": "Mirror, mirror."},
    ]
    prompt = build_outro_prompt(summary(), comment_history=comments)

    first = prompt.index('Turn 1; white; e2e4; "A bold start."')
    second = prompt.index('Turn 2; black; e7e5; "Mirror, mirror."')
    assert first < second


def test_outro_prompt_says_a_draw_has_no_winner():
    """winner=None is a draw, not a missing value -- it must not render as 'None'."""
    prompt = build_outro_prompt(
        summary(result="1/2-1/2", termination="STALEMATE", winner=None), comment_history=[]
    )

    assert "Winner: nobody - it was a draw" in prompt
    assert "None" not in prompt


def test_outro_prompt_on_an_empty_game_does_not_leak_none():
    """Nothing was ever played (an immediate draw): every section is empty and the model
    should see prose, not a bare 'None' or a blank heading."""
    prompt = build_outro_prompt(
        summary(
            result="1/2-1/2", termination="INSUFFICIENT_MATERIAL", winner=None,
            total_moves=1, total_plies=0, captures=0,
            average_cp_loss={"white": 0.0, "black": 0.0},
            worst_blunder=None, moves=[],
        ),
        comment_history=[],
    )

    assert "None" not in prompt
    assert "Worst blunder: (none - no moves were played)" in prompt
    assert "(none - you did not comment on this game)" in prompt


### Pregeneration plumbing, exercised offline (no Claude, no Kokoro, no robot)

class FakeClient:
    """Stands in for anthropic.Anthropic(). Records the calls it was asked to complete."""

    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, system, messages):
        self.calls.append({
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "prompt": messages[0]["content"],
        })
        text = f"comment #{len(self.calls)}"
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

    @property
    def prompts(self):
        return [call["prompt"] for call in self.calls]


class FakeGame:
    """Stands in for ChessGame. cp_loss_for() is the only thing the worker calls on it."""

    def __init__(self):
        self.rated = []

    def cp_loss_for(self, move_uci):
        self.rated.append(move_uci)
        return 120, -40

    def history_snapshot(self, recent_moves=6, recent_cp_losses=5):
        return {
            "recent_moves": ["e2e4"],
            "capture_streak": 0,
            "quiet_streak": 1,
            "average_cp_loss": {"white": 0.0, "black": 0.0},
            "last_cp_losses": {"white": [], "black": []},
        }

    def final_snapshot(self):
        return summary()


def _tone(seconds=0.1, hz=180.0):
    """Stand-in for a Kokoro waveform. Must not be silence: trim() correctly reduces silence
    to length 0, which would leave every baked clip empty."""
    t = np.arange(int(seconds * KOKORO_SAMPLE_RATE), dtype=np.float32) / KOKORO_SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _fake_synthesize(self, text):
    self.synthesized = getattr(self, "synthesized", [])
    self.synthesized.append(text)
    return _tone()


def candidate(uci, **overrides):
    info = {
        "move": uci, "san": "Nf3", "moved_piece": "Knight", "turn": "white",
        "move_number": 3, "capture": False, "captured_piece": None, "castle": None,
        "en_passant": False, "promotion": None, "check": False, "checkmate": False,
    }
    info.update(overrides)
    return {"move": uci, "loss": 0.1, "move_info": info}


def _fake_externals(monkeypatch, tmp_path, played):
    monkeypatch.setattr(robot, "KPipeline", lambda **kwargs: None)
    monkeypatch.setattr(robot.anthropic, "Anthropic", FakeClient)
    # comment_on_move plays through mini.media, which we don't have.
    monkeypatch.setattr(robot, "play", lambda mini, audio, **kwargs: played.append(audio))
    monkeypatch.setattr(robot, "CLIP_CACHE_DIR", tmp_path)  # never touch the real .cache
    monkeypatch.setattr(Speaker, "_synthesize", _fake_synthesize)


@pytest.fixture
def speaker(monkeypatch, tmp_path):
    """The real Speaker with only its externals swapped out, over a warm cache.

    Deliberately not a hand-written subclass mirroring __init__: that mirror silently drifts
    every time __init__ gains an attribute. Running the real one also exercises load_clips and
    the manifest round-trip for free on every test below.

    The bake is explicit here because Speaker no longer does one -- and it must stay explicit.
    Without it every suggest_move test would quietly fall through to the live-synthesis
    fallback and still pass, testing nothing. This mirrors reality: pregenerate_speech runs,
    then the game runs.
    """
    played = []
    _fake_externals(monkeypatch, tmp_path, played)
    bake_clips(tmp_path, "bm_george", "b", lambda text: _tone())

    spk = Speaker(mini=None, config={"speaker": {"voice": "bm_george"}})
    spk.played = played
    spk.synthesized = []  # forget the bake; the tests below care about the hot path
    yield spk
    spk.shutdown()


@pytest.fixture
def cold_speaker(monkeypatch, tmp_path):
    """A Speaker with no cache at all -- nothing has ever been baked."""
    played = []
    _fake_externals(monkeypatch, tmp_path, played)

    spk = Speaker(mini=None, config={"speaker": {"voice": "bm_george"}})
    spk.played = played
    yield spk
    spk.shutdown()


def test_pregenerate_is_cached_per_candidate(speaker):
    """A candidate re-suggested within the same turn must not be regenerated."""
    game = FakeGame()
    speaker.pregenerate_comment(candidate("g1f3"), turn=1, game=game)
    speaker.pregenerate_comment(candidate("g1f3"), turn=1, game=game)

    speaker.pregenerated_comments["g1f3"].result()
    assert len(speaker.client.prompts) == 1
    assert game.rated == ["g1f3"]


def test_pregenerate_cache_resets_on_a_new_turn(speaker):
    """self.turn must actually advance. It never did, so `turn > self.turn` was always
    true and the cache was wiped on every single candidate -- defeating itself."""
    game = FakeGame()
    speaker.pregenerate_comment(candidate("g1f3"), turn=1, game=game)
    speaker.pregenerate_comment(candidate("b1c3"), turn=1, game=game)
    assert speaker.turn == 1
    assert set(speaker.pregenerated_comments) == {"g1f3", "b1c3"}

    speaker.pregenerate_comment(candidate("e2e4"), turn=2, game=game)
    assert speaker.turn == 2
    assert set(speaker.pregenerated_comments) == {"e2e4"}  # last turn's candidates dropped


def test_comment_on_move_returns_the_rating_the_worker_computed(speaker):
    """main.py hands these straight to apply_move, so the accepted move is never
    re-analysed by Stockfish on the critical path."""
    game = FakeGame()
    move = candidate("g1f3")
    speaker.pregenerate_comment(move, turn=1, game=game)

    cp_loss, new_score = speaker.comment_on_move("g1f3", move["move_info"], game)

    assert (cp_loss, new_score) == (120, -40)
    assert game.rated == ["g1f3"]  # rated once, by the worker -- not again here


def test_comment_on_move_records_comment_history(speaker):
    game = FakeGame()
    move = candidate("g1f3")
    speaker.pregenerate_comment(move, turn=1, game=game)
    speaker.comment_on_move("g1f3", move["move_info"], game)

    assert speaker.comment_history == [
        {"turn": 3, "side": "white", "move": "g1f3", "comment": "comment #1"}
    ]

    # ...and that history is fed to the next comment, so callbacks are possible.
    nxt = candidate("b8c6", turn="black", move_number=3)
    speaker.pregenerate_comment(nxt, turn=2, game=game)
    speaker.comment_on_move("b8c6", nxt["move_info"], game)

    assert 'Turn 3; white; g1f3; "comment #1"' in speaker.client.prompts[1]


### Move suggestions are spliced from pregenerated clips, never synthesized

def test_suggest_move_never_synthesizes(speaker):
    """The whole point of the clip library. Kokoro runs at ~0.6x realtime, so synthesizing here
    put ~2.2s between accepting a move and hearing it -- on the main thread, holding the lock
    the comment worker needs."""
    speaker.suggest_move("e2e4", candidate("e2e4")["move_info"])

    assert speaker.synthesized == []
    assert len(speaker.played) == 1


def test_suggest_move_splices_origin_destination_and_the_gap(speaker):
    speaker.suggest_move("e2e4", candidate("e2e4")["move_info"])

    origin, destination = speaker.clips["origin/e2"], speaker.clips["dest/e4"]
    gap = int(speaker.splice_gap_ms * KOKORO_SAMPLE_RATE / 1000)
    assert len(speaker.played[0]) == len(origin) + len(destination) + gap


def test_suggest_move_speaks_castling_as_a_single_clip(speaker):
    speaker.suggest_move("e1g1", candidate("e1g1", castle="kingside")["move_info"])

    assert speaker.synthesized == []
    assert np.array_equal(speaker.played[0], speaker.clips["castle/kingside"])


def test_suggest_move_falls_back_to_live_synthesis_if_a_clip_is_missing(speaker):
    """Shouldn't happen (test_speech_clips pins the inventory), but if it does: be slow, like
    before, rather than silent."""
    speaker.clips.pop("dest/e4")

    speaker.suggest_move("e2e4", candidate("e2e4")["move_info"])

    assert speaker.synthesized == ["E2 to E4?"]
    assert len(speaker.played) == 1


### A cold cache is a warning, not an 8-minute stall

def test_speaker_never_bakes_on_a_cold_cache(monkeypatch, tmp_path, capsys):
    """The whole point of pregenerate_speech: constructing a Speaker must not silently start
    synthesizing 150 clips, which used to stall main.py for 5-8 minutes right after
    calibration and before the first move.

    Built here rather than via the cold_speaker fixture because the warning is printed during
    __init__ -- from a fixture that lands in setup's capture, where the test body cannot see it.
    """
    _fake_externals(monkeypatch, tmp_path, [])
    spk = Speaker(mini=None, config={"speaker": {"voice": "bm_george"}})
    try:
        assert spk.clips == {}
        assert getattr(spk, "synthesized", []) == []

        out = capsys.readouterr().out
        assert f"{len(clip_texts())}/{len(clip_texts())} clips missing" in out
        assert "uv run python -m chess_commentator.pregenerate_speech --voice bm_george" in out
    finally:
        spk.shutdown()


def test_a_warm_speaker_says_nothing_about_missing_clips(monkeypatch, tmp_path, capsys):
    _fake_externals(monkeypatch, tmp_path, [])
    bake_clips(tmp_path, "bm_george", "b", lambda text: _tone())
    capsys.readouterr()  # drop the bake's own progress output

    spk = Speaker(mini=None, config={"speaker": {"voice": "bm_george"}})
    try:
        assert capsys.readouterr().out == ""
    finally:
        spk.shutdown()


def test_a_cold_speaker_still_speaks(cold_speaker):
    """A missing cache must never block a game -- it just costs what it always used to."""
    cold_speaker.suggest_move("e2e4", candidate("e2e4")["move_info"])

    assert cold_speaker.synthesized == ["E2 to E4?"]
    assert len(cold_speaker.played) == 1


def test_suggest_move_does_not_reprint_the_warning_on_every_move(cold_speaker, capsys):
    """The startup warning is complete (self.clips never shrinks), so repeating it per move
    would be pure noise across a whole game."""
    for _ in range(3):
        cold_speaker.suggest_move("e2e4", candidate("e2e4")["move_info"])

    assert capsys.readouterr().out == ""


def test_speaker_derives_the_language_from_the_voice(monkeypatch, tmp_path):
    """Hardcoding lang_code="b" while leaving voice configurable gives an American voice read
    by a British G2P."""
    lang_codes = []
    monkeypatch.setattr(robot, "KPipeline", lambda **kwargs: lang_codes.append(kwargs["lang_code"]))
    monkeypatch.setattr(robot.anthropic, "Anthropic", FakeClient)
    monkeypatch.setattr(robot, "play", lambda *args, **kwargs: None)
    monkeypatch.setattr(robot, "CLIP_CACHE_DIR", tmp_path)
    monkeypatch.setattr(Speaker, "_synthesize", _fake_synthesize)

    spk = Speaker(mini=None, config={"speaker": {"voice": "am_michael"}})
    spk.shutdown()

    assert lang_codes == ["a"]


### The closing roast, end to end (still no Claude, no Kokoro, no robot)

def test_outro_is_generated_with_its_own_token_cap(speaker):
    """The roast is three or four sentences. Reusing max_tokens (64, a one-liner cap) would
    guillotine it mid-sentence -- and silently, since a truncated reply still parses."""
    speaker.speak_outro(speaker.pregenerate_outro(FakeGame()))

    call = speaker.client.calls[-1]
    assert call["max_tokens"] == 200
    assert call["system"] == robot.OUTRO_SYSTEM_PROMPT
    assert "Result: 1-0 (CHECKMATE)" in call["prompt"]


def test_outro_token_cap_is_configurable(monkeypatch, tmp_path):
    monkeypatch.setattr(robot, "KPipeline", lambda **kwargs: None)
    monkeypatch.setattr(robot.anthropic, "Anthropic", FakeClient)
    monkeypatch.setattr(robot, "play", lambda *args, **kwargs: None)
    monkeypatch.setattr(robot, "CLIP_CACHE_DIR", tmp_path)
    monkeypatch.setattr(Speaker, "_synthesize", _fake_synthesize)

    spk = Speaker(mini=None, config={"speaker": {"outro": {"max_tokens": 500, "max_words": 30}}})
    spk.speak_outro(spk.pregenerate_outro(FakeGame()))
    spk.shutdown()

    call = spk.client.calls[-1]
    assert call["max_tokens"] == 500
    assert "at most 30 words" in call["prompt"]


def test_outro_sees_the_comment_history_it_built_during_the_game(speaker):
    game = FakeGame()
    move = candidate("g1f3")
    speaker.pregenerate_comment(move, turn=1, game=game)
    speaker.comment_on_move("g1f3", move["move_info"], game)

    speaker.speak_outro(speaker.pregenerate_outro(game))

    assert 'Turn 3; white; g1f3; "comment #1"' in speaker.client.calls[-1]["prompt"]


def test_pregenerate_outro_drops_the_last_turns_candidates(speaker):
    """Two workers, and the rejected candidates of the final turn may still be queued. Nobody
    will ever consume those comments, so the roast must not wait behind them."""
    game = FakeGame()
    speaker.pregenerate_comment(candidate("g1f3"), turn=1, game=game)
    speaker.pregenerate_comment(candidate("b1c3"), turn=1, game=game)

    speaker.pregenerate_outro(game)

    assert speaker.pregenerated_comments == {}


def test_speak_outro_plays_the_roast(speaker):
    comment = speaker.speak_outro(speaker.pregenerate_outro(FakeGame()))

    assert comment == "comment #1"
    assert len(speaker.played) == 1


def test_comment_on_move_generates_inline_if_never_pregenerated(speaker):
    """Fallback path: a move that was never suggested (so never pregenerated) must still
    get a comment rather than blowing up on a missing cache entry."""
    game = FakeGame()
    move = candidate("g1f3")

    cp_loss, _ = speaker.comment_on_move("g1f3", move["move_info"], game)

    assert cp_loss == 120
    assert len(speaker.client.prompts) == 1
    assert speaker.comment_history[0]["comment"] == "comment #1"
