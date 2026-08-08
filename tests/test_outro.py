"""The game-end sequence, exercised offline: no robot, no Claude, no Kokoro, no HF datasets."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from chess_commentator import outro
from chess_commentator.outro import dance_until, finale, load_move


class FakeMini:
    """Records what was played, in order."""

    def __init__(self, fail_on=None):
        self.moves = []
        self.fail_on = fail_on

    def play_move(self, move, initial_goto_duration=0.0, sound=False):
        if self.fail_on is not None and move.name == self.fail_on:
            raise RuntimeError(f"{move.name} exploded")
        self.moves.append(
            {"name": move.name, "goto": initial_goto_duration, "sound": sound}
        )


class FakeSpeaker:
    def __init__(self, comment="What a shambles. Never again."):
        self.calls = []
        self.comment = comment

    def pregenerate_outro(self, game):
        self.calls.append("pregenerate_outro")
        future = Future()
        future.set_result(None)
        return future

    def exclaim_win(self, game):
        self.calls.append("exclaim_win")

    def speak_outro(self, future):
        self.calls.append("speak_outro")
        return self.comment


def fake_game(checkmate=True):
    return SimpleNamespace(board=SimpleNamespace(is_checkmate=lambda: checkmate))


def fake_move(name):
    return SimpleNamespace(name=name, duration=1.84)


@pytest.fixture
def library(monkeypatch):
    """Every move name resolves, and no HuggingFace cache is ever touched."""
    constructed = []

    class FakeRecordedMoves:
        def __init__(self, dataset):
            constructed.append(dataset)

        def get(self, name):
            return fake_move(name)

    monkeypatch.setattr(outro, "RecordedMoves", FakeRecordedMoves)
    monkeypatch.setattr(outro, "_libraries", {})
    return constructed


### dance_until

def test_dance_repeats_the_minimum_even_if_the_roast_is_already_ready():
    """groovy_sway_and_roll is 1.84s. One repeat is a twitch, not a dance."""
    mini = FakeMini()

    repeats = dance_until(mini, fake_move("dance"), lambda: True, min_repeats=4, max_repeats=8)

    assert repeats == 4
    assert len(mini.moves) == 4


def test_dance_stops_at_max_repeats_if_the_roast_never_arrives():
    mini = FakeMini()

    repeats = dance_until(mini, fake_move("dance"), lambda: False, min_repeats=4, max_repeats=8)

    assert repeats == 8


def test_dance_stops_once_the_roast_is_ready():
    """The dance is there to hide Claude and Kokoro latency, so it ends when they do."""
    mini = FakeMini()
    ready = iter([False, False, False, False, True])

    repeats = dance_until(mini, fake_move("dance"), lambda: next(ready), min_repeats=2, max_repeats=8)

    assert repeats == 6


def test_only_the_first_repeat_eases_out_of_the_capture_pose():
    """The head has been pinned rigid all game, so the first repeat gotos into the dance.
    Every later one already starts where the previous ended -- a goto there would stutter."""
    mini = FakeMini()

    dance_until(mini, fake_move("dance"), lambda: False, min_repeats=1, max_repeats=3)

    assert [move["goto"] for move in mini.moves] == [outro.GOTO_DURATION_S, 0.0, 0.0]


### load_move

def test_load_move_parses_each_library_once(library):
    load_move("some/dataset", "a")
    load_move("some/dataset", "b")
    load_move("other/dataset", "c")

    assert library == ["some/dataset", "other/dataset"]


### finale

def test_finale_dances_then_roasts_then_sulks(library):
    mini, speaker = FakeMini(), FakeSpeaker()

    finale(mini, speaker, fake_game(checkmate=True))

    assert speaker.calls == ["pregenerate_outro", "exclaim_win", "speak_outro"]
    # The dance is submitted after the roast so the roast has runway, but it is *played*
    # before it: dancing in silence is the whole point.
    assert [move["name"] for move in mini.moves] == ["groovy_sway_and_roll"] * 4 + ["downcast1"]


def test_finale_does_not_dance_on_a_draw(library):
    """Nobody celebrates a stalemate. The roast and the sulk still happen -- a draw used to
    end the game in total silence."""
    mini, speaker = FakeMini(), FakeSpeaker()

    finale(mini, speaker, fake_game(checkmate=False))

    assert speaker.calls == ["pregenerate_outro", "speak_outro"]
    assert [move["name"] for move in mini.moves] == ["downcast1"]


def test_finale_sulk_plays_its_own_audio(library):
    mini, speaker = FakeMini(), FakeSpeaker()

    finale(mini, speaker, fake_game(checkmate=False))

    assert mini.moves[-1] == {"name": "downcast1", "goto": outro.GOTO_DURATION_S, "sound": True}


def test_finale_config_overrides_the_moves(library):
    mini, speaker = FakeMini(), FakeSpeaker()
    config = {
        "outro": {
            "dance": {"move": "yeah_nod", "min_repeats": 2, "max_repeats": 2},
            "sleep": {"move": "sleep1"},
        }
    }

    finale(mini, speaker, fake_game(checkmate=True), config)

    assert [move["name"] for move in mini.moves] == ["yeah_nod", "yeah_nod", "sleep1"]


def test_finale_can_be_disabled(library):
    mini, speaker = FakeMini(), FakeSpeaker()

    finale(mini, speaker, fake_game(checkmate=True), {"outro": {"enabled": False}})

    assert speaker.calls == []
    assert mini.moves == []


### Failure isolation: the game is already over, so no stage may cost another one

def test_a_missing_dance_dataset_still_leaves_the_roast_and_the_sulk(monkeypatch):
    def explode(dataset, name):
        if name == "groovy_sway_and_roll":
            raise ValueError("move not found")
        return fake_move(name)

    monkeypatch.setattr(outro, "load_move", explode)
    mini, speaker = FakeMini(), FakeSpeaker()

    finale(mini, speaker, fake_game(checkmate=True))

    assert speaker.calls == ["pregenerate_outro", "exclaim_win", "speak_outro"]
    assert [move["name"] for move in mini.moves] == ["downcast1"]


def test_a_failed_roast_still_leaves_the_sulk(library):
    mini, speaker = FakeMini(), FakeSpeaker()
    speaker.speak_outro = lambda future: (_ for _ in ()).throw(RuntimeError("claude is down"))

    finale(mini, speaker, fake_game(checkmate=True))

    assert [move["name"] for move in mini.moves] == ["groovy_sway_and_roll"] * 4 + ["downcast1"]


def test_a_dead_robot_does_not_propagate(library):
    mini, speaker = FakeMini(fail_on="downcast1"), FakeSpeaker()

    finale(mini, speaker, fake_game(checkmate=True))  # must not raise

    assert speaker.calls == ["pregenerate_outro", "exclaim_win", "speak_outro"]


def test_finale_still_roasts_if_generation_could_not_even_be_submitted(library):
    """pregenerate_outro raising leaves no future to wait on. The dance should fall back to
    its minimum rather than skipping, and the sulk should still land."""
    mini, speaker = FakeMini(), FakeSpeaker()
    speaker.pregenerate_outro = lambda game: (_ for _ in ()).throw(RuntimeError("no api key"))

    finale(mini, speaker, fake_game(checkmate=True))

    assert "speak_outro" not in speaker.calls
    assert [move["name"] for move in mini.moves] == ["groovy_sway_and_roll"] * 4 + ["downcast1"]
