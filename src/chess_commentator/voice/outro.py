"""The send-off: once the game is over, the robot dances, roasts the whole game, and sulks.

Sequenced by finale(), which main.py calls the moment board.is_game_over() goes true. The
whole thing is best-effort -- the game is already decided, so a missing dataset or a failed
Claude call costs you that one stage and nothing else.
"""
from reachy_mini.motion.recorded_move import RecordedMoves

DANCE_DATASET = "pollen-robotics/reachy-mini-dances-library"
DANCE_MOVE = "groovy_sway_and_roll"
SLEEP_DATASET = "pollen-robotics/reachy-mini-emotions-library"
SLEEP_MOVE = "downcast1"

DEFAULT_MIN_REPEATS = 4
DEFAULT_MAX_REPEATS = 8

# Long enough to ease the head out of the rigid capture hold it has been pinned in all game
# (make_head_rigid), short enough not to read as a pause.
GOTO_DURATION_S = 0.5

_libraries = {}


def load_move(dataset: str, name: str):
    """Fetch a named recorded move, caching the library it came from.

    RecordedMoves reads the local HuggingFace cache (~0.1s for either library), so this is
    loaded on demand rather than at startup -- but a library is only worth parsing once.
    """
    if dataset not in _libraries:
        _libraries[dataset] = RecordedMoves(dataset)
    return _libraries[dataset].get(name)


def dance_until(mini, move, is_ready, min_repeats=DEFAULT_MIN_REPEATS, max_repeats=DEFAULT_MAX_REPEATS):
    """Loop `move` until is_ready(), clamped to [min_repeats, max_repeats]. Returns repeats played.

    groovy_sway_and_roll is a 1.84s loop unit with no sidecar audio, not a whole dance, so a
    real dance means repeating it -- and the repeats chain seamlessly.

    is_ready() is the closing roast's future.done: dancing is what hides Claude and Kokoro
    latency, so the dance runs at least min_repeats (a dance, not a twitch) and then keeps
    going only while the roast is still coming. max_repeats caps the wait if it stalls.
    """
    repeats = 0
    while repeats < max_repeats:
        if repeats >= min_repeats and is_ready():
            break
        # Only the first repeat gotos: it eases out of the rigid capture pose, whereas every
        # later one already starts where the previous ended.
        mini.play_move(move, initial_goto_duration=GOTO_DURATION_S if repeats == 0 else 0.0)
        repeats += 1
    return repeats


def _stage(name, fn, *args, **kwargs):
    """Run one outro stage, swallowing whatever it throws.

    Same best-effort idiom as calibration.make_head_rigid. The guards are per-stage on
    purpose: a missing dance dataset must not cost you the roast, and a Claude error must
    not cost you the sulk.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the game is over; nothing here is worth crashing for
        print(f"finale: {name} skipped ({exc})")
        return None


def finale(mini, speaker, game, config=None):
    """See the game out: dance, roast, sulk. Never raises.

    Ordering is deliberate. The roast is submitted first so that everything after it is
    runway: exclaiming the win and the dance together buy ~9-16s, which is about what a
    60-word roast costs to write and synthesize.

    Everything runs on the main thread, strictly sequentially. play_move(sound=True) and
    robot.play() both drive mini.media, which tolerates exactly one thread.
    """
    outro_config = (config or {}).get("outro", {}) or {}
    if not outro_config.get("enabled", True):
        return

    dance_config = outro_config.get("dance", {}) or {}
    sleep_config = outro_config.get("sleep", {}) or {}
    checkmate = game.board.is_checkmate()

    future = _stage("roast generation", speaker.pregenerate_outro, game)

    if checkmate:
        _stage("win exclamation", speaker.exclaim_win, game)

        # Checkmate only: a stalemate is nobody's victory, and dancing about one would be odd.
        def dance():
            move = load_move(
                dance_config.get("dataset", DANCE_DATASET),
                dance_config.get("move", DANCE_MOVE),
            )
            repeats = dance_until(
                mini,
                move,
                # No future -> nothing to wait for, so dance the minimum and move on.
                future.done if future is not None else (lambda: True),
                min_repeats=dance_config.get("min_repeats", DEFAULT_MIN_REPEATS),
                max_repeats=dance_config.get("max_repeats", DEFAULT_MAX_REPEATS),
            )
            print(f"finale: danced {repeats} repeats")

        _stage("dance", dance)

    if future is not None:
        comment = _stage("roast", speaker.speak_outro, future)
        if comment:
            print(f"finale: {comment}")

    # Every ending, checkmate or draw: the "I don't want to see any more of this" beat.
    def sulk():
        move = load_move(
            sleep_config.get("dataset", SLEEP_DATASET),
            sleep_config.get("move", SLEEP_MOVE),
        )
        mini.play_move(move, initial_goto_duration=GOTO_DURATION_S, sound=True)

    _stage("sleep move", sulk)
