"""
The game loop: the robot watches a physical chess game and comments on it.

Run it with the robot plugged in:

    uv run python -m chess_commentator.main

One iteration = one move: wait for the players to signal that they have moved, photograph the
board, read the position, rank the legal moves against that reading, suggest the most likely
one out loud, and -- unless the players reject it -- play it and comment on it. Everything it
needs (calibration, engine, voice, input method) is configured in config.yaml.
"""
from pathlib import Path
from omegaconf import OmegaConf

from chess_commentator.session import setup
from chess_commentator.perception.camera import capture_image
from chess_commentator.perception.board_estimator import BoardEstimator
from chess_commentator.game import ChessGame
from chess_commentator.perception.image_processing import Processor
from chess_commentator.player_input import InputDetector
from chess_commentator.voice.speaker import Speaker
from chess_commentator.voice.outro import finale
from chess_commentator.perception.calibration import make_head_rigid, move_to_capture_pose
from chess_commentator.perception.calibration_monitor import launch_calibration_monitor

from reachy_mini import ReachyMini


def main(mini) -> None:
    """Set everything up from config.yaml, then run the game loop until the game ends."""
    config = OmegaConf.load("config.yaml")

    setup_dir, _, robot_pose = setup(mini, "live")
    calibration_metadata_path = setup_dir / "calibration_metadata.json"
    # Live drift check in its own process: overlays the calibrated corners on the undistorted
    # camera feed so you can watch whether the camera has moved away from the calibrated pose.
    if config.get("monitor_calibration"):
        launch_calibration_monitor(calibration_metadata_path)
    image_processor = Processor(calibration_metadata_path, "config.yaml")
    board_estimator = BoardEstimator("CNN", config, calibration_metadata_path, model_weights_path=Path(config.vision.model_weights_path), device="cpu")
    input_detector = (
        InputDetector(input_source="robot", mini=mini, calibration_metadata_path=calibration_metadata_path) 
        if config.get("input", {"source": "robot"}).get("source", "robot") == "robot" 
        else InputDetector(input_source="keyboard", target_key=None)
    )
    speaker = Speaker(mini, config)
    engine_config = config.get("engine", {})
    game = ChessGame(
        stockfish_path=engine_config.get("stockfish_path"),
        depth=engine_config.get("depth", 16),
    )

    # Keep the head in a rigid, non-drifting hold for the whole game so every board image is
    # captured from the calibrated pose (see make_head_rigid). Re-asserted here in case setup
    # loaded an existing calibration without running calibrate().
    make_head_rigid(mini)

    print("Start Game")
    game_over = False
    i = 0
    while not game_over:
        i += 1
        print("Ready for move")
        move_made = input_detector.detect_input(input_type="move_made")
        print(f"move made: {move_made}")
        # Snap the head back to the exact calibrated pose so every board image is taken from
        # the same position (the head may have drifted while operating the antennas).
        move_to_capture_pose(mini, *robot_pose)
        image_dir = capture_image(mini, setup_dir)
        print("image captured")

        # Rectify the board out of the photo, cut it into 64 squares, classify each of them,
        # then ask the game which legal move best explains what was seen.
        warped_image_path = image_processor.warp(image_dir / "image.png")
        squares_dir = image_processor.cutout(warped_image_path)
        board_estimate = board_estimator.estimate_board(squares_dir)
        move_estimates = game.estimate_move(board_estimate)
        print("moves estimated")


        move_estimate_accepted = False
        accepted = None
        for candidate in move_estimates:
            # Submit BEFORE speaking, not after. Non-blocking (returns in ~1ms), so the
            # worker gets to rate the candidate, write its comment and synthesize the
            # waveform across both the suggestion playback below AND the review window --
            # roughly 2.5s of extra runway for free. Kokoro synthesis is the slow stage
            # (~0.6x realtime) and needs every bit of it.
            speaker.pregenerate_comment(candidate, i, game)
            # move_info, not just the UCI: "e1g1" alone cannot say whether this is castling or
            # a queen walking from e1 to g1, and the two are announced differently.
            speaker.suggest_move(candidate["move"], candidate["move_info"])
            move_estimate_rejected = input_detector.detect_input(input_type="move_estimate_rejected", alloted_time=config.get("review_time", 4))
            move_estimate_accepted = not move_estimate_rejected
            if move_estimate_accepted:
                accepted = candidate
                break

        assert move_estimate_accepted # Assert we didn't loop through all moves without accepting any.
        move = accepted["move"]
        move_info = accepted["move_info"]
        print(f"move estimate accepted: {move}")

        # Speak first: the comment carries the centipawn rating of the move, which the
        # worker already computed, so apply_move can reuse it instead of re-running Stockfish.
        move_cp_loss, new_score = speaker.comment_on_move(move, move_info, game)
        game.apply_move(move, move_info=move_info, cp_loss=move_cp_loss, new_score=new_score)

        game_over = game.board.is_game_over() # checkmate, stalemate, or any draw condition
        if game_over:
            # Dance, roast the game, sulk. Best-effort, and must run before shutdown()
            # below, which cancels the futures the roast is generated on.
            finale(mini, speaker, game, config)
            break # no next turn to switch to

        input_detector.switch_turn()

    speaker.shutdown()
    game.close()

if __name__ == "__main__":
    with ReachyMini(media_backend="default") as mini:
        main(mini)
