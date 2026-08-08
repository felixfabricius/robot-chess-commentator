"""
Getting the robot ready to play: either run a fresh calibration, or reuse the last one.
"""
import yaml
import os
from datetime import datetime
from pathlib import Path
import json

from chess_assistant.calibration import calibrate, position_robot

def setup(mini, mode):
    """
    Prepare a setup directory and put the robot in its capture pose.

    Driven by `setup_folder.create_new` in config.yaml. If set, a new timestamped setup dir is
    created and calibrate() walks the user through clicking the board corners; otherwise the
    most recent setup dir is reused and the robot is simply moved back to the pose recorded in
    its metadata -- so a game can be resumed without re-calibrating, as long as neither the
    board nor the robot has moved.

    Returns (setup_dir, pixel_coordinates, robot_pose): the directory every image of this
    session is written to, the four clicked board corners in pixels, and the (height_mm,
    pitch_deg) pose that main.py replays before each capture.
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root_data_folder = Path(config.get("root_data_folder", "data"))

    assert mode in ["live", "generated", None]

    if config.get("setup_folder", {}).get("create_new"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        # Note: this is not actually used during data generation, so
        # only mode == "live" has an impact.
        if mode:
            setup_dir = root_data_folder / mode / timestamp
        else:
            setup_dir = root_data_folder / timestamp

        setup_dir.mkdir(parents=True)

        setup_data= (
            config.get("setup_folder", {}).get("measurements", {})
            | {key: config.get("setup_folder", {}).get(key) for key in ["chessboard", "documentation_image"]}
        )
        annotate_center = config.get("setup_folder", {}).get("annotate_center", True)
        calibration_data = calibrate(mini, setup_dir, annotate_center=annotate_center)
        if calibration_data is None:
            raise RuntimeError("Calibration was aborted before any points were collected.")

        with open(setup_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(setup_data | calibration_data, f, indent=2)

        # v2 metadata stores the clicked board corners nested under "actual_corners_px".
        pixel_coordinates = {
            field: calibration_data["actual_corners_px"][field]
            for field in ["a1", "a8", "h8", "h1"]
        }

        # The chosen capture pose, replayed before every gameplay image (see main.py).
        robot_pose = (calibration_data["height_mm"], calibration_data["pitch_deg"])

    else:
        assert root_data_folder.is_dir()
        if mode:
            assert (root_data_folder / mode).is_dir()
            folders = os.listdir(root_data_folder / mode)
        else:
            folders = os.listdir(root_data_folder)
        assert len(folders) > 0

        # Setup dirs are named by timestamp, so the last one alphabetically is the most recent.
        if mode:
            setup_dir = root_data_folder / mode / sorted(folders)[-1]
        else:
            setup_dir = root_data_folder / sorted(folders)[-1]         

        with open(setup_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Position robot correctly, and return the coordinates. The v2 metadata stores the
        # robot pose under "height_mm"/"pitch_deg" and the board corners under "actual_corners_px".
        position_robot(mini, metadata["height_mm"], metadata["pitch_deg"])

        pixel_coordinates = {
            field: metadata["actual_corners_px"][field]
            for field in ["a1", "a8", "h8", "h1"]
        }

        robot_pose = (metadata["height_mm"], metadata["pitch_deg"])

    return setup_dir, pixel_coordinates, robot_pose


