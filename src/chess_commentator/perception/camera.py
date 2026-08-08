"""
Taking photos of the board with Reachy's camera.
"""
from pathlib import Path
from datetime import datetime

import cv2

def capture_image(mini, output_dir: Path) -> Path:
    """
    Capture one frame from Reachy's camera and save it as a PNG.

    Each frame gets its own timestamped directory under `output_dir` (the setup dir), because
    everything derived from it -- the warped board, the 64 square cutouts -- is written next to
    it. Returns that directory, not the image path.
    """
    assert isinstance(output_dir, Path)
    assert output_dir.is_dir()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    image_dir = output_dir / f"board_{timestamp}"
    image_dir.mkdir()
    image_path = image_dir / "image.png"

    frame = mini.media.get_frame()

    # get_frame() already returns a BGR array (calibration.py saves it directly
    # and those images are correctly coloured; the previous cvtColor(RGB2BGR)
    # here produced blue/red-swapped images that propagated to every cutout).
    # cv2.imwrite expects BGR, so write the frame as-is.
    cv2.imwrite(str(image_path), frame)

    return image_dir
