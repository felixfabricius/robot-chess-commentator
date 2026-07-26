"""
Reading a board position off the camera image: one classification per square.

`BoardEstimator` runs over the per-square cutouts produced by image_processing.cutout()
and fills a `BoardEstimate` (64 `SquareEstimate`s, each holding a score for all 13 labels:
the 12 pieces plus "empty").

Two backends:
- "CNN" (production): the trained SquareClassifierMultiHead. One forward pass per square;
  the three heads are recombined into 13-way log-probabilities.
- "LLM" (baseline): one Claude call per square, kept as the comparison the CNN was built to
  beat. It only ever returns a hard one-hot, so its "confidences" are 0/1.

Neither backend picks a move. They emit logit-like scores that game.estimate_move() scores
every legal move against; a square the model is unsure about therefore costs a candidate move
some likelihood rather than corrupting the position outright.

`infer_fen_from_image` is the older, cruder baseline still kept for comparison: one LLM call
for the whole board, returning a FEN board string directly.
"""
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, make_dataclass
import anthropic
import torch
from torchvision import tv_tensors
from safetensors.torch import load_file

import numpy as np


from omegaconf import OmegaConf, DictConfig

from chess_assistant.config import SQUARES
from chess_assistant.model.config import (
    TARGET_MAP,
    reconstruct_13way_logprobs,
    logprobs13_from_logits,
    TOP_LEFT_OHE_MAP,
)
from chess_assistant.model.data import EVAL_TRANSFORM
from chess_assistant.model.model import SquareClassifierMultiHead

load_dotenv()

PROMPTS = {
    0: (
        "You are looking at a physical chess board."
        "Return only the board position as a FEN board string, "
        "not the full FEN. Example format (if in starting position): "
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR. "
        "Do not include side to move, castling rights, move counters, or explanation."
        "CAREFULLY inspect each of the 64 squares individually to identify which piece - if any - "
        "is located there."
    ),
    1: (
        "You are classifying one square from a chessboard image.\n"
        "The target square is marked with a red corner markers. Other pieces and neighbouring squares may be visible because the crop includes padding.\n"
        "Classify only the chess piece whose BASE is on the highlighted target square. Ignore all other visible pieces.\n"
        "Return exactly one label from:\n"
        "empty, K, Q, R, B, N, P, k, q, r, b, n, p,\n"
        "where the letter corresponds to the piece in FEN notation, e.g. K stands for white king."
        "Return nothing but this label."
    )
}

@dataclass
class SquareEstimate:
    image_path: Path | None = None
    copied: bool = False
    copied_from: Path | None = None
    K: float = 0
    Q: float = 0
    R: float = 0
    B: float = 0
    N: float = 0
    P: float = 0
    k: float = 0
    q: float = 0
    r: float = 0
    b: float = 0
    n: float = 0
    p: float = 0
    empty: float = 0

BoardEstimate = make_dataclass(
    "BoardEstimate",
    [(square, SquareEstimate | None, None) for square in SQUARES]
)

def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def infer_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    if suffix not in types:
        raise ValueError(f"Unsupported image type: {suffix}")
    return types[suffix]

def infer_fen_from_image(image_path: Path, model: str = "claude-opus-4-8", prompt_version: int = 0) -> str:
    client = anthropic.Anthropic()

    prompt = PROMPTS[prompt_version]

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": infer_media_type(image_path),
                            "data": encode_image_base64(image_path),
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )

    return message.content[0].text

class BoardEstimator:
    def __init__(self, model_type: str = "CNN", config: DictConfig | None = None, calibration_metadata_path: Path | None = None, model_weights_path = None, device = None, model = None, prior_correction: bool = True):
        """
        Build the estimator and hold the board estimate it keeps overwriting, one board
        position per call to estimate_board().

        Args:
            model_type: "CNN" for the trained classifier, "LLM" for the Claude baseline.
            config: the loaded config.yaml. Only read by the LLM path (model + prompt
                version); the CNN path takes its weights explicitly.
            calibration_metadata_path: calibration_metadata.json of the setup the squares
                come from. CNN only, and required: it carries the one piece of metadata the
                model is fed alongside the image (which board corner is top-left).
            model_weights_path: safetensors checkpoint to load into a fresh
                SquareClassifierMultiHead. CNN only.
            device: "cpu" or "cuda" (default "cpu"). CNN only.
            model: an already-constructed model, used instead of model_weights_path. This is
                how evaluation and the tests hand in a model they have in memory, without a
                round trip through disk.
            prior_correction: subtract the model's training log-prior from the 13-way
                log-probabilities (Bayesian prior correction; see reconstruct_13way_logprobs).
                CNN only. Defaults to True, which is what live inference wants: the robot should
                always use the best available decision rule, and the buffer is all-zeros whenever
                the prior is unknown (legacy checkpoints), where subtracting it is a no-op. Eval
                passes this explicitly instead, so a run can be scored both ways.
        """
        assert model_type in ["CNN", "LLM"]
        self.board_estimate = BoardEstimate()
        if model_type == "LLM":
            assert config is not None
            self.model_version = config.vision.get("model_version", "claude-opus-4-8")
            self.prompt_version = config.vision.get("prompt_version", 1)
            self.client = anthropic.Anthropic()
        else:
            assert calibration_metadata_path is not None
            assert model_weights_path is not None or model is not None
            if model is None:
                model = SquareClassifierMultiHead()
                state_dict = load_file(model_weights_path, device="cpu")
                # strict=False tolerates exactly one thing: weights saved before `log_prior` (the
                # Bayesian prior-correction buffer) existed, which simply leaves it at all-zeros
                # = no correction. Everything else is still an error, so a genuinely mismatched
                # checkpoint cannot slip through silently.
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                assert not unexpected, f"Unexpected keys in {model_weights_path}: {unexpected}"
                assert set(missing) <= {"log_prior"}, f"Missing keys in {model_weights_path}: {missing}"
            assert device in ["cpu", "cuda", None, torch.device("cpu"), torch.device("cuda")]
            self.device = torch.device(device) if device is not None else torch.device("cpu")
            with open(calibration_metadata_path, "r", encoding="utf-8") as f:
                calibration_metadata = json.load(f)
            # The model's only metadata: which board corner is top-left in the image.
            self.top_left_corner = calibration_metadata["camera_natural_orientation"]["order"]["tl"]
            model.eval()
            self.model = model.to(self.device)
            # Resolved once here rather than per square. getattr keeps this working for any model
            # handed in that predates the buffer; None means "no correction".
            self.log_prior = getattr(model, "log_prior", None) if prior_correction else None
        self.model_type = model_type

    def estimate_square(self, image_path: Path) -> SquareEstimate:
        """
        Classify one square, given the path of its cutout (.../squares/e4/e4.png).

        The two backends read different files from that directory: the LLM gets the annotated
        PNG (the crop with red markers on the target square's corners, which the prompt refers
        to), the CNN gets the 4-channel masked .npy the crop was saved alongside.

        Returns a SquareEstimate whose 13 label fields hold logit-like scores. The LLM's are a
        hard one-hot; the CNN's are log-probabilities.
        """
        if self.model_type == "LLM":
            image_path = image_path.parent / (image_path.stem + "_annotated" + image_path.suffix)
            message = self.client.messages.create(
                model=self.model_version,
                max_tokens=128,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": infer_media_type(image_path),
                                    "data": encode_image_base64(image_path),
                                }
                            },
                            {"type": "text", "text": PROMPTS[self.prompt_version]}
                        ]
                    }
                ]
            )
            square_estimate = SquareEstimate(
                image_path=image_path,
                copied=False,
                copied_from=None
            )
            try:
                setattr(square_estimate, message.content[0].text, 1.)
            except Exception as e:
                print(e)

            return square_estimate

        else:
            # TODO: the square name is recovered from the filename, so this breaks if the
            # cutout naming convention in image_processing.py ever changes.
            square = image_path.stem
            square_dir = image_path.parent

            # Metadata: one-hot of which board corner is top-left in the image. Must match
            # training (model/data.py) - both use TOP_LEFT_OHE_MAP.
            metadata = torch.zeros(1, 4, dtype=torch.float32)
            metadata[0, TOP_LEFT_OHE_MAP[self.top_left_corner]] = 1
            metadata = metadata.to(self.device)
            assert metadata.shape == (1, 4)

            # The image: RGB plus the square's mask as a 4th channel, transformed exactly as in
            # training (EVAL_TRANSFORM, i.e. no augmentation).
            image = np.load(square_dir / f"{square}_masked.npy")
            rgb = image[..., :3]
            mask = tv_tensors.Mask(image[..., 3])
            rgb = EVAL_TRANSFORM(rgb)
            mask = EVAL_TRANSFORM(mask).unsqueeze(dim=0)
            image = torch.cat([rgb, mask]).unsqueeze(dim=0).to(self.device)
            assert image.shape == (1, 4, 144, 144)
            
            # Whatever the model's output shape, what gets stored is logit-like values (raw
            # logits or reconstructed log-probabilities), which game.py feeds into its own
            # CrossEntropyLoss.
            square_estimate = SquareEstimate(
                image_path=image_path,
                copied=False,
                copied_from=None
            )

            # Whatever the model, what game.py needs is 13-way log-probabilities. The multi-head
            # model 3 returns a 3-tuple recombined via reconstruct_13way_logprobs; the single-head
            # models 1/2 return a single 13-way logit tensor turned into log-probs by
            # logprobs13_from_logits. Both apply the same optional prior correction (self.log_prior)
            # and both yield a normalised log-prob vector, a drop-in for the softmax game.py
            # re-applies downstream.
            with torch.no_grad():
                out = self.model(image, metadata)
            if isinstance(out, tuple):
                logprobs = reconstruct_13way_logprobs(
                    out[0].squeeze(0),
                    out[1].squeeze(0),
                    out[2].squeeze(0),
                    log_prior=self.log_prior,
                )
            else:
                logprobs = logprobs13_from_logits(out.squeeze(0), log_prior=self.log_prior)
            for label, idx in TARGET_MAP.items():
                setattr(square_estimate, label, logprobs[idx].item())
            return square_estimate

    def estimate_board(self, squares_dir):
        """
        Classify all 64 squares of one board image and return the resulting BoardEstimate.

        `squares_dir` is the directory image_processing.cutout() wrote, i.e. one subdirectory
        per square. The estimate is stored on the estimator (self.board_estimate) as well as
        returned; it is overwritten on every call, so callers must not hold on to it across
        board positions.
        """
        for square in SQUARES:
            image_path = squares_dir / square / f"{square}.png"
            square_estimate = self.estimate_square(image_path)
            setattr(self.board_estimate, square, square_estimate)

        return self.board_estimate


if __name__ == "__main__":
    # Debug tool: run the trained classifier over a squares directory and print what it thinks
    # is standing on each square. Useful when the board estimate disagrees with reality and you
    # want to see whether the model is confidently wrong or merely unsure.
    #
    #   uv run python -m chess_assistant.vision \
    #       data/generated/2026-07-01_175334/board_2026-07-01_175602/squares --squares a1 e4
    import argparse
    import math

    parser = argparse.ArgumentParser(description="Print the model's prediction for each square.")
    parser.add_argument(
        "squares_dir",
        type=Path,
        help="A .../<board>/squares directory, as written by image_processing.cutout().",
    )
    parser.add_argument(
        "--squares",
        nargs="+",
        default=SQUARES,
        metavar="SQUARE",
        help="Squares to classify, e.g. --squares a1 e4. Defaults to all 64.",
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--calibration-metadata",
        type=Path,
        default=None,
        help="Defaults to calibration_metadata.json in the setup dir two levels up from squares_dir.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=3, help="How many labels to print per square.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    calibration_metadata_path = (
        args.calibration_metadata
        if args.calibration_metadata is not None
        else args.squares_dir.parent.parent / "calibration_metadata.json"
    )

    board_estimator = BoardEstimator(
        "CNN",
        config,
        calibration_metadata_path=calibration_metadata_path,
        model_weights_path=Path(config.vision.model_weights_path),
        device=args.device,
    )

    for square in args.squares:
        square_estimate = board_estimator.estimate_square(args.squares_dir / square / f"{square}.png")
        # The stored scores are log-probabilities; exp() turns them back into probabilities.
        ranked = sorted(
            ((getattr(square_estimate, label), label) for label in TARGET_MAP),
            reverse=True,
        )[: args.top_k]
        predictions = "  ".join(f"{label}: {math.exp(logprob):.3f}" for logprob, label in ranked)
        print(f"{square}  {predictions}")

