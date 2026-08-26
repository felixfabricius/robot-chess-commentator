"""
Reading a chess board off the camera image, square by square.

`BoardEstimator` runs over the per-square cutouts produced by image_processing.cutout() and fills
a `BoardEstimate` (64 `SquareEstimate`s, each holding a score for all 13 labels: the 12 pieces
plus "empty"). Three square-level backends:
- "CNN" (production): the trained SquareClassifierMultiHead. One forward pass per square; the
  three heads are recombined into 13-way log-probabilities.
- "LLM" with llm_method="square_label" (iii): one Claude call per square returning a hard label
  (a one-hot reading).
- "LLM" with llm_method="square_logits" (iv): one Claude call per square returning a score for all
  13 labels, converted to log-probabilities.

None of them ever picks a move. They emit logit-like scores that game.estimate_move() scores every
legal move against, so a square the model is unsure about costs a candidate move some likelihood
rather than corrupting the position outright.

The coarser whole-image strategies -- and every Claude prompt, schema and transport detail the two
LLM backends here rely on -- live in `vlm/`. This module owns only the square-level loop.
"""
import json
from pathlib import Path

import anthropic
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from safetensors.torch import load_file
from torchvision import tv_tensors

from chess_commentator.board import SQUARES, BoardEstimate, SquareEstimate
from chess_commentator.labels import TARGET_MAP, TOP_LEFT_OHE_MAP, reconstruct_13way_logprobs
from chess_commentator.cnn.data import EVAL_TRANSFORM
from chess_commentator.cnn.model import SquareClassifierMultiHead
from chess_commentator.vlm.client import call_claude
from chess_commentator.vlm.prompts import (
    PROMPTS,
    SQUARE_LABEL_SCHEMA,
    SQUARE_LOGITS_SCHEMA,
    SQUARE_MAX_TOKENS,
)
from chess_commentator.vlm.strategies import parsed_to_square_estimate


class BoardEstimator:
    def __init__(self, model_type: str = "CNN", config: DictConfig | None = None, calibration_metadata_path: Path | None = None, model_weights_path = None, device = None, model = None, prior_correction: bool = True,
                 model_version: str | None = None, prompt_version: int | None = None, llm_method: str = "square_label", reasoning: str = "none"):
        """
        Build the estimator and hold the board estimate it keeps overwriting, one board
        position per call to estimate_board().

        Args:
            model_type: "CNN" for the trained classifier, "LLM" for a per-square Claude backend.
            config: the loaded config.yaml. Only read by the LLM path, and only as a fallback for
                model_version / prompt_version; the CNN path takes its weights explicitly.
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
            model_version, prompt_version: LLM only. Override the Claude model id and prompt
                version directly (so evaluation can sweep them); fall back to `config.vision` then
                to defaults when not given.
            llm_method: LLM only. "square_label" (iii; a hard one-hot label) or "square_logits"
                (iv; a score per label, stored as log-probabilities).
            reasoning: LLM only. "none" / "text" / "thinking" (see vlm.client.call_claude). Defaults to
                "none" because a square method makes 64 calls per board.
        """
        assert model_type in ["CNN", "LLM"]
        self.board_estimate = BoardEstimate()
        if model_type == "LLM":
            assert llm_method in ("square_label", "square_logits")
            cfg_vision = config.vision if config is not None else {}
            self.model_version = model_version or cfg_vision.get("model_version", "claude-sonnet-5")
            self.prompt_version = prompt_version if prompt_version is not None else cfg_vision.get("prompt_version", 1)
            self.llm_method = llm_method
            self.reasoning = reasoning
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

        Returns a SquareEstimate whose 13 label fields hold logit-like scores. square_label gives
        a hard one-hot; square_logits gives log-probabilities; the CNN gives log-probabilities.

        LLM calls record their token usage and wall-clock time in `self.last_usage` /
        `self.last_elapsed` so the evaluation harness can accumulate cost and timing per board.
        """
        if self.model_type == "LLM":
            image_path = image_path.parent / (image_path.stem + "_annotated" + image_path.suffix)
            schema = SQUARE_LABEL_SCHEMA if self.llm_method == "square_label" else SQUARE_LOGITS_SCHEMA
            parsed, usage, elapsed = call_claude(
                self.client, self.model_version, [image_path],
                PROMPTS[self.llm_method][self.prompt_version], schema,
                reasoning=self.reasoning, max_tokens=SQUARE_MAX_TOKENS[self.llm_method],
            )
            # Same interpretation the batch path uses, so the two agree square-for-square.
            square_estimate = parsed_to_square_estimate(self.llm_method, parsed, image_path=image_path)
            self.last_usage = usage
            self.last_elapsed = elapsed
            return square_estimate

        else:
            # TODO: the square name is recovered from the filename, so this breaks if the
            # cutout naming convention in perception/image_processing.py ever changes.
            square = image_path.stem
            square_dir = image_path.parent

            # Metadata: one-hot of which board corner is top-left in the image. Must match
            # training (cnn/data.py) - both use TOP_LEFT_OHE_MAP.
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

            # Multi-head model (model 3): recombine the three heads into 13-way
            # log-probabilities. softmax(log p) == p and the reconstructed probs sum to
            # 1, so log p is a drop-in for the old single-head logits under the softmax
            # game.py re-applies downstream.
            with torch.no_grad():
                logit_empty, logits_color, logits_type = self.model(image, metadata)
            logprobs = reconstruct_13way_logprobs(
                logit_empty.squeeze(0),
                logits_color.squeeze(0),
                logits_type.squeeze(0),
                log_prior=self.log_prior,
            )
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

        For the LLM backends the 64 per-square calls' token usage and wall-clock time are summed
        into `self.board_input_tokens` / `self.board_output_tokens` / `self.board_elapsed`, so the
        evaluation harness can read a board's cost and timing after one estimate_board() call.
        """
        is_llm = self.model_type == "LLM"
        self.board_input_tokens = 0
        self.board_output_tokens = 0
        self.board_elapsed = 0.0
        for square in SQUARES:
            image_path = squares_dir / square / f"{square}.png"
            square_estimate = self.estimate_square(image_path)
            setattr(self.board_estimate, square, square_estimate)
            if is_llm:
                self.board_input_tokens += self.last_usage.input_tokens
                self.board_output_tokens += self.last_usage.output_tokens
                self.board_elapsed += self.last_elapsed

        return self.board_estimate


if __name__ == "__main__":
    # Debug tool: run the trained classifier over a squares directory and print what it thinks
    # is standing on each square. Useful when the board estimate disagrees with reality and you
    # want to see whether the model is confidently wrong or merely unsure.
    #
    #   uv run python -m chess_commentator.perception.board_estimator \
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

