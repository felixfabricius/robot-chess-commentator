"""Label and metadata encodings shared by training (model/data.py, model/train.py,
model/evaluate.py) and inference (vision.py), plus the two functions that convert between the
13-way label space and the multi-head model's factored one. Anything both sides need to agree on
lives here, so the two encodings cannot drift apart.
"""

import torch
import torch.nn.functional as F

TARGET_MAP = {piece: label for label, piece in enumerate([
    "empty",
    "K", "Q", "R", "B", "N", "P",
    "k", "q", "r", "b", "n", "p"
])}

INVERSE_TARGET_MAP = {label: piece for piece, label in TARGET_MAP.items()}

# matches torch.nn.CrossEntropyLoss default ignore_index
IGNORE_INDEX = -100

COLOR_MAP = {"white": 0, "black": 1}
INVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}

TYPE_MAP = {"K": 0, "Q": 1, "R": 2, "B": 3, "N": 4, "P": 5}
INVERSE_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}

# One-hot index for which board corner is top-left in the camera image
# (calibration_metadata["camera_natural_orientation"]["order"]["tl"]). This is the model's
# only metadata now. Shared between training (model/data.py) and inference (vision.py) so the
# two encodings can never drift apart.
TOP_LEFT_OHE_MAP = {"a8": 0, "a1": 1, "h1": 2, "h8": 3}


def decompose_label(label: str) -> tuple[float, int, int]:
    """
    Decompose a 13-way label (as in TARGET_MAP) into:
      - is_piece: 1.0 if occupied, 0.0 if empty
      - color_target: 0 (white) / 1 (black), or IGNORE_INDEX if empty
      - type_target: 0..5 (K/Q/R/B/N/P), or IGNORE_INDEX if empty
    """
    if label == "empty":
        return 0.0, IGNORE_INDEX, IGNORE_INDEX
    color_target = COLOR_MAP["white"] if label.isupper() else COLOR_MAP["black"]
    type_target = TYPE_MAP[label.upper()]
    return 1.0, color_target, type_target


def recompose_label13(color_target: torch.Tensor, type_target: torch.Tensor) -> torch.Tensor:
    """Tensor-level inverse of decompose_label: rebuild the 13-way integer label (indexed per
    TARGET_MAP) from the per-head color/type targets.

      empty -> 0 ; white piece -> type + 1 ; black piece -> type + 1 + 6

    The non-empty rows are exactly those with color_target != IGNORE_INDEX (decompose_label sets
    both color and type to IGNORE_INDEX iff empty). Shared by evaluate.py (which needs the 13-way
    label to score the reconstructed multi-head distribution) and the single-head model 2 path
    (whose targets are the same decomposed 5-tuple), so the two can never disagree on the encoding.
    """
    nonempty = color_target != IGNORE_INDEX
    label13 = torch.zeros_like(type_target)
    label13[nonempty] = type_target[nonempty] + 1 + color_target[nonempty] * 6
    return label13


def logprobs13_from_logits(logits, log_prior=None):
    """Single-head analogue of reconstruct_13way_logprobs: turn a model's raw 13-way logits
    (shape (..., 13), indexed per TARGET_MAP) into normalised log-probabilities, with the same
    optional Bayesian prior correction.

    Models 1 & 2 have a single 13-way head, so there is nothing to recombine - a log_softmax is all
    the reconstruction there is. `log_prior` (shape (13,)) subtracts the training log-prior exactly
    as in reconstruct_13way_logprobs; for a single head that is just logit adjustment,
    log_softmax(log_softmax(logits) - log_prior) == log_softmax(logits - log_prior), and the result
    is re-normalised so the return value is a proper log-probability vector either way.

    The contract matches reconstruct_13way_logprobs exactly (a normalised (..., 13) log-prob
    vector), so the two are interchangeable everywhere downstream: argmax, nn.CrossEntropyLoss, and
    the softmax game.py re-applies all give the intended answer.
    """
    out = F.log_softmax(logits, dim=-1)
    if log_prior is not None:
        out = F.log_softmax(out - log_prior, dim=-1)
    return out


def log_prior_from_label_counts(counts: dict[str, int]) -> torch.Tensor:
    """Laplace-smoothed 13-way log-prior from a {label: count} mapping over TARGET_MAP labels.

    Add-one smoothing is not cosmetic here: a class absent from the training split (entirely
    possible once the data-efficiency ablation subsamples down to ~5k rows) would otherwise have
    count 0 and log-prior -inf, and subtracting -inf sends that class's corrected score to +inf.
    Leaving the entry at 0.0 instead - the other obvious "fix" - is worse still, since log-prior 0
    means P(class) == 1. One pseudo-count keeps every entry finite and small.
    """
    smoothed = torch.ones(13, dtype=torch.float32)  # one pseudo-count per class
    for label, count in counts.items():
        smoothed[TARGET_MAP[label]] += count
    return torch.log(smoothed / smoothed.sum())


def reconstruct_13way_logprobs(logit_empty, logits_color, logits_type, log_prior=None):
    """
    Combine the three heads into log-probabilities over the original 13-way TARGET_MAP
    labels, under a conditional-independence assumption between color and type given
    non-empty. Shape: (..., 13), indexed per TARGET_MAP / INVERSE_TARGET_MAP.

    `log_prior` (optional, shape (13,)) switches on Bayesian prior correction / "logit
    adjustment": a softmax model trained with cross-entropy approximates
    log P_train(class | x) = log P(x | class) + log pi_class + const, so subtracting the training
    log-prior leaves scores proportional to the evidence log P(x | class) alone - i.e. the
    decision stops inheriting whichever pieces happened to be frequent in training. The result is
    re-normalised, so the return value is a proper log-probability vector either way.

    Note the correction must cover ALL 13 classes including "empty" (~56% of squares, so by far
    the largest prior term); removing the prior from the 12 piece classes only would skew the
    estimate harder than not correcting at all.

    Everything stays in log space via logsigmoid / log_softmax, so this is numerically stable
    and needs no epsilon-clamping. The result is a normalised log-probability vector, and since
    softmax(log p) == p it is a drop-in replacement for the old single-head logits: feeding it to
    argmax, to nn.CrossEntropyLoss, or to the softmax that game.py re-applies downstream all give
    the intended answer.
    """
    # The empty head is trained with BCEWithLogitsLoss against `is_piece` (1 = piece), so
    # sigmoid(logit_empty) == P(piece). `logit_empty` is therefore a piece-logit despite its
    # name, and the empty/non-empty branches must be assigned accordingly.
    log_p_nonempty = F.logsigmoid(logit_empty)         # log P(non-empty) = log sigmoid(logit_empty) = log P(piece)
    log_p_empty = F.logsigmoid(-logit_empty)           # log P(empty)     = log(1 - P(piece))
    log_p_color = F.log_softmax(logits_color, dim=-1)  # (..., 2)
    log_p_type = F.log_softmax(logits_type, dim=-1)    # (..., 6)

    out = torch.empty(logit_empty.shape + (13,), dtype=logit_empty.dtype, device=logit_empty.device)
    out[..., TARGET_MAP["empty"]] = log_p_empty
    for label, idx in TARGET_MAP.items():
        if label == "empty":
            continue
        color_idx = COLOR_MAP["white"] if label.isupper() else COLOR_MAP["black"]
        type_idx = TYPE_MAP[label.upper()]
        out[..., idx] = log_p_nonempty + log_p_color[..., color_idx] + log_p_type[..., type_idx]

    if log_prior is not None:
        # Applied to the whole (..., 13) vector at once - including "empty" - then re-normalised
        # so the contract above ("a normalised log-probability vector") still holds. log_softmax
        # only shifts by a per-row constant, so argmax / CrossEntropyLoss / game.py's softmax are
        # unaffected by the re-normalisation itself; the prior subtraction is what moves them.
        out = F.log_softmax(out - log_prior, dim=-1)
    return out
