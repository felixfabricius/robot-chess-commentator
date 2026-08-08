"""Tests for the shared label encoding: decompose_label splitting a 13-way label into the three
head targets, and reconstruct_13way_logprobs putting the heads back together again.
"""

import torch
from torch import nn

from chess_commentator.model.config import (
    TARGET_MAP,
    IGNORE_INDEX,
    decompose_label,
    log_prior_from_label_counts,
    reconstruct_13way_logprobs,
)


def _confident_white_king_heads(logit_empty: float):
    """Heads that say "white king" with high confidence, at the given occupancy logit."""
    return (
        torch.tensor([logit_empty]),
        torch.tensor([[10.0, -10.0]]),                            # white
        torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0]]),  # K
    )


### decompose_label
def test_decompose_label_empty():
    assert decompose_label("empty") == (0.0, IGNORE_INDEX, IGNORE_INDEX)

def test_decompose_label_white_piece():
    # "R" -> white rook: is_piece 1.0, color 0 (white), type 2 (R)
    assert decompose_label("R") == (1.0, 0, 2)

def test_decompose_label_black_piece():
    # "n" -> black knight: is_piece 1.0, color 1 (black), type 4 (N)
    assert decompose_label("n") == (1.0, 1, 4)


### reconstruct_13way_logprobs
def test_reconstruct_sums_to_one():
    torch.manual_seed(0)
    logit_empty = torch.randn(5)
    logits_color = torch.randn(5, 2)
    logits_type = torch.randn(5, 6)
    logprobs = reconstruct_13way_logprobs(logit_empty, logits_color, logits_type)
    assert logprobs.shape == (5, 13)
    # Softmax over the last dim sums to 1, as it would for any logits...
    assert torch.allclose(torch.softmax(logprobs, dim=-1).sum(dim=-1), torch.ones(5), atol=1e-5)
    # ...but these are already normalised log-probabilities, so exp() sums to 1 as well. That is
    # the property the conditional-independence recombination has to preserve.
    assert torch.allclose(logprobs.exp().sum(dim=-1), torch.ones(5), atol=1e-5)

def test_reconstruct_argmax_unambiguous():
    # A very confident "non-empty, black, knight" should argmax to TARGET_MAP["n"].
    # sigmoid(logit_empty) == P(piece), so a confident piece has a large POSITIVE logit_empty.
    logit_empty = torch.tensor([10.0])                                      # P(piece) ~ 1
    logits_color = torch.tensor([[-10.0, 10.0]])                            # black
    logits_type = torch.tensor([[-10.0, -10.0, -10.0, -10.0, 10.0, -10.0]])  # N (type index 4)
    logprobs = reconstruct_13way_logprobs(logit_empty, logits_color, logits_type)
    assert logprobs.argmax(dim=-1).item() == TARGET_MAP["n"]

def test_reconstruct_occupancy_direction():
    # Guards the occupancy sign: sigmoid(logit_empty) == P(piece) (empty head trained on is_piece).
    color = torch.tensor([[10.0, -10.0]])                             # white
    king = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])  # K
    # confident piece (large +ve logit_empty) -> argmax is a piece, not empty
    piece = reconstruct_13way_logprobs(torch.tensor([10.0]), color, king)
    assert piece.argmax(dim=-1).item() == TARGET_MAP["K"]
    # confident empty (large -ve logit_empty) -> argmax is empty regardless of the color/type heads
    empty = reconstruct_13way_logprobs(torch.tensor([-10.0]), color, king)
    assert empty.argmax(dim=-1).item() == TARGET_MAP["empty"]

def test_reconstruct_matches_cross_entropy():
    # The log-probs must be a drop-in for logits: CrossEntropyLoss over them reproduces
    # -log(p_target) computed directly from the reconstructed probabilities.
    torch.manual_seed(1)
    logit_empty = torch.randn(4)
    logits_color = torch.randn(4, 2)
    logits_type = torch.randn(4, 6)
    logprobs = reconstruct_13way_logprobs(logit_empty, logits_color, logits_type)
    target = torch.tensor([TARGET_MAP["empty"], TARGET_MAP["K"], TARGET_MAP["n"], TARGET_MAP["p"]])

    ce = nn.CrossEntropyLoss()(logprobs, target)
    probs = logprobs.exp()
    manual = -torch.log(probs[torch.arange(4), target]).mean()
    assert torch.allclose(ce, manual, atol=1e-5)


### log_prior_from_label_counts
def test_log_prior_is_a_normalised_distribution():
    counts = {"empty": 500, "K": 10, "p": 80}
    log_prior = log_prior_from_label_counts(counts)
    assert log_prior.shape == (13,)
    assert torch.allclose(log_prior.exp().sum(), torch.tensor(1.0), atol=1e-6)

def test_log_prior_orders_by_frequency():
    counts = {"empty": 500, "P": 80, "K": 10}
    log_prior = log_prior_from_label_counts(counts)
    assert log_prior[TARGET_MAP["empty"]] > log_prior[TARGET_MAP["P"]]
    assert log_prior[TARGET_MAP["P"]] > log_prior[TARGET_MAP["K"]]

def test_log_prior_absent_class_is_finite_and_smallest():
    """The guard that matters for the data-efficiency ablation: a class missing from a subsampled
    train split must get a finite, small prior -- not -inf (which would send its corrected score to
    +inf) and not 0.0 (which would claim P(class) == 1).
    """
    counts = {"empty": 1000, "P": 100}  # "Q" never observed
    log_prior = log_prior_from_label_counts(counts)
    absent = log_prior[TARGET_MAP["Q"]]
    assert torch.isfinite(absent)
    assert absent < 0                                  # a real log-probability, not 0.0
    assert absent == log_prior.min()                   # the least likely class
    assert absent < log_prior[TARGET_MAP["P"]]

def test_log_prior_all_classes_absent_is_uniform():
    log_prior = log_prior_from_label_counts({})
    assert torch.allclose(log_prior, torch.full((13,), -torch.tensor(13.0).log()), atol=1e-6)


### reconstruct_13way_logprobs -- Bayesian prior correction
def test_reconstruct_log_prior_is_optional():
    """Guards the signature: evaluate.py and vision.py both call this with three positional args."""
    torch.manual_seed(0)
    heads = (torch.randn(4), torch.randn(4, 2), torch.randn(4, 6))
    assert torch.allclose(
        reconstruct_13way_logprobs(*heads), reconstruct_13way_logprobs(*heads, log_prior=None)
    )

def test_reconstruct_prior_correction_stays_normalised():
    torch.manual_seed(0)
    heads = (torch.randn(5), torch.randn(5, 2), torch.randn(5, 6))
    log_prior = log_prior_from_label_counts({"empty": 500, "P": 80, "K": 10})
    corrected = reconstruct_13way_logprobs(*heads, log_prior=log_prior)
    assert corrected.shape == (5, 13)
    assert torch.allclose(corrected.exp().sum(dim=-1), torch.ones(5), atol=1e-5)

def test_reconstruct_prior_correction_covers_all_13_classes():
    """The correction must subtract the prior from EVERY class, "empty" included.

    Since corrected == log_softmax(uncorrected - log_prior), the quantity
    (corrected - uncorrected + log_prior) is the log-normaliser: one constant per row, identical
    across all 13 classes. If any class were skipped (e.g. "empty", by far the largest prior term),
    its entry would differ and the spread below would be non-zero.
    """
    torch.manual_seed(0)
    heads = (torch.randn(6), torch.randn(6, 2), torch.randn(6, 6))
    log_prior = log_prior_from_label_counts({"empty": 500, "P": 80, "K": 10, "q": 3})
    uncorrected = reconstruct_13way_logprobs(*heads)
    corrected = reconstruct_13way_logprobs(*heads, log_prior=log_prior)

    residual = corrected - uncorrected + log_prior          # (6, 13)
    spread = residual.max(dim=-1).values - residual.min(dim=-1).values
    assert torch.allclose(spread, torch.zeros(6), atol=1e-5)

def test_reconstruct_prior_correction_matches_bayes_identity():
    torch.manual_seed(1)
    heads = (torch.randn(4), torch.randn(4, 2), torch.randn(4, 6))
    log_prior = log_prior_from_label_counts({"empty": 300, "R": 40, "n": 5})
    expected = torch.log_softmax(reconstruct_13way_logprobs(*heads) - log_prior, dim=-1)
    assert torch.allclose(reconstruct_13way_logprobs(*heads, log_prior=log_prior), expected, atol=1e-6)

def test_reconstruct_uniform_prior_is_a_noop():
    """A flat prior carries no information, so correcting by it must not move anything."""
    torch.manual_seed(2)
    heads = (torch.randn(4), torch.randn(4, 2), torch.randn(4, 6))
    uniform = torch.full((13,), -torch.tensor(13.0).log())
    assert torch.allclose(
        reconstruct_13way_logprobs(*heads, log_prior=uniform),
        reconstruct_13way_logprobs(*heads),
        atol=1e-5,
    )

def test_reconstruct_zero_prior_is_a_noop_up_to_normalisation():
    """All-zeros is the model's default buffer value, and must leave the decision untouched."""
    torch.manual_seed(3)
    heads = (torch.randn(4), torch.randn(4, 2), torch.randn(4, 6))
    assert torch.allclose(
        reconstruct_13way_logprobs(*heads, log_prior=torch.zeros(13)),
        reconstruct_13way_logprobs(*heads),
        atol=1e-5,
    )

def test_reconstruct_prior_correction_promotes_rare_class():
    """The point of the correction: a class that was rare in training stops being penalised for it.

    Occupancy is deliberately tilted towards empty, so the uncorrected argmax is "empty"; once the
    dominant empty prior is divided out, the (rare) white king wins on the evidence.
    """
    heads = _confident_white_king_heads(-1.0)
    counts = {"empty": 900, "K": 5}  # empty ~massively over-represented, K rare
    log_prior = log_prior_from_label_counts(counts)

    assert reconstruct_13way_logprobs(*heads).argmax(dim=-1).item() == TARGET_MAP["empty"]
    corrected = reconstruct_13way_logprobs(*heads, log_prior=log_prior)
    assert corrected.argmax(dim=-1).item() == TARGET_MAP["K"]

def test_reconstruct_prior_correction_is_drop_in_for_cross_entropy():
    """Corrected output must remain usable as logits, i.e. still a proper log-prob vector."""
    torch.manual_seed(4)
    heads = (torch.randn(3), torch.randn(3, 2), torch.randn(3, 6))
    log_prior = log_prior_from_label_counts({"empty": 200, "B": 20, "k": 2})
    logprobs = reconstruct_13way_logprobs(*heads, log_prior=log_prior)
    target = torch.tensor([TARGET_MAP["empty"], TARGET_MAP["B"], TARGET_MAP["k"]])

    ce = nn.CrossEntropyLoss()(logprobs, target)
    manual = -torch.log(logprobs.exp()[torch.arange(3), target]).mean()
    assert torch.allclose(ce, manual, atol=1e-5)
