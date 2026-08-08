"""Scoring the six board-reading methods against each other.

Split out of the old `model/` package, where it sat next to a same-named `evaluate.py` that does
something entirely different (per-epoch validation during training). It never belonged there: the
CNN is only one of the six methods it measures, and it imports nothing from `cnn/`.
"""
