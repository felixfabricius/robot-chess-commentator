"""The 1.3 MB square classifier: architecture, dataset, training loop, and the Hydra entry point.

Named `cnn` rather than `model` because config.yaml already uses "model" for the *Claude* model id
(`vision.model_version`), and because the old `model/` package had three different meanings of the
word stacked in one directory.

`evaluate.py` and `run.py` import upward into `perception/` and `game.py` on purpose: the metric
that decides whether a checkpoint is good is board-level, not square-level, so validation has to
run the consumer. Nothing is imported here, so that never hardens into an import cycle.

The label encoding lives one level up in `labels.py`, not here -- inference needs it too, and it
should not have to import the training package to read a label.
"""
