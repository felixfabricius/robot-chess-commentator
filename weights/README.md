# Model weights

`model_state_dict.safetensors` — the trained board-reading CNN used by `BoardEstimator`
(`src/chess_commentator/perception/board_estimator.py`), pointed at by `vision.model_weights_path`
in `config.yaml`.

| | |
| --- | --- |
| Architecture | `SquareClassifierMultiHead` (`src/chess_commentator/cnn/model.py`) |
| Size | 1.3 MB fp32 — 328,853 trainable parameters; 330,088 values in the state dict, the rest BatchNorm buffers and the 13-way log-prior |
| Input | 4×144×144 (RGB + square mask) plus a 4-dim one-hot of which board corner is top-left |
| Output | three heads — `empty` (1 logit), `color` (2), `type` (6) — recombined into 13-way log-probs by `reconstruct_13way_logprobs` |
| Source checkpoint | W&B run `yjgkplgc` — the `optimised_plus_prior_correction` row of `evaluation/results.csv` |

The metric that matters is **move accuracy: 94.6% val / 94.7% test**, board-weighted over 74 and 75
positions, **with prior correction on**. Per-square accuracy is 73.8% val / 74.6% test — a
diagnostic, not the result. Roughly 16 of 64 squares are wrong on a typical board and the system
works anyway, because `ChessGame.estimate_move` only ever scores *legal* moves and a move only
touches 2–4 squares — see the project README.

Only the `model_state_dict` is stored here. The training checkpoint was 3.99 MB, two thirds of
which was AdamW optimizer state that inference does not need. Stored as safetensors rather than a
pickled `.pt` so that loading it cannot execute arbitrary code.

## Prior correction is on, and this checkpoint requires it

`prior_correction` subtracts the training log-prior so scores reflect the evidence for each class
rather than how frequent it was in training — and `empty` is 56% of squares, so the prior is far
from flat.

This is **not** the `optimised` checkpoint with a flag flipped. `optimised_plus_prior_correction` is
a separate training run: prior-corrected eval loss also drives checkpoint selection, so 44 of the 45
tensors differ. The only tensor the two runs share is `log_prior` itself, which both fit from the
same training label counts. Running these weights *without* the correction is a configuration
nobody measured.

`main.py` and `demo/demo.py` therefore pass `prior_correction=True` explicitly rather than relying
on `BoardEstimator`'s default — the default happens to agree today, but the point is that it is a
requirement of this checkpoint, not an incidental default.

Note `src/chess_commentator/cnn/config.yaml` still sets `inference.prior_correction: False`, which
governs *training* runs (including which epoch is checkpointed), not this file. Flip it if new runs
should be selected the same way this one was.

## License

Copyright 2026 Felix Fabricius.

The weights in this directory are licensed under the **Apache License, Version 2.0** — see
[`LICENSE`](LICENSE) in this directory. This differs from the GPL-3.0-or-later license covering the
source code in the rest of the repository; see the *License* section of the project README for why.

The same weights are published at
[`felixfabricius/robot-chess-commentator-cnn`](https://huggingface.co/felixfabricius/robot-chess-commentator-cnn),
built by `chess_commentator.dataset.publish_model`.
