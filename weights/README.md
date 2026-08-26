# Model weights

`model_state_dict.safetensors` — the trained board-reading CNN used by `BoardEstimator`
(`src/chess_commentator/perception/board_estimator.py`), pointed at by `vision.model_weights_path`
in `config.yaml`.

| | |
| --- | --- |
| Architecture | `SquareClassifierMultiHead` (`src/chess_commentator/cnn/model.py`) |
| Parameters | 330,075 (1.32 MB, fp32) |
| Input | 4×144×144 (RGB + square mask) plus a 4-dim one-hot of which board corner is top-left |
| Output | three heads — `empty` (1 logit), `color` (2), `type` (6) — recombined into 13-way log-probs by `reconstruct_13way_logprobs` |
| Source checkpoint | W&B artifact `model_and_optimizer-v18`, run `rd68eyo6`, epoch 13 |

Held-out (val) accuracy is **74.1% per square**, i.e. roughly 16 of 64 squares wrong on a typical
board. The system works anyway because `ChessGame.estimate_move` only ever scores *legal* moves,
and a move only touches 2–4 squares — see the project README.

Only the `model_state_dict` is stored here. The original checkpoint was 3.99 MB, two thirds of
which was AdamW optimizer state that inference does not need; stripping it leaves 1.32 MB. Stored
as safetensors rather than a pickled `.pt` so that loading it cannot execute arbitrary code.

## License

Copyright 2026 Felix Fabricius.

The weights in this directory are licensed under the **Apache License, Version 2.0** — see
[`LICENSE`](LICENSE) in this directory. This differs from the GPL-3.0-or-later license covering the
source code in the rest of the repository; see the *License* section of the project README for why.
