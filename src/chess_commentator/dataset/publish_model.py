"""Build (and optionally push) the Hugging Face model repo for the square classifier.

The repo is deliberately **standalone**: it ships a generated ``modeling.py`` so the weights can be
loaded with nothing but ``torch`` and ``safetensors``. A bare safetensors file would be a dict of
tensors named ``image_feature_extraction_1.0.weight`` and so on, and reconstructing a 330k-parameter
CNN with a zero-init residual branch and a depthwise dilated convolution from tensor shapes alone is
not a reasonable ask.

**Licensing.** ``modeling.py`` is generated from ``cnn/model.py`` and ``labels.py``, and is
published under Apache-2.0 alongside the weights, while the same two files stay GPL-3.0-or-later in
the repository. That is not a contradiction: the project's GPL obligation comes from python-chess,
which is load-bearing via ``chess.Board.legal_moves`` -- and neither of these two files imports it,
directly or transitively. Both import only ``torch``. The copyright holder is free to offer the same
work under more than one license, which is the same reasoning that already puts ``weights/`` under
Apache-2.0.

**Generated, not hand-copied**, so the copy cannot drift from the source: rerun this module after
any change to the architecture or the label encodings. ``tests/dataset/test_publish_model.py``
fails if the generated module and the package disagree.

Run with::

    uv run python -m chess_commentator.dataset.publish_model
    uv run python -m chess_commentator.dataset.publish_model --push
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path("dist") / "huggingface" / "cnn-model"
REPO_ID = "felixfabricius/robot-chess-commentator-cnn"

# W&B run yjgkplgc -- the `optimised_plus_prior_correction` row of evaluation/results.csv, which is
# benchmarked WITH prior correction. It is a separate training run from `optimised`, not the same
# weights with a flag flipped: 44 of its 45 tensors differ, because prior-corrected eval loss also
# drives checkpoint selection (see the run-name comment in cnn/run.py). The only tensor the two
# share is log_prior itself, which both fit from the same training label counts.
#
# Everything that consumes these weights must therefore run with prior_correction=True, or it is
# running a configuration nobody measured.
CHECKPOINT = (
    REPO_ROOT / "trained_models" / "optimised_plus_prior_correction" / "weights.safetensors"
)

SOURCE_FILES = (
    REPO_ROOT / "src" / "chess_commentator" / "labels.py",
    REPO_ROOT / "src" / "chess_commentator" / "cnn" / "model.py",
)

MODELING_HEADER = '''"""Square classifier for the robot chess commentator -- architecture and label encodings.

GENERATED FILE. Assembled from `labels.py` and `cnn/model.py` of
https://github.com/felixfabricius/robot-chess-commentator by
`chess_commentator.dataset.publish_model`. Edit those, not this.

Copyright 2026 Felix Fabricius. Licensed under the Apache License, Version 2.0.
This file is also available under GPL-3.0-or-later as part of the repository above; neither of its
two source files imports python-chess, so the repository's GPL obligation does not reach them.

    import numpy as np, torch
    from safetensors.torch import load_file
    from modeling import SquareClassifierMultiHead, TARGET_MAP, reconstruct_13way_logprobs

    model = SquareClassifierMultiHead()
    model.load_state_dict(load_file("model_state_dict.safetensors"))
    model.eval()
"""

import torch
import torch.nn.functional as F
from torch import nn

'''


def strip_module_preamble(source: str) -> str:
    """Drop a file's module docstring and import block, keeping its definitions.

    The generated module supplies its own header and imports, and the two sources import only
    torch, so nothing else can be lost here -- ``verify_generated`` re-checks that by compiling and
    comparing against the package.
    """
    lines = source.splitlines()
    start = 0
    in_docstring = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0 and stripped.startswith('"""'):
            in_docstring = not stripped.endswith('"""') or stripped == '"""'
            start = index + 1
            continue
        if in_docstring:
            if stripped.endswith('"""'):
                in_docstring = False
            start = index + 1
            continue
        if stripped.startswith(("import ", "from ")) or not stripped:
            start = index + 1
            continue
        break
    return "\n".join(lines[start:]).strip() + "\n"


def strip_main_block(source: str) -> str:
    """Drop a trailing ``if __name__ == "__main__":`` block.

    These are CLI harnesses for running a module out of the repository -- meaningless in a
    file published on its own, and the only place either source names ``chess_commentator``.
    """
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.startswith('if __name__ == "__main__":'):
            return "\n".join(lines[:index]).rstrip() + "\n"
    return source


def build_modeling(source_files=SOURCE_FILES) -> str:
    parts = [MODELING_HEADER]
    for path in source_files:
        parts.append(f"# ---------------------------------------------------------------- {path.name}\n")
        body = strip_main_block(strip_module_preamble(path.read_text(encoding="utf-8")))
        parts.append(body)
        parts.append("\n")
    return "".join(parts)


def verify_generated(modeling_source: str) -> None:
    """Fail loudly if the generated module disagrees with the package it was generated from."""
    namespace: dict = {}
    exec(compile(modeling_source, "modeling.py", "exec"), namespace)  # noqa: S102

    from chess_commentator.cnn.model import SquareClassifierMultiHead as Reference
    from chess_commentator.labels import TARGET_MAP, TOP_LEFT_OHE_MAP

    if namespace["TARGET_MAP"] != TARGET_MAP:
        raise AssertionError("generated TARGET_MAP differs from chess_commentator.labels")
    if namespace["TOP_LEFT_OHE_MAP"] != TOP_LEFT_OHE_MAP:
        raise AssertionError("generated TOP_LEFT_OHE_MAP differs from chess_commentator.labels")

    generated_keys = list(namespace["SquareClassifierMultiHead"]().state_dict())
    if generated_keys != list(Reference().state_dict()):
        raise AssertionError("generated model state_dict keys differ from the package model")


def build(out_dir: Path = OUT_DIR, checkpoint: Path = CHECKPOINT) -> Path:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found. trained_models/ is gitignored -- fetch the checkpoint first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    modeling = build_modeling()
    verify_generated(modeling)
    (out_dir / "modeling.py").write_text(modeling, encoding="utf-8")

    shutil.copyfile(checkpoint, out_dir / "model_state_dict.safetensors")
    shutil.copyfile(REPO_ROOT / "weights" / "LICENSE", out_dir / "LICENSE")
    (out_dir / "README.md").write_text(build_card(), encoding="utf-8")
    return out_dir


def build_card() -> str:
    return """---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-classification
tags:
- chess
- robotics
- board-state-recognition
datasets:
- felixfabricius/robot-chess-commentator-squares
---

# Robot chess commentator: board-reading CNN

Reads a chess position from one photograph. Used by a [Reachy
Mini](https://github.com/felixfabricius/robot-chess-commentator) that watches and commentates a chess game on 
a physical board.

## Results

The task is identifying the move that was played, so move accuracy is the metric that matters.

| | val | test |
| --- | --- | --- |
| **Move accuracy** | 94.6% | 94.7% |
| Per-square accuracy | 73.8% | 74.6% |

Board-weighted over 74 val and 75 test positions, **with prior correction on** -- the configuration
this checkpoint was selected and measured in.

For comparison, the best vision-language baseline measured on the same test split (Claude Opus 5,
per-square log-probabilities, low-effort thinking) reaches 21.3% move accuracy at roughly
$0.20 per board. This model runs locally in about a second at no cost.

## Usage

The model does not take a photograph. It takes a single rectified, masked square crop, so the
board must first be warped to a top-down view and cut into 64 crops (`Processor.warp()` /
`.cutout()` in the repository). The published dataset ships those crops directly:

```python
import numpy as np, torch
from datasets import load_dataset
from safetensors.torch import load_file
from modeling import (
    SquareClassifierMultiHead, TARGET_MAP, TOP_LEFT_OHE_MAP, reconstruct_13way_logprobs,
)

model = SquareClassifierMultiHead()
model.load_state_dict(load_file("model_state_dict.safetensors"))
model.eval()

row = load_dataset("felixfabricius/robot-chess-commentator-squares", "squares", split="test")[0]
rgb = np.array(row["image"])                 # (144, 144, 3) uint8
mask = np.array(row["mask"]) // 255          # (144, 144)    uint8, {0, 1}

image = torch.from_numpy(np.dstack([rgb, mask])).permute(2, 0, 1).float()
image[:3] /= 255.0                           # RGB to [0, 1]; the mask channel stays {0, 1}

metadata = torch.zeros(1, 4)
corner = ["a8", "a1", "h1", "h8"][row["top_left_corner"]]
metadata[0, TOP_LEFT_OHE_MAP[corner]] = 1

with torch.no_grad():
    heads = (t.squeeze(0) for t in model(image[None], metadata))
    logprobs = reconstruct_13way_logprobs(*heads, log_prior=model.log_prior)

names = list(TARGET_MAP)
print(names[int(logprobs.argmax())])
```

There is **no mean/std normalisation** -- the only scaling is uint8 / 255.

## Architecture

| | |
| --- | --- |
| Class | `SquareClassifierMultiHead` |
| Size | 1.3 MB fp32 -- 328,853 trainable parameters; 330,088 values in the state dict, the rest BatchNorm buffers and the 13-way log-prior |
| Input | `4x144x144` (RGB + square mask) plus a 4-dim one-hot of which board corner is top-left |
| Output | three heads -- `empty` (1 logit), `color` (2), `type` (6) |
| Source | W&B run `yjgkplgc` |

A three-block convolutional trunk with a residual branch (last BatchNorm zero-initialised, so it
starts as the identity) whose depthwise dilated convolution widens the receptive field to roughly
seven board cells -- enough to see a tall piece leaning in from a neighbouring square.

The three heads are recombined into 13-way log-probabilities by `reconstruct_13way_logprobs`, under
a conditional independence assumption between colour and type given non-empty. Factoring the
problem this way lets every piece image teach the colour head, rather than splitting the evidence
across twelve piece classes.

Note that `empty_head` is trained with `BCEWithLogitsLoss` against `is_piece`, so
`sigmoid(logit_empty)` is P(**piece**) despite the name.

### Prior correction is on, and this checkpoint expects it

`reconstruct_13way_logprobs(..., log_prior=model.log_prior)` subtracts the training prior, so scores
reflect the evidence for each class rather than how frequent that class happened to be in training
-- and `empty` is 56% of all squares, so the prior is far from flat.

This checkpoint was **selected and benchmarked with the correction on**, so leave it on. Running it
without is a configuration nobody measured. The snippet above does this by passing `log_prior`; the
repository's `BoardEstimator` does it with `prior_correction=True`.

## Training data

[`felixfabricius/robot-chess-commentator-squares`](https://huggingface.co/datasets/felixfabricius/robot-chess-commentator-squares)
contains 23,744 labelled squares from 371 positions.

## Limitations

Trained on one board, one piece set and one camera across 50 setups. It might transfer poorly to a 
different chess set or very different lightning.

## License

Apache-2.0. Copyright 2026 Felix Fabricius. `modeling.py` is generated from the
[GPL-3.0-or-later repository](https://github.com/felixfabricius/robot-chess-commentator) and
published under Apache-2.0; see the header of that file.
"""


def push(out_dir: Path = OUT_DIR, repo_id: str = REPO_ID) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model")
    print(f"pushed -> https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    out_dir = build(args.out_dir, args.checkpoint)
    print(f"built {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name:32s} {path.stat().st_size:>10,} B")

    if args.push:
        push(out_dir, args.repo_id)


if __name__ == "__main__":
    main()
