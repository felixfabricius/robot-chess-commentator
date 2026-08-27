# robot-chess-commentator
This codebase enables a [Reachy Mini](https://huggingface.co/blog/reachy-mini) robot to commentate chess games. It recognises moves with 94.5% first-try accuracy, and then generates move-specific comments.
<p align="center">
  <img src=".github/assets/demo_compressed.gif" alt="Robot comments on a chess move">
</p>

## Layout

The package is `chess_commentator`, under `src/`. `main.py` orchestrates various package parts to let the robot commentate games:
| | |
| --- | --- |
| **During a game** | |
| _Set up_| |
| `session.py` | orientate robot towards chess board; locate chess board corners in camera image  |
| `game.py` | keep track of the board position |
| _For each move_ | |
| `player_input.py` | register moves and move announcement rejections via keyboard or robot antennas |
| `perception/` | capture and warp images; cut out 64 inputs for square classifier; call on square classifier to classify squares |
| `game.py/` | estimate move from square classification results; use chess engine to rate move |
| `voice/` | announce move; generate move comment text; turn text into audio using Kokoro TTS |
| | |
| **Other parts of the package** | |
| `cnn/` | train and evaluate 1.3 MB convolutional neural network for square classification |
| `dataset/` | facilitate creation of dataset comprising 23,744 labeled images of chess squares via useful interface |
| `vlm/` | use various Claude versions and prompts to recognise moves |
| `benchmark/` | generate evaluation results for different CNN-powered and Claude-powered move recognition approaches |
| `board.py` and `labels.py` | shared vocabulary for chess squares, pieces and classification scores |

Next to the package, the repo also includes
- `demo/`: demonstrate move recognition system via demo script - can be run without a robot _(see [Demo](#demo))_
- `evaluation/`: store and analyse eval results created via `chess_commentator.benchmark.harness`
- `weights/`: shipped model; also available via HuggingFace under _**insert license**_ and _**provide url**_
- `tests/`: mirrors package structure; does not require robot or API keys
- `config.yaml`: various knobs


## How to use
### Demo
The 

### Commentate chess games
If you a 

> **TODO:** this section only documents the speech cache. Still to write: `uv sync`, installing
> Stockfish, the `.env` file (`ANTHROPIC_API_KEY`), and board/camera calibration.

### Pregenerate the move announcements

```bash
uv run python -m chess_commentator.voice.pregenerate
```

Run this once before the first game. It synthesizes the ~150 fragments that every move
suggestion is spliced together from — `"E2 to,"`, `"E4?"`, `"Castle kingside?"` and so on — and
caches them under `.cache/speech/<voice>/` (gitignored, ~10 MB). Takes 5–8 minutes. It is
resumable: a crash keeps whatever it already baked, and re-running only fills the gaps.

This step is **optional**. Skip it and the robot still plays — it just falls back to
synthesizing each suggestion live, which costs about 2.2 seconds per suggested move, on the
main thread, while everyone waits. It warns you at startup if the cache is cold.

Re-run it after changing `speaker.voice` in `config.yaml`, which invalidates the cache.

## License

This project ships three things, and they are **not** under the same license.

| Artifact | License |
| --- | --- |
| Source code | [GPL-3.0-or-later](LICENSE) |
| Bundled model weights (`weights/`) | [Apache-2.0](weights/LICENSE) |
| Training dataset (on Hugging Face, not in this repo) | CC-BY-4.0 |

### Why GPL, and not something more permissive

Not by preference — by obligation. This project depends on
[python-chess](https://github.com/niklasf/python-chess), which is **GPL-3.0-or-later**, and it is
not an incidental dependency: `chess.Board.legal_moves` is what lets the robot rank candidate moves
against a noisy board reading, which is the core idea of the whole system. Distributing a program
that links a GPL library means the combined work is GPL, so GPL-3.0-or-later it is.

Every other dependency is GPL-3.0-compatible: BSD (PyTorch, torchvision, SciPy, OmegaConf),
Apache-2.0 (OpenCV, safetensors, Kokoro), MIT (anthropic, Polars, W&B, Hydra), PSF (Matplotlib),
and LGPL (pygame).

### Stockfish

[Stockfish](https://stockfishchess.org/) is also GPL-3.0, but it imposes nothing here: it is
invoked as a **separate process** over the UCI protocol and its binary is never redistributed with
this repository — you install it yourself. That is an arms-length arrangement, no different from
shelling out to any other program.

### Why the weights are Apache-2.0 rather than GPL

The model weights are the *output* of the training code, not a derivative work of it — the same
reason a program compiled with GCC does not inherit GCC's license, which the FSF
[states explicitly](https://www.gnu.org/licenses/gpl-faq.html#CanIUseGPLToolsForNF). The GPL on the
training code therefore does not reach them, and they are released permissively so that anyone can
reuse them. The same weights and license are published on Hugging Face.

Copyright © 2026 Felix Fabricius.
