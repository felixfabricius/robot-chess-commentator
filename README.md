# robot-chess-commentator

A [Reachy Mini](https://huggingface.co/blog/reachy-mini) desktop robot that watches two humans play
physical chess, reads the board from a single camera photo with a custom 1.3 MB CNN, ranks the legal
moves to work out what you just played, and then comments on it out loud — praising the good moves
and roasting the bad ones.

> **Note:** this README is a work in progress. Setup instructions, architecture, and results are
> still to come — the Setup section below is a stub covering one step of many.

## Layout

The package is `chess_commentator`, under `src/`. Reading down the stack:

| | |
| --- | --- |
| `main.py` | the game loop — one iteration is one move |
| `session.py` | setup directory + putting the robot in its capture pose |
| `game.py` | board state; ranks every *legal* move against a noisy reading |
| `player_input.py` | "was there an input?", via the antennas or the keyboard |
| `board.py`, `labels.py` | the shared vocabularies: square/piece names, and the 13-way label encoding |
| `perception/` | camera frame → rectified board → 64 square crops → a board reading |
| `vlm/` | Claude-on-**images**: prompts, schemas, and the whole-image reading strategies |
| `voice/` | Claude-on-**text** + Kokoro TTS + playback |
| `cnn/` | the 1.3 MB square classifier and its training loop |
| `dataset/` | offline tools that produce the labelled training data |
| `benchmark/` | scoring the six board-reading methods against each other |

Every `__init__.py` is import-free, so the heavy and hardware-bound dependencies (torch,
anthropic, kokoro, reachy_mini) stay confined to the modules that actually need them — importing
`board` pulls in nothing, and importing `vlm` pulls in no torch.

`tests/` mirrors this tree. `demo/` sits outside the package and consumes it as an installed
dependency; it is the only end-to-end run that needs neither the robot nor an API key.

## Setup

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
