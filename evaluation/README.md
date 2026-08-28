This folder stores evaluation results produced by `chess_commentator.benchmark`, and a script to analyse them (`eval.ipynb`).

## Evaluation methodology and validation results
### Data split
The dataset contains labelled images of 371 chess positions (which makes 371 * 64 = 23,744 chess squares). Those chess positions were recorded throughout 50 different "setups": a setup refers to a particular position of the robot relative to the chess board, and usually includes a few chess positions from a single chess game. 
A technicality: For the statistical analysis done in `eval.ipynb`, the "setup" is the unit of analysis: asking "how likely does approach A outperform approach B on a new game" really means "how likely does approach A outperform approach B on a new setup which is created in a similar manner to the setups in the dataset". 
The dataset is split into train (206 positions), validation (77) and test (88) positions. Images from the same setup are part of the same split.

### Evaluated approaches
I evaluated move recognition based on various versions of a custom convolutional neural network, and based on various uses of the Claude API.

#### CNN-powered approaches
The approaches differ in three ways:
1. Geometry: 
- Crops are padded such that each square cutout is large enough to contain any possible piece that might be standing on the square. Some variants (`..._per_square` and `optimised`) optimise the crop dimensions for each square to only contain as much of the image as necessary. Others (`..._global`) crop each square in a similar way, which results in larger than necessary crops. 
- Even if cropped opimally, images tend to contain multiple square and multiple pieces. To help the model focus on the right part of its image input, I used masks. A mask is a fourth input channel (beyond the three RGB ones) that stores a value of 1 for pixels the model should focus on, and 0 for the others. 3 mask variants are used: `none_...` means no mask; `square_...` means that the actual chess square is masked; and in the `optimised` variant, the part of the image that contains the possible piece location is masked
2. Size of training dataset: the `optimised_10k` and `optimised_5k` refer to models trained on around 5k and 10k images of squares (rather than the full 13184)
3. `optimised_plus_prior_correction` adjusts model predictions to account for dominance of some classes (most squares are empty) during training

#### Claude-powered approaches
Claude-powered approaches differ in "approach" (prompt used), model version, reasoning and resoning effort.
- Prompts:
  | | |
  | --- | --- |
  | `square_label` | one call per square; hard label |
  | `square_logits` | one call per square; return one score for each of the 13 possibilities |
  | `move` | one call per board; return candidate moves ordered by likelihood |
  | `board` | one call per board; return candidate FENs (board positions) ordered by likelihood |
  | `fen_whole` | one call per board; return a single FEN string |
- Models: Opus 5, Sonnet 5, Opus 4.8
- Reasoning: off / extended thinking; if on, then with max tokens set to 8192
- Effort: low / medium / high

All VLM runs go through the Message Batches API which reduces costs by 50%.

### Metrics
- `correct_board`: 1 if the first predicted move equals the move actually played, 0 otherwise. This is the metric that matters most for commentator functionality. Note this also works for approaches which return a position prediction rather than a move prediction: given the known previous position, each move leads to a single new position, and each new position is the result of a single move; move and position can therefore be inferred from another.
- `correct_square`: proportion of the 64 squares correctly recognised. For a given square, this equals 1 if the piece that's actually on the square received the highest score. Interestingly, positions can be correctly recognised even if this is low. *See blog for details.*
- `board_rank`: a rank-based move recognition accuracy. Unlike the `correct_board` metric, this doesn't just take into account whether the first guess is the right move or not, but how many guesses are needed to return the right move. This can only be compared across methods which provide square-only scores. 
- Other metrics: `first_output_illegal` `none_legal` and `n_suggested` for VLMs which predict entire move at once, tokens, cost
- One row per (approach, setup); each metric stored as both the per-board list and the
  setup-level mean.

### CNN results on validation set
Validation-split results, averaged over the 5 validation setups (74 positions):

| Variant | `correct_board` | `correct_square` | `board_rank` |
| --- | --- | --- | --- |
| `none_global`: no mask | 30.7% | 60.2% | 0.784 |
| `square_global`: square mask, uniform padding | 86.0% | 71.6% | 0.989 |
| `square_per_square`: square mask, per-square padding | 85.9% | 76.0% | 0.984 |
| `optimised`: piece-location mask, per-square padding | 91.0% | 72.8% | 0.997 |
| `optimised_plus_prior_correction`: add class-prior correction | 94.8% | 74.0% | 0.997 |
| `optimised_5k`: ~5k training squares | 78.8% | 63.9% | 0.980 |
| `optimised_10k`: ~10k training squares | 91.0% | 70.8% | 0.995 |


### Claude results on validation set
See results table in the blog post, or `eval.ipynb` for the results table. The best performing approach used Claude-Opus-5 with thinking-mode reasoning at low effort to return scores for each possibility for each square (`square_logits` prompt). It scored 32.7% move accuracy, and 50.9% square accuracy on the validation set.    

## Results — test split (88 board positions, 11 setups)

| Method | Move accuracy | Square accuracy | USD/board |
| --- | --- | --- | --- |
| CNN (`optimised_plus_prior_correction`) | 94.6% | 74.0% | 0 |
| Claude Opus 5, `square_logits`, thinking/low | 22.7% | 48.7% | 0.20 |

Bootstrap CIs and the paired sign test are in `eval.ipynb`, or the blog post.

## Reproducing
To regenerate results.csv: 
```bash
uv run python -m chess_commentator.benchmark.harness   # see the module docstring for flags
```
Note that the VLM results produce substantial API costs.
