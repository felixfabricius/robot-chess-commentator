"""
Input:
- Type of estimator
    A) CNN. Specify:
    - Path to the CNN weights. Different versions: 
        - for models trained using different masks and crops.
        - for models trained with different amounts of data 
        
    For ablations:
    - Path to csv data file. This would allow me to also generate CIs 
      for the ablation experiments with different masks etc.
    - Model type: multi head vs. non multi head architecture.

    B) LLM. Specify:
    - Type of LLM (let's stay within Anthropic ecosystem for now).
    - Prompt version; can store prompts in some additional config folder
    - Evaluation level:
        - Correct move only: for this: 
            - store previous FEN. for prompt: maybe print this as 
        - Square metrics also. In this case: will be more compatible with BoardEstimator class.

Output:
- For general case: square-level prediction; eval a single model
    - Point estimate and Confidence interval for: proportion of correct squares; 
      how to bootstrap this? do we need to account for fact that proportion of correctness
      will differ across label when bootstrapping?
    - Point estimate and Confidence interval for: proportion of correct boards 
    - Point estimate and CI for: board rank metric

    - Point estimate for: time taken for inference per average chess game 
      (which is around 40 moves or 80 board positions)
    - If applicable (API): cost for inference per avrage chess game


- For case: only board-level or move-level prediction using LLM
    - Point estimate (and CI?) for proportion of correct boards
    - Point estimate (and CI?) for proportion of suggested legal moves

- Stuff that could additionally be interesting:
    - confusion matrices on a square-level. WandB style perhaps? 

- For case: compare CNN to LLM
    - Interested primarily in: 
      Null hypothesis: the CNN is not beter than the LLM at classifying moves.
      Alternative: it is better
    - Test:
      for each setup, need paired difference for
        - proportion of correct moves; here: be conservative. if draw: then CNN is not better.
        - move rank metric; draw less likely, so I would actually prefer this. but more difficult to tell story.

 



    


How to format output: perhaps write to a CSV file?

"""
import json
import numpy as np
import polars as pl
from pathlib import Path, WindowsPath

from chess_assistant.game import ChessGame
from chess_assistant.vision import BoardEstimator
from chess_assistant.config import SQUARES, PIECES

def main(
    model_type: str,
    prompt_version: int | None,
    model_weights_path: Path | None,
    prior_correction: bool = False,
    data_path: Path = Path("data/generated/data.csv"),
    output_path: Path = Path("evaluation_results.csv")
):
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    data_root = data_path.parent

    # For the base case: evaluate style loop; no need to use dataloader though.
    # Reason: want to iterate over setup ids, and board ids. Perhaps double for-loop is the most sensible choice here.
    assert model_type in ["LLM", "CNN"]

    data = pl.read_csv(data_path)
    sample_calibration_metadata_path = data["calibration_metadata_path"][0]
    val_data = data.filter(pl.col("setup_split").eq("val"))
    test_data = data.filter(pl.col("setup_split").eq("test"))

    if model_type == "CNN":
        board_estimator = BoardEstimator(
            model_type=model_type,
            model_weights_path=model_weights_path,
            # Used to determine top-left corner for each setup; will be overwritten when iterating over setups.
            calibration_metadata_path=sample_calibration_metadata_path,
            prior_correction=prior_correction
        )

    # TODO: this doesn't apply for LLM variant yet;
    # and in particular: this doesn't apply for the LLM variant where we only get an estimated move (rather than estimated squares)
    # In that case, don't even need a ChessGame object I think. (Except: FEN is neat to check if provided move is legal.)
    for split, df in [("val", val_data), ("test", test_data)]:
        unique_setups = df["setup_id"].unique().to_list()
        unique_setups.sort()

        row = {}

        for setup in unique_setups:
            df_setup = df.filter(
                pl.col("setup_id").eq(setup),
                # Could keep boards with invalid position to estimate proportion
                # of correct squares, but that proportion is much less important than the
                # board level metrics, so filtering down to save comp. resources.
                pl.col("valid_position")
            )
            unique_boards = df_setup["image_id"].unqiue().to_list()
            unique_boards.sort()

            correct_square = []
            correct_board = []
            board_rank = []
            n = len(unique_boards)

            # Correct the current top-left corner value
            # TODO: is WindowsPath the correct choice here for robustness? (would like to also be able to execute this code on Mac)
            calibration_metadata_path = WindowsPath(df_setup["calibration_metadata_path"][0]) 
            with open(calibration_metadata_path, "r", encoding="utf-8") as f:
                calibration_metadata = json.load(f)
                # Safe to also set also if LLM.
                board_estimator.top_left_corner = calibration_metadata["camera_natural_orientation"]["order"]["tl"]

            for board in unique_boards:
                correct_square_for_board = []
                squares_dir = data_root / setup / board / "squares"
                assert squares_dir.is_dir()
                board_estimator.estimate_board(squares_dir)
                board_estimate = board_estimator.board_estimate

                df_board = df_setup.filter(pl.col("image_id").eq("board"))
                assert df_board.height > 0
                if df_board.height != 64:
                    print(f"Warning: number of rows for board_id {board} in split {split}: {df_board.height}")

                # Given the board estimate, compute proportion of correctly estimated squares
                for square in SQUARES:
                    label = df_board.filter(pl.col("square").eq(square))["label"].item()
                    square_estimate = getattr(board_estimate, square)
                    max_logit = float("-inf")
                    pred_piece = ""
                    for piece in PIECES:
                        logit = getattr(square_estimate, piece)
                        if logit > max_logit:
                            max_logit = logit
                            pred_piece = piece
                    correct_square_for_board.append(label == pred_piece)
                correct_square.append(np.mean(correct_square_for_board))

                # Check if move is correct
                previous_board_fen = df_board["previous_board_fen"][0]
                game = ChessGame(fen=previous_board_fen, model_type="CNN") # TODO: change this parameter name. By default want logits and CE loss-based move estimation (even for LLM). 
                estimated_moves = game.estimate_move(board_estimate)

                actual_move = df_board["move_uci"][0]
                correct_board.append(actual_move==estimated_moves[0]["move"])

                n_moves = len(game.board.legal_moves())
                if n_moves < 2:
                    board_rank.append(None)
                    continue
                actual_move_position = 0
                for move in estimated_moves:
                    if move["move"] == actual_move:
                        break
                    actual_move_position += 1
                board_rank.append(1 - actual_move_position / (n_moves - 1))

            row[f"{split}_{setup}_correct_square"] = correct_square
            row[f"{split}_{setup}_correct_board"] = correct_board 
            row[f"{split}_{setup}_board_rank"] = board_rank

    evaluation_results = pl.concat([pl.read_csv(output_path), pl.DataFrame(row)], how="vertical")
    evaluation_results.write_csv(output_path)
