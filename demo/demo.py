#%%
# Set to False to run the image processing steps for two images at the same time.
ONE_EXAMPLE_ONLY = True

#%%
import json
import sys
import torch
from pathlib import Path

import chess_assistant
from chess_assistant.image_processing import Processor
from chess_assistant.vision import BoardEstimate, BoardEstimator
from chess_assistant.game import ChessGame
from chess_assistant.config import PIECES, SQUARES, PIECE_DISPLAY

# Anchor every path to the repo root rather than the working directory, so the demo behaves
# identically wherever it is launched from. chess_assistant is installed from
# <repo>/src/chess_assistant, so its __file__ locates the repo without relying on __file__ of
# this script (which is not defined when cells are run interactively).
REPO_ROOT = Path(chess_assistant.__file__).resolve().parents[2]
WEIGHTS_PATH = REPO_ROOT / "weights" / "model_state_dict.safetensors"
OUT_DIR = REPO_ROOT / "demo" / "out" / "setup_1"

# Run with --pause to walk through the demo one step at a time.
PAUSE = "--pause" in sys.argv


def pause(next_step: str) -> None:
    """Wait for Enter before `next_step`, but only when running with --pause."""
    if PAUSE:
        input(f"\n  [Press Enter to {next_step}] ")


setup_1_path = REPO_ROOT / "demo" / "assets" / "setup_1"
calibration_metadata_path = setup_1_path / "calibration_metadata.json"
position_info = {
    "position_1": {
        "img_path": setup_1_path / "position_1" / "image.png",
        "metadata_path": setup_1_path / "position_1" / "metadata.json"
    },
    "position_2": {
        "img_path": setup_1_path / "position_2" / "image.png",
        "metadata_path": setup_1_path / "position_2" / "metadata.json"
    },
}
positions = list(position_info.keys())

def warp_image(img_path: Path, calibration_metadata_path: Path, img_processor, position: str) -> Path:
    warped_image_path = img_processor.warp(img_path, OUT_DIR / position / "warped_image.png")
    return warped_image_path

def cutout_squares(img_processor: Processor, warped_image_path: Path) -> Path:
    squares_dir = img_processor.cutout(warped_image_path)
    return squares_dir

def estimate_move(
    squares_dir: Path,
    calibration_metadata_path: Path,    
    # FEN encodes a chess position. To detect a move, we look at 
    # all the moves that are legal at a given chess position, and then
    # use our "board_estimate" to determine which of them is most likely.
    previous_fen: str 
) -> tuple[list[str], BoardEstimate]:
    board_estimator = BoardEstimator(
        model_type="CNN",
        calibration_metadata_path=calibration_metadata_path,
        model_weights_path=WEIGHTS_PATH
    )
    board_estimate = board_estimator.estimate_board(squares_dir)
    
    game = ChessGame(fen=previous_fen)
    # This is a list of all possible moves, ordered descendibgly by
    # how likely they are to be the one correct move given the previous
    # board position and an image of the current board state.
    estimated_moves = game.estimate_move(board_estimate)
    losses = torch.tensor([move_dict["loss"] for move_dict in estimated_moves], dtype=torch.float32)
    move_probabilities = list(torch.softmax(-losses, dim=0).numpy())
    return list(zip([move_dict["move"] for move_dict in estimated_moves], move_probabilities)), board_estimate


def evaluate_estimate(board_estimate, board_metadata_path, position):
    with open(board_metadata_path, "r", encoding="utf-8") as f:
        board_metadata = json.load(f)
    # Evaluate how many of the images were correct
    n_correct = 0
    output = []
    # Pad piece names to the longest ("Bishop (w)" / "Knight (w)") so the probability
    # parentheses line up vertically across rows.
    name_width = max(len(name) for name in PIECE_DISPLAY.values())
    for square in SQUARES:
        label = board_metadata["piece_map"][square]
        
        square_estimate = getattr(board_estimate, square)
        logits = torch.tensor([getattr(square_estimate, piece_symbol) for piece_symbol in PIECES], dtype=torch.float32)
        probs = torch.softmax(logits, dim=0)
        top_2_probs, top_2_indices = torch.topk(probs, k=2)
        most_likely = PIECES[top_2_indices[0]]
        most_likely_prob = top_2_probs[0]
        second_most_likely = PIECES[top_2_indices[1]]
        second_most_likely_prob = top_2_probs[1]

        if label == most_likely:
            n_correct += 1

        output.append(
            f"{square:<7}| "
            f"{('INCORRECT' if label != most_likely else ''):<11}| "
            f"{PIECE_DISPLAY[label]:<14}| "
            f"{f'{PIECE_DISPLAY[most_likely]:<{name_width}} ({most_likely_prob.item():.1%})':<24}| "
            f"{f'{PIECE_DISPLAY[second_most_likely]:<{name_width}} ({second_most_likely_prob.item():.1%})':<24}"
        )

    header = (
        f"{'Square':<7}| {'Incorrect?':<11}| {'Actual piece':<14}| "
        f"{'Pred. most likely':<24}| {'Pred. 2nd most likely':<24}"
    )
    print(f"Evaluating estimates of individual squares for {position}")
    print("---------------------------------------------------------\n")
    print(f"Correct square estimates: {n_correct} / 64\n")
    print(header)
    print("-" * len(header))
    print("\n".join(output))
    print(
        "\nIt is likely the case that many of the estimates for individual squarees are incorrect, yet the estimated move ",
        "is correct (and perhaps highly confidently so).\n",
        "Part of the reason many individual square estimates may not be correct is that trainings data is dominated by 'empty' ",
        "squares, leading to empty squares being more likely to be predicted. (While possible, this is not fully offsetted by adjusting weights.)\n",
        "More importantly, the reason moves are still estimated well, is that we only compare legal moves to each other.\n",
        "We tend to estimate the correct move as long as the resulting board position looks (much less) wrong than legal alternatives!",
        sep=""
    )
    print("\n\n")

    

# %%
if ONE_EXAMPLE_ONLY:
    positions = positions[:1]
img_processor = Processor(calibration_metadata_path)

# %%
pause("warp the image")
for position in positions:
    img_path = position_info[position]["img_path"]
    warped_image_path = warp_image(img_path, calibration_metadata_path, img_processor, position)
    position_info[position]["warped_img_path"] = warped_image_path
    print((
        f"Image warping for {position}\n"
        "----------------------------\n"
        "The image has been warped based on the labelled board corners "
        "('actual_corners_px' and 'extended_board_corners' in the calibration metadata). "
        f"\nThe warped image is at: {warped_image_path}\n\n"
    ))

#%%
pause("cut out the individual squares")
for position in positions:
    warped_img_path = position_info[position]["warped_img_path"]
    squares_dir = cutout_squares(img_processor, warped_img_path)
    position_info[position]["squares_dir"] = squares_dir
    print((
        f"Cutting out individual squares for {position}\n"
        "---------------------------------------------\n"
        "Individual squares have been cut out and saved in the directory: "
        f"{squares_dir}.\n"
        "The position of the actual chess board square within its "
        "cutout tends to differ across squares: to fully capture pieces standing "
        "on squares, need to extend the cutout into a direction which depends on the "
        "camera orientation and differs across squares.\n\n"
    ))

#%%
pause("estimate the move")
for position in positions:
    with open(position_info[position]["metadata_path"], "r", encoding="utf-8") as f:
        metadata = json.load(f)
        position_info[position]["previous_position"] = metadata["previous_board_fen"]
        move = metadata["move_uci"]
        position_info[position]["move"] = move
    estimated_moves, board_estimate = estimate_move(
        position_info[position]["squares_dir"],
        calibration_metadata_path,
        position_info[position]["previous_position"]
    )
    position_info[position]["board_estimate"] = board_estimate
    print(
        f"Estimating moves for {position}\n"
        "-------------------------------\n"
        "For the 64 squares the model estimated how which piece (if any) is located there. \n",
        "Based on those estimates, the legal moves from the previous position were ordered by ",
        "how likely they are to have lead to the board position seen in the image. \n",
        f"The actual move is {move}.\n"
        f"From the previous position, there were {len(estimated_moves)} possible moves.\n",
        f"The estimated most likely ones to have been played to lead to the board image in {position_info[position]["img_path"]} are:\n",
        *tuple([f'  {i + 1}: {move} (probability: {prob:.4%})\n' for i, (move, prob) in enumerate(estimated_moves[:3])]),
        f"In this case, the estimated move is therefore {'correct' if estimated_moves[0][0] == move else 'incorrect'}."
        "\n\n",
        sep=""
    )

#%%
pause("evaluate the square estimates against the true board")
for position in positions:
    evaluate_estimate(position_info[position]["board_estimate"], position_info[position]["metadata_path"], position)
