"""Inspect what Claude ACTUALLY generates for one board on the move/board/fen_whole methods, to
diagnose the max_tokens runaway.

A truncated *structured-output* response hands back no text block, so the partial output is
invisible via the normal path. This script STREAMS the call and prints the raw token deltas as they
arrive -- so you see the runaway/looping content even when the final message has no complete object.

    # what the model emits under the real (structured) move schema:
    uv run python evaluation/inspect_llm.py move

    # the same board with structured output OFF -- reveals the raw free-form behaviour (is it
    # repeating a move? writing endless prose?):
    uv run python evaluation/inspect_llm.py move --free

    # try the board/fen_whole methods, a different board, more room, or reasoning:
    uv run python evaluation/inspect_llm.py board --index 3
    uv run python evaluation/inspect_llm.py move --reasoning text --max-tokens 4096

Needs ANTHROPIC_API_KEY. One call per run -- cheap.
"""
import argparse
import json
from pathlib import Path

import anthropic
import polars as pl

from chess_assistant.vision import (
    _MOVE_SCHEMA,
    _BOARD_SCHEMA,
    _FEN_WHOLE_SCHEMA,
    PROMPTS,
    _image_block,
    _previous_position_prompt,
    _with_reasoning_field,
)

_SCHEMA = {"move": _MOVE_SCHEMA, "board": _BOARD_SCHEMA, "fen_whole": _FEN_WHOLE_SCHEMA}


def inspect(method, index, split, model, max_tokens, structured, reasoning,
            data_path=Path("data/generated/data.csv")):
    data_root = Path(data_path).parent
    data = pl.read_csv(data_path).filter(
        pl.col("valid_game_position"), pl.col("move_uci") != "", pl.col("previous_board_fen") != ""
    ).filter(pl.col("setup_split") == split)
    board_ids = sorted(data.select("setup_id", "image_id").unique().iter_rows())
    setup_id, board_id = board_ids[index]
    row = data.filter(pl.col("image_id") == board_id).row(0, named=True)

    warped = data_root / setup_id / board_id / "image_warped.png"
    if method == "fen_whole":
        prompt = PROMPTS["fen_whole"][1]
    else:
        with open(data_root / setup_id / "calibration_metadata.json", encoding="utf-8") as f:
            corner_map = json.load(f)["camera_natural_orientation"]["order"]
        prompt = _previous_position_prompt(PROMPTS[method][1], row["previous_board_fen"], corner_map)

    schema = _SCHEMA[method]
    if reasoning == "text":
        schema = _with_reasoning_field(schema)
        prompt += "\nFirst reason step by step in the `reasoning` field, then fill in the answer."

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [_image_block(warped), {"type": "text", "text": prompt}]}],
    )
    # Mirror _call_claude: thinking is set EXPLICITLY (some models think by default when omitted).
    # Use summarized display for "thinking" so the streamed thinking is actually visible here.
    if reasoning == "thinking":
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    else:
        kwargs["thinking"] = {"type": "disabled"}
    if structured:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

    print(f"=== {method} | {setup_id}/{board_id} | actual move {row['move_uci']} | "
          f"structured={structured} reasoning={reasoning} max_tokens={max_tokens} ===\n")

    stop_reason, usage, n_chars = None, None, 0
    with client_stream(kwargs) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                delta = event.delta
                text = getattr(delta, "text", None) or getattr(delta, "thinking", None)
                if text:
                    print(text, end="", flush=True)
                    n_chars += len(text)
            elif event.type == "message_delta":
                stop_reason = event.delta.stop_reason or stop_reason
                usage = event.usage or usage

    print(f"\n\n--- stop_reason={stop_reason!r}  usage={usage}  chars_streamed={n_chars} ---")
    if stop_reason == "max_tokens":
        print("Runaway confirmed: the model never terminated. Look above at WHAT it repeated.")


def client_stream(kwargs):
    return anthropic.Anthropic().messages.stream(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream one Claude call to see the raw generation.")
    parser.add_argument("method", choices=["move", "board", "fen_whole"])
    parser.add_argument("--index", type=int, default=0, help="Which board in the split (0-based).")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning", default="none", choices=["none", "text", "thinking"])
    parser.add_argument("--free", dest="structured", action="store_false",
                        help="Disable structured output to see the raw free-form generation.")
    args = parser.parse_args()

    inspect(args.method, args.index, args.split, args.model, args.max_tokens,
            args.structured, args.reasoning)
