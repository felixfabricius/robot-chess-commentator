"""One Claude structured-output call: building the request, sending it, and parsing the answer.

Everything here is method-agnostic -- it knows about images, schemas and the reasoning knob, but
nothing about chess. The chess-specific prompts and schemas live in `prompts`, and the six reading
strategies that combine the two live in `strategies`.

Request *construction* is deliberately separate from *sending* (`build_message_params` vs
`call_claude`), because the benchmark harness submits the same requests through the Message Batches
API hours before it reads the answers. Both paths build byte-identical requests; only the transport
differs.
"""
import base64
import json
import time
from pathlib import Path

from dotenv import load_dotenv

# Every Anthropic client in the project is constructed after importing this module, so loading the
# .env here covers all of them from one place. This used to sit at the top of the monolithic
# vision.py; keeping it on the transport layer preserves that behaviour after the split.
load_dotenv()


class LLMResponseError(RuntimeError):
    """A Claude call returned nothing usable -- no text block (the structured JSON never completed,
    almost always because generation hit the max_tokens cap or was a refusal) or a text block that
    doesn't parse as JSON. Carries the response's stop_reason / usage / elapsed so the caller can
    still record what the (wasted) call cost and treat the board as a failed prediction rather than
    crashing the run."""

    def __init__(self, message, *, stop_reason=None, usage=None, elapsed=None):
        super().__init__(message)
        self.stop_reason = stop_reason
        self.usage = usage
        self.elapsed = elapsed


def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def infer_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    if suffix not in types:
        raise ValueError(f"Unsupported image type: {suffix}")
    return types[suffix]


def image_block(image_path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": infer_media_type(image_path),
            "data": encode_image_base64(image_path),
        },
    }


def with_reasoning_field(schema: dict) -> dict:
    """A copy of `schema` with a leading `reasoning` string property (also required), so a
    reasoning="text" call writes its chain of thought into the structured output before the answer
    fields."""
    properties = {"reasoning": {"type": "string", "description": "Step-by-step reasoning, written before the answer."}}
    properties.update(schema.get("properties", {}))
    return {
        "type": "object",
        "properties": properties,
        "required": ["reasoning"] + list(schema.get("required", [])),
        "additionalProperties": False,
    }


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def build_message_params(model, image_paths, prompt, answer_schema, reasoning="none",
                         max_tokens=1024, effort="medium"):
    """Assemble the kwargs dict for one Claude structured-output call. Shared by the synchronous
    path (`call_claude`) and the batch path (the benchmark harness builds a request per call from
    this), so both send byte-identical requests and only the transport differs. Applies the
    `reasoning` knob:
      - "none":     answer only; uses `max_tokens` as given, thinking disabled.
      - "text":     a visible `reasoning` field is added to the schema and filled before the answer;
                    `max_tokens` floored at 4096, thinking disabled.
      - "thinking": adaptive thinking (hidden), same structured final answer; `max_tokens` floored
                    at 8192, and `effort` sent to bound how much it thinks.

    `effort` ("low".."max") tunes thinking depth / total token spend. It is sent ONLY in "thinking"
    mode -- that is where runaway thinking eats the whole max_tokens budget and truncates the answer,
    and confining it there also sidesteps the 400 Opus 5 returns for disabled-thinking + xhigh/max
    effort (adaptive thinking has no separate thinking budget on current models -- `budget_tokens`
    is rejected -- so lowering effort is the lever for shorter, cheaper thinking).

    Hidden thinking is set EXPLICITLY, never left to the model default: some models (e.g. Sonnet 5)
    run adaptive thinking whenever `thinking` is omitted, and that thinking shares the max_tokens
    budget -- so an omitted param silently burns the whole budget on thinking and the answer never
    lands (stop_reason=max_tokens, thinking_tokens ~= max_tokens). Reasoning also shares the output
    budget with the (tiny) answer, hence the raised floors.
    """
    assert reasoning in ("none", "text", "thinking")
    assert effort in EFFORT_LEVELS
    schema = with_reasoning_field(answer_schema) if reasoning == "text" else answer_schema
    if reasoning == "text":
        prompt = prompt + "\nFirst reason step by step in the `reasoning` field, then fill in the answer."
        max_tokens = max(max_tokens, 4096)
    elif reasoning == "thinking":
        max_tokens = max(max_tokens, 8192)
    thinking = {"type": "adaptive"} if reasoning == "thinking" else {"type": "disabled"}

    output_config = {"format": {"type": "json_schema", "schema": schema}}
    if reasoning == "thinking":
        output_config["effort"] = effort

    content = [image_block(p) for p in image_paths] + [{"type": "text", "text": prompt}]
    return {
        "model": model,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "output_config": output_config,
        "messages": [{"role": "user", "content": content}],
    }


def parse_message(message) -> dict:
    """Extract and JSON-parse the structured answer from a completed Claude message.

    With structured output the JSON arrives in a text block. If there is none, the object never
    completed -- almost always the response hit the max_tokens cap (constrained decoding truncated
    mid-object, often because the model degenerated into runaway output) or was a safety refusal.
    A present-but-invalid text block is the same "unusable answer" outcome. Either way raise a typed
    LLMResponseError carrying stop_reason/usage so the caller can record the (wasted) cost and treat
    the board as a failed prediction rather than crashing. Shared by `call_claude` (synchronous)
    and the batch result handler; the batch path leaves `elapsed` at None (no per-call wall-clock).
    """
    text_block = next((block for block in message.content if block.type == "text"), None)
    if text_block is None:
        raise LLMResponseError(
            f"no text block to parse (stop_reason={message.stop_reason!r}, "
            f"output_tokens={message.usage.output_tokens})",
            stop_reason=message.stop_reason, usage=message.usage,
        )
    try:
        return json.loads(text_block.text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"text block is not valid JSON (stop_reason={message.stop_reason!r}): {exc}",
            stop_reason=message.stop_reason, usage=message.usage,
        ) from exc


def call_claude(client, model, image_paths, prompt, answer_schema, reasoning="none",
                max_tokens=1024, effort="medium"):
    """One synchronous Claude call with a structured (json_schema) answer. Returns
    (parsed, usage, elapsed): `parsed` is the JSON object matching `answer_schema` (plus a
    `reasoning` key when reasoning == "text"), `usage` is the token usage (for cost), `elapsed` is
    wall-clock seconds for this call (for timing). See `build_message_params` for the reasoning /
    effort knobs. A max_tokens truncation / unparseable answer raises LLMResponseError (with
    `elapsed` set)."""
    params = build_message_params(model, image_paths, prompt, answer_schema, reasoning, max_tokens, effort)

    start = time.monotonic()
    if reasoning == "thinking":
        # Stream to avoid the SDK's non-streaming timeout guard at the large thinking max_tokens.
        with client.messages.stream(**params) as stream:
            message = stream.get_final_message()
    else:
        message = client.messages.create(**params)
    elapsed = time.monotonic() - start

    try:
        parsed = parse_message(message)
    except LLMResponseError as exc:
        exc.elapsed = elapsed  # attach the wall-clock the batch path can't measure
        raise
    return parsed, message.usage, elapsed
