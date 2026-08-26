"""Claude-on-images: transport, prompts, schemas, and the whole-image reading strategies.

Split from `voice/` by input modality rather than by vendor. This package sends *pictures* of the
board to Claude and asks what is on it; `voice/` sends *text* about a move that has already been
read and asks for a remark about it. They share a provider and nothing else.

Carved out of what used to be one 878-line `vision.py`, where the Claude half and the CNN half
sat in the same file. That fusion meant importing the estimator pulled in anthropic, importing the
prompts pulled in torch, and two external consumers had to reach for eight underscore-prefixed
names to get at request-building. The names are public here because they always were in practice.

    client      one structured-output call: params, transport, parse, the reasoning knob
    prompts     the task text, the json_schemas, and the slot helpers they share
    strategies  the six reading methods, plus the batch-request builders

Import-free like every __init__ in this tree: `strategies` imports `prompts` imports nothing heavy,
and keeping it that way is what lets `perception` depend on `vlm` without a cycle.
"""
