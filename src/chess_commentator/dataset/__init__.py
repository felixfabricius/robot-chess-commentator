"""The offline tools that manufacture the labelled square dataset the CNN trains on.

`generate` captures and labels new boards interactively; `regenerate` rebuilds crops from frames
already on disk without re-photographing anything, preserving the labels (which are the one thing
that cannot be recovered from the pixels).
"""
