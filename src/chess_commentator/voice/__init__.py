"""Claude-on-text commentary, Kokoro TTS, playback, and the two clip-cache CLIs.

The split against `vlm/` is by input modality, not by vendor: this package sends *text* to Claude
(a one-liner about the move just played, the closing roast), `vlm/` sends *images*.

Import-free like the rest: `speaker` imports kokoro and `outro` reaches for the robot SDK, while
`clips` deliberately does neither -- it takes synthesis as an injected callable so it stays
testable without either dependency.
"""
