"""A Reachy Mini that watches two humans play physical chess and commentates on it out loud.

Layout, top of the stack downward:

    main            the game loop -- one iteration is one move
    session         setup directory + putting the robot in its capture pose
    game            board state; ranks every *legal* move against a noisy reading
    player_input    "was there an input?", via the antennas or the keyboard
    board, labels   the two vocabularies everything agrees on: square/piece names, and the
                    13-way label encoding shared by training and inference

    perception/     camera frame -> rectified board -> 64 square crops -> a board reading
    vlm/            Claude-on-images: prompts, schemas, and the whole-image strategies
    voice/          Claude-on-text + Kokoro TTS + playback
    cnn/            the 1.3 MB square classifier and its training loop
    dataset/        offline tools that produce the labelled training data
    benchmark/      scoring the six board-reading methods against each other

Every `__init__.py` in this tree is import-free. The heavy and hardware-bound dependencies
(torch, anthropic, kokoro, reachy_mini) are confined to the modules that actually need them, so
importing one module never drags in another's.
"""

__version__ = "0.1.0"
