"""Tests for the Hugging Face model export.

``modeling.py`` is a second copy of the architecture and the label encodings, living in a different
repository under a different license. The whole point of generating it is that it cannot drift from
the package; these tests are what makes that true rather than merely intended.
"""

import pytest
import torch
from safetensors.torch import load_file

from chess_commentator.cnn.model import SquareClassifierMultiHead
from chess_commentator.dataset.publish_model import (
    SOURCE_FILES,
    build_card,
    build_modeling,
    strip_main_block,
    strip_module_preamble,
    verify_generated,
)
from chess_commentator.labels import TARGET_MAP, TOP_LEFT_OHE_MAP, reconstruct_13way_logprobs


@pytest.fixture(scope="module")
def modeling_namespace():
    """The generated module, executed in isolation."""
    namespace: dict = {}
    exec(compile(build_modeling(), "modeling.py", "exec"), namespace)
    return namespace


def test_sources_do_not_import_python_chess():
    """The Apache-2.0 relicensing rests entirely on this: neither source file links python-chess,
    so the repository's GPL obligation does not reach them."""
    for path in SOURCE_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import chess ")
            assert not stripped == "import chess"
            assert not stripped.startswith("from chess ")
            assert not stripped.startswith("from chess.")


def test_generated_module_imports_nothing_from_the_package(modeling_namespace):
    source = build_modeling()
    assert "chess_commentator" not in source.replace(
        "https://github.com/felixfabricius/robot-chess-commentator", ""
    ).replace("`chess_commentator.dataset.publish_model`", "")
    assert "Apache License, Version 2.0" in source


def test_strip_main_block_removes_the_cli_harness():
    source = 'X = 1\n\n\nif __name__ == "__main__":\n    print("hi")\n'
    assert strip_main_block(source) == "X = 1\n"
    assert strip_main_block("X = 1\n") == "X = 1\n"


def test_strip_module_preamble_keeps_definitions():
    stripped = strip_module_preamble('"""Doc.\n\nMore.\n"""\n\nimport torch\n\nX = 1\n')
    assert stripped == "X = 1\n"


def test_generated_encodings_match_the_package(modeling_namespace):
    assert modeling_namespace["TARGET_MAP"] == TARGET_MAP
    assert modeling_namespace["TOP_LEFT_OHE_MAP"] == TOP_LEFT_OHE_MAP


def test_generated_model_matches_the_package(modeling_namespace):
    generated = modeling_namespace["SquareClassifierMultiHead"]()
    reference = SquareClassifierMultiHead()
    assert list(generated.state_dict()) == list(reference.state_dict())
    for key, tensor in generated.state_dict().items():
        assert tensor.shape == reference.state_dict()[key].shape


def test_generated_model_computes_the_same_logprobs(modeling_namespace):
    """Same weights through both copies must give the same answer, or the published model is not
    the model documented by the card."""
    reference = SquareClassifierMultiHead()
    generated = modeling_namespace["SquareClassifierMultiHead"]()
    generated.load_state_dict(reference.state_dict())
    reference.eval()
    generated.eval()

    torch.manual_seed(0)
    image = torch.rand(2, 4, 144, 144)
    metadata = torch.zeros(2, 4)
    metadata[:, TOP_LEFT_OHE_MAP["h1"]] = 1

    with torch.no_grad():
        expected = reconstruct_13way_logprobs(*reference(image, metadata))
        actual = modeling_namespace["reconstruct_13way_logprobs"](*generated(image, metadata))
    torch.testing.assert_close(actual, expected)


def test_verify_generated_rejects_a_drifted_copy():
    """The drift guard has to actually fire, or it is decoration."""
    drifted = build_modeling().replace(
        'enumerate([\n    "empty",', 'enumerate([\n    "EMPTY",', 1
    )
    with pytest.raises(AssertionError, match="TARGET_MAP"):
        verify_generated(drifted)


def test_card_leads_with_move_accuracy():
    """Move accuracy is the result; per-square accuracy is a diagnostic. The card must not invert
    that, because 75% badly understates a system that reads 96% of moves correctly."""
    card = build_card()
    assert card.index("Move accuracy") < card.index("Per-square accuracy")
    assert "94.7%" in card and "94.6%" in card
    # The shipped checkpoint is only valid with the correction on; the card must say so.
    assert "prior correction is on" in card.lower()


def test_bundled_weights_load_into_the_generated_model(modeling_namespace):
    """The checkpoint actually shipped in weights/ must load cleanly -- including its log_prior."""
    from chess_commentator.dataset.publish_model import REPO_ROOT

    weights = REPO_ROOT / "weights" / "model_state_dict.safetensors"
    if not weights.exists():
        pytest.skip("weights/model_state_dict.safetensors not present")

    model = modeling_namespace["SquareClassifierMultiHead"]()
    missing, unexpected = model.load_state_dict(load_file(weights, device="cpu"), strict=False)
    assert not missing and not unexpected
    assert "log_prior" in model.state_dict()
    assert not torch.allclose(model.log_prior, torch.zeros(13)), (
        "bundled weights carry an all-zero log_prior, so prior_correction would be a silent no-op"
    )
