"""Which hook the runner captures at, when the SAE declares two names for it.

``hook_name`` is a TransformerLens convention; ``hf_hook_name`` is the module path resolved from the
artifact's own config. For some releases they denote different tensors, so which one is authoritative
is a correctness question rather than a formatting one.
"""

from __future__ import annotations

import pytest

from sae_dashboard.neuronpedia.neuronpedia_runner import resolve_capture_hook_name

TL_NAME = "blocks.0.hook_mlp_in"
HF_NAME = "model.layers.0.pre_feedforward_layernorm.output"


class _Metadata:
    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


@pytest.fixture(params=["attribute", "mapping"], ids=["attr-metadata", "dict-metadata"])
def metadata_factory(request):
    """SAE metadata is reached both ways in this codebase; the resolver must handle both."""
    return _Metadata if request.param == "attribute" else (lambda **fields: dict(fields))


def test_transformer_lens_path_keeps_the_tl_name(metadata_factory):
    meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
    assert resolve_capture_hook_name(meta, use_huggingface=False) == TL_NAME


def test_huggingface_path_prefers_the_declared_module_path(metadata_factory):
    meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
    assert resolve_capture_hook_name(meta, use_huggingface=True) == HF_NAME


def test_falls_back_when_the_loader_publishes_no_module_path(metadata_factory):
    """Only two of SAELens' loaders populate the field, so the fallback is the common case."""
    meta = metadata_factory(hook_name="blocks.0.hook_resid_post")
    assert resolve_capture_hook_name(meta, use_huggingface=True) == "blocks.0.hook_resid_post"


@pytest.mark.parametrize("empty", ["", None], ids=["empty-string", "none"])
def test_an_unpopulated_field_is_not_a_module_path(metadata_factory, empty):
    meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=empty)
    assert resolve_capture_hook_name(meta, use_huggingface=True) == TL_NAME


def test_switching_capture_location_is_announced(metadata_factory, capsys):
    """Silently changing where activations come from is the failure mode this whole area has."""
    meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
    resolve_capture_hook_name(meta, use_huggingface=True)
    out = capsys.readouterr().out
    assert HF_NAME in out and TL_NAME in out


def test_no_announcement_when_the_names_agree(metadata_factory, capsys):
    meta = metadata_factory(hook_name=HF_NAME, hf_hook_name=HF_NAME)
    resolve_capture_hook_name(meta, use_huggingface=True)
    assert capsys.readouterr().out == ""


class TestExplicitCaptureOverride:
    """`capture_hook_name` names the capture location outright.

    It exists because some SAEs declare a TransformerLens name for a tensor other than the one they
    were trained on, and on the TransformerBridge the correct name (`blocks.{i}.ln2.hook_out`) is one
    no SAE metadata field carries.
    """

    BRIDGE_NAME = "blocks.0.ln2.hook_out"

    def test_override_wins_over_the_declared_name(self, metadata_factory):
        meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
        assert (
            resolve_capture_hook_name(meta, use_huggingface=False, capture_hook_name=self.BRIDGE_NAME)
            == self.BRIDGE_NAME
        )

    def test_override_wins_on_the_huggingface_path_too(self, metadata_factory):
        """Otherwise the two mechanisms would disagree depending on an unrelated flag."""
        meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
        assert (
            resolve_capture_hook_name(meta, use_huggingface=True, capture_hook_name=self.BRIDGE_NAME)
            == self.BRIDGE_NAME
        )

    def test_unset_override_changes_nothing(self, metadata_factory):
        meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
        assert resolve_capture_hook_name(meta, use_huggingface=False, capture_hook_name=None) == TL_NAME

    def test_override_is_announced_with_both_names(self, metadata_factory, capsys):
        meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
        resolve_capture_hook_name(meta, use_huggingface=False, capture_hook_name=self.BRIDGE_NAME)
        out = capsys.readouterr().out
        assert self.BRIDGE_NAME in out and TL_NAME in out
        # The label vocabulary cannot express every capture location, so the run log has to say that
        # the source record's `hook_point` and the capture location may read differently.
        assert "label" in out

    def test_no_announcement_when_the_override_matches(self, metadata_factory, capsys):
        meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
        resolve_capture_hook_name(meta, use_huggingface=False, capture_hook_name=TL_NAME)
        assert capsys.readouterr().out == ""
