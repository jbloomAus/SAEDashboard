import pytest
from transformers import AutoTokenizer

from sae_dashboard.utils_fns import decode_byte_level_token, is_byte_level_tokenizer


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("èĻ½çĦ¶", "虽然"),
        ("ãģ®ãģŁãĤģãģ«", "のために"),
        ("ĠtarafÄ±ndan", " tarafından"),
        ("ðŁĺī", "😉"),
        ("âĢĻ", "’"),
        ("Ċ", "\n"),
        ("ĉ", "\t"),
        ("Ġsocial", " social"),
        ("<|im_start|>", "<|im_start|>"),
    ],
)
def test_decode_byte_level_token(raw: str, expected: str):
    assert decode_byte_level_token(raw) == expected


def test_split_characters_become_replacement_chars():
    # one full character, then the first byte of the next
    assert decode_byte_level_token("ä¸Ģä") == "一\ufffd"
    assert decode_byte_level_token("ª½") == "\ufffd\ufffd"


def test_tokens_that_are_not_byte_level_are_unchanged():
    assert decode_byte_level_token("already 虽然") == "already 虽然"
    # byte 0 would be NUL, which Postgres text cannot hold
    assert decode_byte_level_token("Ā") == "Ā"


@pytest.mark.parametrize("model_id", ["openai-community/gpt2", "Qwen/Qwen3.5-2B-Base"])
def test_byte_level_tokenizers_are_detected(model_id: str):
    assert is_byte_level_tokenizer(AutoTokenizer.from_pretrained(model_id))


def test_sentencepiece_tokenizer_is_not_byte_level():
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    except OSError as e:
        pytest.skip(f"gated model not available: {e}")
    assert not is_byte_level_tokenizer(tokenizer)


def test_decoded_vocab_matches_tokenizer_decode():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
    vocab = tokenizer.get_vocab()
    for raw in ["èĻ½çĦ¶", "ä¿±ä¹Ĳéĥ¨", "Ġsocial", "ðŁĺī"]:
        assert decode_byte_level_token(raw) == tokenizer.decode([vocab[raw]])
