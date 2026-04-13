"""Tests for the tokenizer utility."""
from chunkops.tokenizer import count_tokens


def test_empty_string():
    assert count_tokens("") == 0


def test_single_word():
    assert count_tokens("hello") >= 1


def test_token_count_grows_with_length():
    short = "hello"
    long = "hello world this is a longer sentence with more tokens"
    assert count_tokens(long) > count_tokens(short)


def test_token_count_is_positive():
    assert count_tokens("The quick brown fox jumps over the lazy dog") > 0


def test_whitespace_only():
    # split() returns [] for whitespace — should return 0
    assert count_tokens("   ") == 0


def test_reasonable_range_for_sentence():
    text = "The transformer architecture replaced recurrence with self-attention."
    tokens = count_tokens(text)
    # Should be roughly 12–15 tokens
    assert 8 <= tokens <= 20
