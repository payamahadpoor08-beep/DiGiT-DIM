"""
Tests for the BPE tokenizer (and shared tokenizer interface with ByteTokenizer).

These are pure-Python: no model/torch is exercised, so they pin the tokenizer's
behaviour on its own.
"""
import warnings

import pytest

warnings.filterwarnings("ignore")

from harmonic_llm.training import ByteTokenizer, BPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog was not amused. the fox ran away quickly. "
) * 20


@pytest.fixture(scope="module")
def bpe():
    return BPETokenizer.train(CORPUS, vocab_size=400)


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------
class TestRoundTrip:
    @pytest.mark.parametrize("text", [
        "hello world",
        "the quick brown fox",
        "",
        "unicode: café — naïve — 日本語 — 🚀",
        "  leading and trailing spaces  ",
        "tabs\tand\nnewlines",
    ])
    def test_exact_roundtrip(self, bpe, text):
        assert bpe.decode(bpe.encode(text, add_special=False)) == text

    def test_roundtrip_strips_specials(self, bpe):
        text = "recover me"
        ids = bpe.encode(text, add_special=True)
        assert ids[0] == bpe.BOS and ids[-1] == bpe.EOS
        assert bpe.decode(ids) == text          # specials dropped on decode


# ---------------------------------------------------------------------------
# Vocab / specials layout
# ---------------------------------------------------------------------------
class TestVocab:
    def test_vocab_size_accounts_for_merges_and_specials(self, bpe):
        assert len(bpe) == 256 + len(bpe.merges) + 3
        assert bpe.VOCAB_SIZE == len(bpe)

    def test_specials_are_distinct_and_top_of_range(self, bpe):
        assert bpe.PAD == bpe.base_size
        assert {bpe.PAD, bpe.BOS, bpe.EOS} == {bpe.base_size,
                                               bpe.base_size + 1,
                                               bpe.base_size + 2}

    def test_all_ids_within_vocab(self, bpe):
        ids = bpe.encode(CORPUS[:200], add_special=True)
        assert all(0 <= i < len(bpe) for i in ids)


# ---------------------------------------------------------------------------
# Compression vs raw bytes
# ---------------------------------------------------------------------------
class TestCompression:
    def test_bpe_shorter_than_bytes(self, bpe):
        text = "the quick brown fox " * 10
        n_bpe = len(bpe.encode(text, add_special=False))
        n_bytes = len(text.encode("utf-8"))
        assert n_bpe < n_bytes

    def test_more_merges_compress_more(self):
        text = "abcabcabc " * 50
        small = BPETokenizer.train(text, vocab_size=270)
        large = BPETokenizer.train(text, vocab_size=400)
        assert len(large.encode(text, add_special=False)) <= \
               len(small.encode(text, add_special=False))


# ---------------------------------------------------------------------------
# Determinism & persistence
# ---------------------------------------------------------------------------
class TestDeterminismAndIO:
    def test_training_is_deterministic(self):
        a = BPETokenizer.train(CORPUS, vocab_size=400)
        b = BPETokenizer.train(CORPUS, vocab_size=400)
        assert a.merges == b.merges

    def test_save_load_roundtrip(self, bpe, tmp_path):
        p = tmp_path / "bpe.json"
        bpe.save(str(p))
        loaded = BPETokenizer.load(str(p))
        assert loaded.merges == bpe.merges
        text = "the lazy dog"
        assert loaded.encode(text) == bpe.encode(text)
        assert loaded.decode(loaded.encode(text, add_special=False)) == text


# ---------------------------------------------------------------------------
# Interface parity with ByteTokenizer
# ---------------------------------------------------------------------------
class TestInterfaceParity:
    @pytest.mark.parametrize("make", [ByteTokenizer, lambda: BPETokenizer.train(CORPUS, 400)])
    def test_shared_interface(self, make):
        tok = make()
        for attr in ("PAD", "BOS", "EOS"):
            assert isinstance(getattr(tok, attr), int)
        ids = tok.encode("hi there", add_special=True)
        assert ids[0] == tok.BOS and ids[-1] == tok.EOS
        assert isinstance(tok.decode(ids), str)
        assert len(tok) > 0
