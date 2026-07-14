"""
Tests for the autoregressive generation loop (Transformer.generate),
the sampling helper, and the tokenizer-level generate_text glue.
"""
import warnings

import pytest
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

import harmonic_llm as hl
from harmonic_llm._core import Transformer
from harmonic_llm.training import ByteTokenizer, generate_text


@pytest.fixture(scope="module")
def tiny_model():
    return hl.build_model(hl.ModelConfig.tiny())


@pytest.fixture
def prompt_ids():
    return torch.randint(0, 256, (1, 8))


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------
class TestSampling:
    def test_temperature_zero_is_argmax(self):
        logits = torch.tensor([[0.1, 5.0, 0.3, -2.0]])
        tok = Transformer._sample_next_token(logits, temperature=0.0)
        assert tok.shape == (1, 1)
        assert tok.item() == 1                       # index of the max logit

    def test_top_k_one_is_deterministic(self):
        logits = torch.randn(4, 50)
        tok = Transformer._sample_next_token(logits, temperature=1.0, top_k=1)
        assert torch.equal(tok.squeeze(-1), logits.argmax(dim=-1))

    def test_output_shape_and_range(self):
        logits = torch.randn(3, 20)
        tok = Transformer._sample_next_token(logits, temperature=0.8, top_p=0.9)
        assert tok.shape == (3, 1)
        assert (tok >= 0).all() and (tok < 20).all()


# ---------------------------------------------------------------------------
# Transformer.generate
# ---------------------------------------------------------------------------
class TestGenerate:
    def test_length_grows_by_max_new_tokens(self, tiny_model, prompt_ids):
        out = tiny_model.generate(prompt_ids, max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, prompt_ids.size(1) + 5)
        # prompt is preserved as a prefix
        assert torch.equal(out[:, : prompt_ids.size(1)], prompt_ids)

    def test_greedy_is_deterministic(self, tiny_model, prompt_ids):
        a = tiny_model.generate(prompt_ids, max_new_tokens=6, temperature=0.0)
        b = tiny_model.generate(prompt_ids, max_new_tokens=6, temperature=0.0)
        assert torch.equal(a, b)

    def test_eos_stops_early(self, tiny_model, prompt_ids):
        # Greedily peek the first token, then make it the EOS id: generation must
        # stop right after emitting it (prompt_len + 1).
        first = tiny_model.generate(prompt_ids, max_new_tokens=1, temperature=0.0)
        first_tok = int(first[0, -1].item())
        out = tiny_model.generate(prompt_ids, max_new_tokens=20, temperature=0.0,
                                  eos_token_id=first_tok)
        assert out.shape[1] == prompt_ids.size(1) + 1

    def test_generate_restores_training_mode(self, tiny_model, prompt_ids):
        tiny_model.train()
        tiny_model.generate(prompt_ids, max_new_tokens=2, temperature=0.0)
        assert tiny_model.training is True

    def test_respects_context_window(self, tiny_model):
        # A prompt longer than max_seq_len must not crash; the loop keeps the
        # most recent window.
        long_prompt = torch.randint(0, 256, (1, hl.ModelConfig.tiny().max_seq_len + 10))
        out = tiny_model.generate(long_prompt, max_new_tokens=3, temperature=0.0)
        assert out.shape[1] == long_prompt.size(1) + 3


# ---------------------------------------------------------------------------
# Text-level glue
# ---------------------------------------------------------------------------
class TestGenerateText:
    def test_returns_decodable_string(self, tiny_model):
        tok = ByteTokenizer()
        text = generate_text(tiny_model, tok, "hi", max_new_tokens=8, temperature=0.0)
        assert isinstance(text, str)

    def test_solve_problem_fallback_uses_generate(self, tiny_model, prompt_ids):
        # solve_problem's fallback path calls self.generate; it must now resolve.
        out, rationale = tiny_model.solve_problem(prompt_ids, max_steps=3)
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] >= prompt_ids.size(1)
