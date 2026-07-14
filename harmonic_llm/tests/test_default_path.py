"""
Behavioural tests for the core building blocks on the default forward path:
RMSNorm, ParallelEmbedding, ParallelHead, and MoEAttention (the verified default
attention). These pin shape, finiteness, gradient flow and -- for the attention
expert -- causality, so regressions in the hot path surface immediately.

(HarmonicFlow FFN, the Sinkhorn router's balance/no-drop guarantees, ZeroMass and
full-sequence logits are covered in test_model.py; this file fills the gaps.)
"""
import warnings

import pytest
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

from harmonic_llm._core import (
    RMSNorm, ParallelEmbedding, ParallelHead, MoEAttention,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class TestRMSNorm:
    def test_shape_preserved_and_finite(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        y = norm(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_normalises_to_unit_rms(self):
        norm = RMSNorm(128)                       # affine weight is ones at init
        x = torch.randn(4, 10, 128) * 7.0         # arbitrary scale
        y = norm(x).float()
        rms = y.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.05)

    def test_gradients_flow(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 4, 32, requires_grad=True)
        norm(x).pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# ParallelEmbedding
# ---------------------------------------------------------------------------
class TestParallelEmbedding:
    def test_shape_and_finite(self):
        emb = ParallelEmbedding(256, 64)
        ids = torch.randint(0, 256, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()          # not NaN (init bugfix)

    def test_distinct_tokens_distinct_rows(self):
        emb = ParallelEmbedding(256, 64)
        out = emb(torch.tensor([[0, 1]]))
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# ParallelHead
# ---------------------------------------------------------------------------
class TestParallelHead:
    def test_all_positions_vs_last(self):
        head = ParallelHead(256, 64)
        x = torch.randn(2, 8, 64)
        full = head.get_logits(x, all_positions=True)
        last = head.get_logits(x, all_positions=False)
        assert full.shape == (2, 8, 256)
        assert last.shape == (2, 256)
        assert torch.allclose(full[:, -1], last, atol=1e-4)

    def test_finite(self):
        head = ParallelHead(256, 64)
        assert torch.isfinite(head.get_logits(torch.randn(1, 4, 64))).all()


# ---------------------------------------------------------------------------
# MoEAttention (default attention path)
# ---------------------------------------------------------------------------
class TestMoEAttention:
    def test_shape_and_finite(self):
        attn = MoEAttention(dim=64, num_experts=4, num_heads=8, topk=2, causal=True)
        x = torch.randn(2, 10, 64)
        y = attn(x, start_pos=0)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_gradients_flow(self):
        attn = MoEAttention(dim=32, num_experts=2, num_heads=4, topk=1, causal=True)
        x = torch.randn(1, 6, 32, requires_grad=True)
        attn(x, start_pos=0).pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_single_expert_is_causal(self):
        # With one expert the routing weight is uniform, so the block reduces to a
        # single causal MHA expert: perturbing the last token must not change any
        # earlier position's output.
        torch.manual_seed(0)
        attn = MoEAttention(dim=32, num_experts=1, num_heads=4, topk=1, causal=True)
        attn.eval()
        x = torch.randn(1, 8, 32)
        with torch.no_grad():
            y1 = attn(x, start_pos=0)
            x2 = x.clone()
            x2[0, -1] += 5.0                        # change only the last token
            y2 = attn(x2, start_pos=0)
        assert torch.allclose(y1[0, :-1], y2[0, :-1], atol=1e-5)
