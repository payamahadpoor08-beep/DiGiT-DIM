"""
Tests for multi-head latent attention (MLA).

MLA previously broke at most head/dim ratios (a mislabelled attention einsum, a
6-D expand and a wrong reduction axis in sparse_attn, an uninitialised sink, a
bf16/float mismatch, and heavy default-on sub-modules applied to the 4-D
attention output). These pin the fixed behaviour across a ratio grid and keep MoE
as the verified default.
"""
import warnings

import pytest
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

import harmonic_llm as hl
from harmonic_llm._core import MultiHeadLatentAttention


def _mla_config(n_heads, head_dim):
    return hl.ModelConfig(
        dim=n_heads * head_dim, n_layers=1, n_heads=n_heads, head_dim=head_dim,
        vocab_size=256, max_seq_len=64, max_batch_size=2, n_flows=4, window_size=16,
        use_latent_attention=True, use_moe_attention=False,
    )


RATIO_GRID = [
    (4, 32), (4, 64), (8, 32), (8, 64), (6, 48),
    (2, 64), (16, 64), (3, 48), (5, 40), (12, 32),
]


class TestMLARatioSweep:
    @pytest.mark.parametrize("n_heads,head_dim", RATIO_GRID)
    def test_forward_finite_and_shaped(self, n_heads, head_dim):
        model = hl.build_model(_mla_config(n_heads, head_dim))
        ids = torch.randint(0, 256, (1, 16))
        out = model(ids, start_pos=0)
        logits = out[0] if isinstance(out, tuple) else out
        assert logits.shape == (1, 16, 256)
        assert torch.isfinite(logits).all()

    def test_uses_latent_attention_module(self):
        model = hl.build_model(_mla_config(4, 32))
        assert isinstance(model.layers[0].attn, MultiHeadLatentAttention)

    def test_gradients_flow(self):
        model = hl.build_model(_mla_config(4, 32))
        ids = torch.randint(0, 256, (1, 16))
        logits = model(ids, start_pos=0)
        logits = logits[0] if isinstance(logits, tuple) else logits
        logits.float().pow(2).mean().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and any(torch.isfinite(g).any() for g in grads)


class TestMLAParityWithMoE:
    def test_same_output_shape_as_moe(self):
        ids = torch.randint(0, 256, (1, 16))
        mla = hl.build_model(_mla_config(4, 32))
        moe_cfg = _mla_config(4, 32)
        moe_cfg.use_latent_attention = False
        moe_cfg.use_moe_attention = True
        moe = hl.build_model(moe_cfg)
        lm = mla(ids, start_pos=0); lm = lm[0] if isinstance(lm, tuple) else lm
        lo = moe(ids, start_pos=0); lo = lo[0] if isinstance(lo, tuple) else lo
        assert lm.shape == lo.shape

    def test_generation_works_with_mla(self):
        model = hl.build_model(_mla_config(4, 32))
        ids = torch.randint(0, 256, (1, 8))
        out = model.generate(ids, max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 13)
        assert torch.isfinite(model(out[:, :16], start_pos=0).float()).all()


class TestMLAConstructionInvariants:
    def test_rejects_non_divisible_groups(self):
        cfg = _mla_config(4, 32)
        args = cfg.to_model_args()
        args.o_groups = 3               # 4 not divisible by 3
        with pytest.raises(ValueError, match="divisible by o_groups"):
            MultiHeadLatentAttention(0, args)

    def test_rejects_rope_larger_than_head(self):
        cfg = _mla_config(4, 32)
        args = cfg.to_model_args()
        args.rope_head_dim = 48         # > head_dim 32
        with pytest.raises(ValueError, match="rope_head_dim"):
            MultiHeadLatentAttention(0, args)

    def test_rejects_odd_rope(self):
        cfg = _mla_config(4, 32)
        args = cfg.to_model_args()
        args.rope_head_dim = 15         # odd
        with pytest.raises(ValueError, match="even"):
            MultiHeadLatentAttention(0, args)
