"""
Core test suite for Harmonic-LLM.

Run with: pytest tests/ -v
"""
import warnings

import pytest
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

import harmonic_llm as hl
from harmonic_llm.builder import count_parameters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_model():
    return hl.build_model(hl.ModelConfig.tiny())


@pytest.fixture
def token_batch():
    return torch.randint(0, 256, (1, 16))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_presets_construct(self):
        for preset in (hl.ModelConfig.tiny, hl.ModelConfig.small,
                       hl.ModelConfig.base, hl.ModelConfig.large):
            cfg = preset()
            assert cfg.dim > 0 and cfg.n_layers > 0

    def test_validation_rejects_bad_dim(self):
        with pytest.raises(ValueError):
            hl.ModelConfig(dim=0)

    def test_validation_rejects_bad_qmode(self):
        with pytest.raises(ValueError):
            hl.ModelConfig(q_mode="nonsense")

    def test_yaml_roundtrip(self, tmp_path):
        cfg = hl.ModelConfig.tiny()
        p = tmp_path / "c.yaml"
        cfg.to_yaml(str(p))
        cfg2 = hl.ModelConfig.from_yaml(str(p))
        assert cfg2.to_dict() == cfg.to_dict()

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError):
            hl.ModelConfig.from_dict({"dim": 128, "bogus_key": 1})

    def test_to_model_args_satisfies_invariants(self):
        # Every preset must translate into a constructible ModelArgs.
        args = hl.ModelConfig.tiny().to_model_args()
        assert args.o_lora_rank <= args.head_dim * (args.n_heads // args.o_groups)
        assert args.head_dim % args.rope_head_dim == 0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_builds(self, tiny_model):
        assert count_parameters(tiny_model) > 0

    def test_deterministic_param_count(self):
        a = count_parameters(hl.build_model(hl.ModelConfig.tiny()))
        b = count_parameters(hl.build_model(hl.ModelConfig.tiny()))
        assert a == b

    def test_ffn_is_native_architecture(self, tiny_model):
        # Every block's FFN must be the native HarmonicFlow, not a legacy MoE.
        for layer in tiny_model.layers:
            if hasattr(layer, "ffn"):
                assert type(layer.ffn).__name__ in ("HarmonicFlowMoE", "HarmonicFlowFFN")

    def test_legacy_moe_classes_removed(self):
        for gone in ("MoE", "DeepSeekMoE", "Expert", "Gate", "AdaptiveMoERouter"):
            assert not hasattr(hl._core, gone), f"legacy class {gone} should be removed"


# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------
class TestForwardBackward:
    def test_forward_shape(self, tiny_model, token_batch):
        out = tiny_model(token_batch, start_pos=0)
        logits = out[0] if isinstance(out, tuple) else out
        assert logits.shape[-1] == 256          # vocab

    def test_forward_no_nan(self, tiny_model, token_batch):
        out = tiny_model(token_batch, start_pos=0)
        logits = out[0] if isinstance(out, tuple) else out
        assert not torch.isnan(logits).any()
        assert torch.isfinite(logits).all()

    def test_backward_finite_grad(self, tiny_model, token_batch):
        tiny_model.zero_grad()
        out = tiny_model(token_batch, start_pos=0)
        logits = out[0] if isinstance(out, tuple) else out
        loss = logits.float().mean()
        loss.backward()
        gnorm = sum(p.grad.pow(2).sum() for p in tiny_model.parameters()
                    if p.grad is not None).sqrt()
        assert torch.isfinite(gnorm)

    def test_can_train_a_few_steps(self):
        model = hl.build_model(hl.ModelConfig.tiny())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(3):
            opt.zero_grad()
            out = model(torch.randint(0, 256, (1, 16)), start_pos=0)
            logits = out[0] if isinstance(out, tuple) else out
            loss = logits.float().pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


# ---------------------------------------------------------------------------
# Native architecture components
# ---------------------------------------------------------------------------
class TestHarmonicFlow:
    def test_resonant_flow_bounded_output(self):
        flow = hl.ResonantFlow(64, 128, n_harmonics=4)
        x = torch.randn(2, 8, 64)
        y = flow(x)
        assert y.shape == x.shape
        # tanh-gated correction is bounded
        assert y.abs().max() <= 2.5

    def test_sinkhorn_router_balanced(self):
        router = hl.SinkhornFlowRouter(64, n_flows=8, n_iters=6)
        plan = router(torch.randn(256, 64))
        # each token's assignment is a distribution
        assert torch.allclose(plan.sum(dim=1), torch.ones(256), atol=1e-4)
        # load is balanced across flows (the whole point)
        loads = plan.sum(dim=0)
        assert (loads.std() / loads.mean()) < 0.05

    def test_sinkhorn_router_no_token_dropped(self):
        # Every token must receive nonzero total routing mass.
        router = hl.SinkhornFlowRouter(64, n_flows=8)
        plan = router(torch.randn(100, 64))
        assert (plan.sum(dim=1) > 0).all()

    def test_ffn_residual_and_learns(self):
        ffn = hl.HarmonicFlowFFN(64, n_flows=8)
        x = torch.randn(2, 8, 64)
        opt = torch.optim.Adam(ffn.parameters(), lr=1e-2)
        tgt = torch.randn(2, 8, 64)
        first = None
        for i in range(40):
            opt.zero_grad()
            loss = (ffn(x) - tgt).pow(2).mean()
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
        assert loss.item() < first        # it actually learns


# ---------------------------------------------------------------------------
# ZeroMass
# ---------------------------------------------------------------------------
class TestZeroMass:
    def test_zero_mass_invariant(self):
        import torch.nn as nn
        lin = nn.Linear(64, 96, bias=False)
        z = hl.ZeroMassXOX.from_linear(lin, operation_type="q_proj",
                                       mode="stored", bundle_size=512, cache_size=32)
        # B @ A must be exactly zero at init
        assert (z.B @ z.A).abs().max().item() < 1e-5

    def test_zero_mass_resident_fraction_small(self):
        z = hl.ZeroMassXOX(1024, 1024, operation_type="q_proj", mode="stored",
                           bundle_size=1 << 18, cache_size=2)
        pc = z.param_count()
        # resident adapter should be a small fraction of the base weight
        assert pc["ratio"] < 0.05


# ---------------------------------------------------------------------------
# Two-tier fine-tuning and base eviction
# ---------------------------------------------------------------------------
class TestTwoTierFinetuning:
    def test_base_ft_rank_is_300x_smaller(self):
        z = hl.ZeroMassXOX(512, 512, rank=3000, base_rank_ratio=300,
                           bundle_size=1 << 16, cache_size=4)
        assert z.ft_rank == 3000 // 300
        z.disk.close()

    def test_base_update_touches_only_2pct(self):
        z = hl.ZeroMassXOX(512, 512, rank=600, base_rank_ratio=300,
                           base_update_fraction=0.02,
                           bundle_size=1 << 16, cache_size=4)
        touched = int(z.ft_mask.sum())
        assert touched == int(512 * 0.02)
        z.disk.close()

    def test_both_tiers_receive_gradient(self):
        z = hl.ZeroMassXOX(128, 128, rank=600, base_rank_ratio=300,
                           bundle_size=4096, cache_size=8)
        z(torch.randn(2, 4, 128)).pow(2).mean().backward()
        assert z.A.grad.abs().max() > 0        # adapter tier
        assert z.ft_A.grad.abs().max() > 0     # base tier
        z.disk.close()


class TestBaseEviction:
    def test_finalize_keeps_base_by_default(self):
        import os
        import torch.nn as nn
        z = hl.ZeroMassXOX.from_linear(nn.Linear(64, 64, bias=False),
                                       mode="stored", rank=8,
                                       bundle_size=1024, cache_size=4)
        path = z.disk.path
        report = z.finalize()                  # default: evict=False
        assert report["base_kept"] is True
        assert os.path.exists(path)
        z.disk.close()

    def test_finalize_evict_actually_deletes_base(self):
        import os
        import torch.nn as nn
        z = hl.ZeroMassXOX.from_linear(nn.Linear(64, 64, bias=False),
                                       mode="stored", rank=8,
                                       bundle_size=1024, cache_size=4)
        path = z.disk.path
        assert os.path.exists(path)
        report = z.finalize(evict=True)
        assert report["evicted"] is True
        assert report["freed_bytes"] > 0
        assert not os.path.exists(path)        # the file is really gone

    def test_layer_still_works_after_eviction(self):
        import torch.nn as nn
        z = hl.ZeroMassXOX.from_linear(nn.Linear(64, 64, bias=False),
                                       mode="stored", rank=8,
                                       bundle_size=1024, cache_size=4)
        z.finalize(evict=True)
        out = z(torch.randn(1, 4, 64))         # regenerates base from seed
        assert out.shape == (1, 4, 64)
        assert not torch.isnan(out).any()


class TestFullSequenceLogits:
    def test_head_returns_all_positions(self, tiny_model):
        # The LM objective needs logits at EVERY position, in one forward.
        out = tiny_model(torch.randint(0, 256, (2, 12)), start_pos=0)
        logits = out[0] if isinstance(out, tuple) else out
        assert logits.dim() == 3                   # (B, T, V)
        assert logits.shape[:2] == (2, 12)

    def test_causal_lm_loss_in_one_forward(self, tiny_model):
        import torch.nn.functional as F
        x = torch.randint(0, 256, (2, 12))
        y = torch.randint(0, 256, (2, 12))
        out = tiny_model(x, start_pos=0)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        assert torch.isfinite(loss)
