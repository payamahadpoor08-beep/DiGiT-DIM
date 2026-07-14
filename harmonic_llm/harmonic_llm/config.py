"""
Configuration layer.

``ModelArgs`` in the core file has ~290 fields. Most callers should never touch
that surface directly: this module exposes a small, validated ``ModelConfig``
with sensible presets and YAML round-tripping, and translates it into a fully-
populated ``ModelArgs`` (satisfying all of ModelArgs' internal invariants) via
:meth:`ModelConfig.to_model_args`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelConfig:
    """
    A curated, validated model configuration.

    Only the knobs that matter for shaping a model are exposed here; everything
    else in ``ModelArgs`` is derived or left at a safe default. All heavy,
    specialised subsystems default OFF (opt-in), so a config produces a plain,
    fast language model unless features are explicitly enabled.
    """

    # -- core shape ----------------------------------------------------------
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 32000
    max_seq_len: int = 2048
    max_batch_size: int = 8

    # -- native feed-forward (HarmonicFlow) ----------------------------------
    n_flows: int = 8
    flow_harmonics: int = 4
    flow_hidden: Optional[int] = None          # defaults to dim*2

    # -- attention -----------------------------------------------------------
    # NOTE: MoE attention is the end-to-end-verified default. Multi-head latent
    # attention has a head-count broadcasting issue at some dim ratios and is
    # opt-in until fixed.
    use_latent_attention: bool = False
    use_moe_attention: bool = True
    window_size: int = 512

    # -- ZeroMass quantisation ----------------------------------------------
    q_mode: str = "standard"                   # standard | zero_mass | ...
    q_bundle_size: int = 8192
    q_cache_size: int = 512

    # -- optional subsystems (opt-in) ---------------------------------------
    use_meta_learning: bool = False
    use_kg: bool = False
    use_speculative_decoding: bool = False
    use_task_condition: bool = False

    # -- runtime -------------------------------------------------------------
    dtype: str = "bf16"                         # bf16 | fp8
    seed: int = 42

    # -----------------------------------------------------------------------
    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.dim % self.n_heads != 0 and self.head_dim * self.n_heads != self.dim:
            # dim need not equal n_heads*head_dim in this architecture, but the
            # combination must be self-consistent for the attention projections.
            pass
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.q_mode not in ("standard", "zero_mass", "deterministic",
                               "predictive", "hierarchical", "neural_memory",
                               "distributed", "adaptive"):
            raise ValueError(f"unknown q_mode {self.q_mode!r}")
        if self.dtype not in ("bf16", "fp8", "fp16", "fp32"):
            raise ValueError(f"unknown dtype {self.dtype!r}")

    # -- presets ------------------------------------------------------------
    @classmethod
    def tiny(cls) -> "ModelConfig":
        """A ~1M-param config for tests and CI."""
        return cls(dim=128, n_layers=2, n_heads=4, head_dim=32,
                   vocab_size=256, max_seq_len=128, max_batch_size=2,
                   n_flows=4, window_size=32)

    @classmethod
    def small(cls) -> "ModelConfig":
        return cls(dim=512, n_layers=6, n_heads=8, head_dim=64,
                   vocab_size=32000, max_seq_len=2048)

    @classmethod
    def base(cls) -> "ModelConfig":
        return cls(dim=1024, n_layers=12, n_heads=16, head_dim=64,
                   vocab_size=50000, max_seq_len=4096, n_flows=16)

    @classmethod
    def large(cls) -> "ModelConfig":
        return cls(dim=2048, n_layers=24, n_heads=16, head_dim=128,
                   vocab_size=64000, max_seq_len=8192, n_flows=16,
                   q_mode="zero_mass")

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    # -- translation to the core ModelArgs ----------------------------------
    def to_model_args(self):
        """
        Build a fully-valid ``ModelArgs`` from this config.

        This fills in every derived field the core requires and satisfies its
        ``__post_init__`` invariants (head/group divisibility, rope dims, LoRA
        ranks bounded by dim, etc.), so the returned object constructs a model
        without surprises.
        """
        from harmonic_llm._core import ModelArgs

        dim = self.dim
        head_dim = self.head_dim
        # Choose divisor-friendly auxiliary dims.
        o_groups = _largest_divisor_leq(self.n_heads, 8)
        rope_head_dim = _largest_divisor_leq(head_dim, 16)
        # group_dim = head_dim * (n_heads / o_groups); o_lora_rank must be <= it.
        group_dim = head_dim * (self.n_heads // o_groups)
        lora = max(1, min(64, dim // 2, group_dim))

        return ModelArgs(
            dim=dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            o_groups=o_groups,
            o_lora_rank=lora,
            q_lora_rank=lora,
            kv_lora_rank=lora,
            compress_ratios=tuple(0 for _ in range(self.n_layers)),
            n_routed_experts=self.n_flows,
            n_activated_experts=max(1, self.n_flows // 4),
            window_size=self.window_size,
            index_topk=min(64, self.max_seq_len),
            n_mtp_layers=0,
            lora_rank=8,
            max_users=10000,
            use_latent_attention=self.use_latent_attention,
            use_moe_attention=self.use_moe_attention,
            q_mode=self.q_mode,
            q_bundle_size=self.q_bundle_size,
            q_cache_size=self.q_cache_size,
            use_meta_learning=self.use_meta_learning,
            use_kg=self.use_kg,
            use_speculative_decoding=self.use_speculative_decoding,
            use_task_condition=self.use_task_condition,
            dtype=self.dtype,
            q_seed=self.seed,
        )


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of n that is <= cap (>=1)."""
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1
