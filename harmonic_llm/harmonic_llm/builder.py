"""
Model construction entry point.

``build_model`` is the one function most users call: it takes a
:class:`ModelConfig` (or nothing, for the default) and returns a ready
``Transformer``, with deterministic seeding and optional device placement.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

from harmonic_llm.config import ModelConfig

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducible construction/runs."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: Optional[Union[ModelConfig, str]] = None,
                device: Optional[str] = None):
    """
    Build a Transformer from a config.

    Args:
        config: a ``ModelConfig``, a path to a YAML config, or None for
            ``ModelConfig.small()``.
        device: optional torch device string ('cuda', 'cpu', ...). If None the
            model stays on its construction device.

    Returns:
        A ready ``Transformer``.
    """
    import torch
    from harmonic_llm._core import Transformer

    if config is None:
        config = ModelConfig.small()
    elif isinstance(config, str):
        config = ModelConfig.from_yaml(config)
    elif not isinstance(config, ModelConfig):
        raise TypeError(f"config must be ModelConfig | str | None, got {type(config)}")

    set_seed(config.seed)
    args = config.to_model_args()

    logger.info("Building Transformer: dim=%d layers=%d heads=%d vocab=%d",
                config.dim, config.n_layers, config.n_heads, config.vocab_size)
    model = Transformer(args)

    if device is not None:
        model = model.to(device)

    n_params = count_parameters(model)
    logger.info("Model built: %s total parameters (%.1fM)", f"{n_params:,}", n_params / 1e6)
    return model


def count_parameters(model, trainable_only: bool = False) -> int:
    """Total (or trainable) parameter count."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def parameter_summary(model) -> dict:
    """A per-top-level-module parameter breakdown, for logging/inspection."""
    summary = {}
    for name, module in model.named_children():
        summary[name] = sum(p.numel() for p in module.parameters())
    summary["_total"] = count_parameters(model)
    summary["_trainable"] = count_parameters(model, trainable_only=True)
    return summary
