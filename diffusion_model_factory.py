# diffusion_model_factory.py
from __future__ import annotations

from typing import Any, Dict, Optional
import importlib


# Default conventions (module + class names)
_ARCH_REGISTRY = {
    "mlp": ("diffusion_mlp_arch", "TrajectoryMLPDenoiser"),
    "cnn": ("diffusion_cnn_arch", "TrajectoryCNNDenoiser"),
    "transformer": ("diffusion_transformer_arch", "TrajectoryTransformerDenoiser"),
}


def build_denoiser(
    *,
    arch: str,
    horizon: int,
    traj_dim: int,
    cond_dim: int,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    # common hyperparams (ignored by models that don't use them)
    hidden_dim: int = 1024,
    depth: int = 4,
    time_emb_dim: int = 128,
    dropout: float = 0.0,
    # optional override if you want exact control
    module_name: Optional[str] = None,
    class_name: Optional[str] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Build a diffusion denoiser model (architecture-only).

    Requirement for ALL architectures:
      forward(sample, timestep, cond, return_dict=True) -> output with `.sample` (epsilon prediction)
      OR a tuple/tensor where first item is epsilon.

    By default it looks for:
      - MLP:         diffusion_mlp_arch.TrajectoryMLPDenoiser
      - CNN:         diffusion_cnn_arch.TrajectoryCNNDenoiser
      - Transformer: diffusion_transformer_arch.TrajectoryTransformerDenoiser
    """
    arch = arch.lower().strip()
    if module_name is None or class_name is None:
        if arch not in _ARCH_REGISTRY:
            raise ValueError(f"Unknown arch='{arch}'. Options: {list(_ARCH_REGISTRY.keys())}")
        module_name, class_name = _ARCH_REGISTRY[arch]

    kwargs: Dict[str, Any] = dict(
        horizon=horizon,
        traj_dim=traj_dim,
        cond_dim=cond_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        time_emb_dim=time_emb_dim,
        dropout=dropout,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(
            f"Could not import module '{module_name}' for arch='{arch}'. "
            f"Did you create the file {module_name}.py?\nOriginal error: {e}"
        )

    try:
        cls = getattr(mod, class_name)
    except AttributeError as e:
        raise ImportError(
            f"Module '{module_name}' does not define class '{class_name}' for arch='{arch}'.\n"
            f"Original error: {e}"
        )

    return cls(**kwargs)
