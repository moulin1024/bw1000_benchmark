"""JAX environment and distributed-process initialization helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

from .config import DType, Platform


_VISIBLE_DEVICE_VARIABLES = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


def configure_jax_environment(
    *,
    platform: Platform | None,
    dtype: DType,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Set options that must be present before JAX initializes a backend."""
    target = os.environ if environ is None else environ
    if platform:
        target["JAX_PLATFORMS"] = platform
    if dtype == "float64":
        target["JAX_ENABLE_X64"] = "true"
    target.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def local_device_id(environ: Mapping[str, str] | None = None) -> int:
    """Select the process-local JAX device using the Slurm launch contract."""
    source = os.environ if environ is None else environ
    visible = next(
        (source[name] for name in _VISIBLE_DEVICE_VARIABLES if source.get(name)),
        None,
    )
    if visible is not None and len(visible.split(",")) == 1:
        return 0
    return int(source.get("SLURM_LOCALID", "0"))


def initialize_jax_distributed(
    jax_module: Any,
    *,
    enabled: bool,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Initialize JAX distributed once when requested by the caller."""
    if enabled and not jax_module.distributed.is_initialized():
        jax_module.distributed.initialize(local_device_ids=[local_device_id(environ)])
