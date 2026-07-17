"""Named, measured platform presets for the Poisson solver."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Poisson3DConfig


@dataclass(frozen=True, slots=True)
class Poisson3DPreset:
    """A solver configuration plus benchmark-launch defaults."""

    name: str
    description: str
    config_items: tuple[tuple[str, object], ...]
    benchmark_items: tuple[tuple[str, object], ...]

    @property
    def config_overrides(self) -> dict[str, object]:
        return dict(self.config_items)

    @property
    def benchmark_defaults(self) -> dict[str, object]:
        return dict(self.benchmark_items)

    def create_config(self, **overrides) -> Poisson3DConfig:
        values = self.config_overrides
        values.update(overrides)
        config = Poisson3DConfig(**values)
        config.validate()
        return config


MN5_CUDA = Poisson3DPreset(
    name="mn5-cuda",
    description=(
        "MareNostrum 5 single-node CUDA preset measured on four GPUs; "
        "row-at-a-time Thomas is faster than static chunking."
    ),
    config_items=(
        ("nx", 1024),
        ("ny", 1024),
        ("nz", 1024),
        ("dtype", "float64"),
        ("method", "spike"),
        ("tridiag", "thomas"),
        ("thomas_chunk", 1),
        ("spike_interface_collective", "allgather"),
        ("spike_interface_solver", "selected-rows"),
        ("pipeline_execution", "staged"),
        ("data_layout", "z-first"),
        ("platform", "cuda"),
    ),
    benchmark_items=(
        ("gpu_counts", "4"),
        ("task_mode", "single"),
        ("warmup", 5),
        ("samples", 20),
        ("iterations", 10),
    ),
)

DCU_ROCM = Poisson3DPreset(
    name="dcu-rocm",
    description=(
        "Eight-DCU ROCm preset; Thomas chunk 16 and a single process driving "
        "all devices are the measured fast path."
    ),
    config_items=(
        ("nx", 1024),
        ("ny", 1024),
        ("nz", 1024),
        ("dtype", "float64"),
        ("method", "spike"),
        ("tridiag", "thomas"),
        ("thomas_chunk", 16),
        ("spike_interface_collective", "allgather"),
        ("spike_interface_solver", "selected-rows"),
        ("pipeline_execution", "staged"),
        ("data_layout", "z-first"),
        ("platform", "rocm"),
    ),
    benchmark_items=(
        ("gpu_counts", "8"),
        ("task_mode", "single"),
        ("warmup", 5),
        ("samples", 20),
        ("iterations", 10),
    ),
)

_PRESETS = {
    MN5_CUDA.name: MN5_CUDA,
    DCU_ROCM.name: DCU_ROCM,
}
_ALIASES = {
    "mn5": "mn5-cuda",
    "mn5_cuda": "mn5-cuda",
    "dcu": "dcu-rocm",
    "dcu_rocm": "dcu-rocm",
    "bw1000": "dcu-rocm",
}


def available_poisson3d_presets() -> tuple[str, ...]:
    return tuple(_PRESETS)


def get_poisson3d_preset(name: str) -> Poisson3DPreset:
    normalized = name.strip().lower()
    normalized = _ALIASES.get(normalized, normalized.replace("_", "-"))
    try:
        return _PRESETS[normalized]
    except KeyError as exc:
        available = ", ".join(available_poisson3d_presets())
        raise ValueError(
            f"unknown Poisson3D preset {name!r}; available presets: {available}"
        ) from exc


def main() -> int:
    for name in available_poisson3d_presets():
        preset = get_poisson3d_preset(name)
        config = preset.create_config()
        benchmark = preset.benchmark_defaults
        print(
            f"{name}: platform={config.platform}, GPUs={benchmark['gpu_counts']}, "
            f"thomas_chunk={config.thomas_chunk}, "
            f"collective={config.spike_interface_collective}, "
            f"pipeline={config.pipeline_execution}, layout={config.data_layout}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
