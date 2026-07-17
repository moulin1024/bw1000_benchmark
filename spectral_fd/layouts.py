"""Pure shape and slab-decomposition helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DataLayout, PoissonMethod


@dataclass(frozen=True, slots=True)
class SlabDecomposition:
    """Global and per-device sizes for the current z-slab decomposition."""

    nx: int
    ny: int
    nz: int
    global_devices: int
    local_devices: int
    method: PoissonMethod

    def validate(self) -> None:
        if self.global_devices <= 0 or self.local_devices <= 0:
            raise ValueError("global_devices and local_devices must be positive")
        if self.local_devices > self.global_devices:
            raise ValueError("local_devices cannot exceed global_devices")
        if self.ny % self.global_devices:
            raise ValueError(
                f"ny={self.ny} must be divisible by GPU count {self.global_devices}"
            )
        if self.nz % self.global_devices:
            raise ValueError(
                f"nz={self.nz} must be divisible by GPU count {self.global_devices}"
            )
        if self.method != "transpose" and self.nz_local < 2:
            raise ValueError(f"SPIKE methods need nz/GPUs >= 2; got {self.nz_local}")

    @property
    def nxh(self) -> int:
        return self.nx // 2 + 1

    @property
    def ny_local(self) -> int:
        return self.ny // self.global_devices

    @property
    def nz_local(self) -> int:
        return self.nz // self.global_devices

    def local_physical_shape(self, layout: DataLayout) -> tuple[int, ...]:
        if layout == "z-first":
            return (
                self.local_devices,
                self.nz_local,
                self.ny,
                self.nx,
            )
        if layout == "xyz":
            return (
                self.local_devices,
                self.nx,
                self.ny,
                self.nz_local,
            )
        raise ValueError(f"unsupported data layout: {layout!r}")

    def global_physical_shape(self, layout: DataLayout) -> tuple[int, int, int]:
        if layout == "z-first":
            return (self.nz, self.ny, self.nx)
        if layout == "xyz":
            return (self.nx, self.ny, self.nz)
        raise ValueError(f"unsupported data layout: {layout!r}")
