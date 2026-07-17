from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from spectral_fd import (
    Poisson3DConfig,
    available_poisson3d_presets,
    get_poisson3d_preset,
)
from spectral_fd.layouts import SlabDecomposition


ROOT = Path(__file__).resolve().parents[1]


class Poisson3DPresetTests(unittest.TestCase):
    def test_named_presets_capture_measured_thomas_chunks(self) -> None:
        mn5 = Poisson3DConfig.from_preset("mn5-cuda")
        dcu = Poisson3DConfig.from_preset("dcu-rocm")

        self.assertEqual(available_poisson3d_presets(), ("mn5-cuda", "dcu-rocm"))
        self.assertEqual((mn5.platform, mn5.thomas_chunk), ("cuda", 1))
        self.assertEqual((dcu.platform, dcu.thomas_chunk), ("rocm", 16))
        self.assertEqual(mn5.spike_interface_collective, "allgather")
        self.assertEqual(dcu.spike_interface_collective, "allgather")

    def test_preset_aliases_and_overrides(self) -> None:
        config = Poisson3DConfig.from_preset(
            "bw1000",
            nx=128,
            ny=256,
            nz=64,
            platform="cpu",
        )

        self.assertEqual((config.nx, config.ny, config.nz), (128, 256, 64))
        self.assertEqual(config.platform, "cpu")
        self.assertEqual(config.thomas_chunk, 16)
        self.assertEqual(
            get_poisson3d_preset("mn5").benchmark_defaults["gpu_counts"],
            "4",
        )

    def test_shell_presets_match_python_presets(self) -> None:
        for name in available_poisson3d_presets():
            with self.subTest(name=name):
                command = (
                    "source ./benchmark_presets.sh; "
                    f"apply_poisson_benchmark_preset {name}; "
                    "printf '%s %s %s %s %s %s %s' "
                    "\"${THOMAS_CHUNK}\" \"${SPIKE_INTERFACE_COLLECTIVE}\" "
                    "\"${TASK_MODE}\" \"${GPU_COUNTS}\" \"${WARMUP}\" "
                    "\"${SAMPLES}\" \"${ITERATIONS}\""
                )
                result = subprocess.run(
                    ["bash", "-c", command],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )

                preset = get_poisson3d_preset(name)
                config = preset.create_config()
                benchmark = preset.benchmark_defaults
                self.assertEqual(
                    result.stdout,
                    (
                        f"{config.thomas_chunk} "
                        f"{config.spike_interface_collective} "
                        f"{benchmark['task_mode']} {benchmark['gpu_counts']} "
                        f"{benchmark['warmup']} {benchmark['samples']} "
                        f"{benchmark['iterations']}"
                    ),
                )


class SlabDecompositionTests(unittest.TestCase):
    def test_shapes_for_four_device_z_first_layout(self) -> None:
        decomposition = SlabDecomposition(
            nx=1024,
            ny=1024,
            nz=1024,
            global_devices=4,
            local_devices=4,
            method="spike",
        )
        decomposition.validate()

        self.assertEqual(decomposition.nxh, 513)
        self.assertEqual(decomposition.ny_local, 256)
        self.assertEqual(decomposition.nz_local, 256)
        self.assertEqual(
            decomposition.local_physical_shape("z-first"),
            (4, 256, 1024, 1024),
        )
        self.assertEqual(
            decomposition.global_physical_shape("z-first"),
            (1024, 1024, 1024),
        )

    def test_rejects_nondivisible_vertical_grid(self) -> None:
        decomposition = SlabDecomposition(
            nx=64,
            ny=64,
            nz=66,
            global_devices=4,
            local_devices=4,
            method="spike",
        )

        with self.assertRaisesRegex(ValueError, "nz=66"):
            decomposition.validate()
