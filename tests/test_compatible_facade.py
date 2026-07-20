from __future__ import annotations

import unittest

import numpy as np

from spectral_fd import (
    Poisson3DConfig,
    Poisson3DSolver,
    runtime_from_initialized_jax,
)


class CompatibleFacadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import jax
        import jax.numpy as jnp

        cls.jax = jax
        cls.jnp = jnp
        cls.config = Poisson3DConfig(
            nx=8,
            ny=8,
            nz=8,
            lx=8.0,
            ly=8.0,
            lz=8.0,
            dtype="float32",
            method="transpose",
            tridiag="thomas",
            data_layout="z-first",
            discretization="cell-centered-compatible",
        )
        cls.solver = Poisson3DSolver(
            cls.config,
            runtime=runtime_from_initialized_jax(jax),
        )

    def test_application_owned_runtime_and_outer_jit(self) -> None:
        rhs = self.jnp.zeros(
            self.solver.local_input_shape,
            dtype=self.jnp.float32,
        )
        rhs = rhs.at[:, -1].set(1.0)

        eager = self.solver.solve(rhs)
        compiled = self.jax.jit(self.solver.solve)(rhs)

        self.assertGreater(float(self.jnp.linalg.norm(eager)), 0.0)
        np.testing.assert_allclose(compiled, eager, rtol=2.0e-6, atol=2.0e-6)
        self.assertAlmostEqual(float(self.jnp.mean(eager)), 0.0, places=6)
        self.assertLess(float(np.asarray(self.solver.residual(rhs))[0]), 2.0e-5)

    def test_implicit_gradient_uses_the_factorized_solve(self) -> None:
        rhs = self.jnp.zeros(
            self.solver.local_input_shape,
            dtype=self.jnp.float32,
        )
        rhs = rhs.at[:, 0].set(-1.0).at[:, -1].set(1.0)
        weight = self.jnp.zeros_like(rhs)
        weight = weight.at[:, 1].set(-1.0).at[:, -2].set(1.0)
        dz2 = (self.config.lz / self.config.nz) ** 2

        def neumann_dg(values):
            result = self.jnp.zeros_like(values)
            result = result.at[:, 0].set(
                (values[:, 1] - values[:, 0]) / dz2
            )
            result = result.at[:, 1:-1].set(
                (
                    values[:, 2:]
                    - 2.0 * values[:, 1:-1]
                    + values[:, :-2]
                )
                / dz2
            )
            return result.at[:, -1].set(
                (values[:, -2] - values[:, -1]) / dz2
            )

        gradient = self.jax.grad(
            lambda value: self.jnp.vdot(
                weight,
                self.solver.implicit_solve(neumann_dg, value),
            ).real
        )(rhs)

        expected = self.solver.solve(weight)
        np.testing.assert_allclose(gradient, expected, rtol=3.0e-5, atol=3.0e-5)

    def test_nonsymmetric_implicit_solve_requires_transpose(self) -> None:
        rhs = self.jnp.zeros(
            self.solver.local_input_shape,
            dtype=self.jnp.float32,
        )
        with self.assertRaisesRegex(ValueError, "transpose_solve"):
            self.solver.implicit_solve(
                lambda value: value,
                rhs,
                symmetric=False,
            )

    def test_layout_tridiagonal_and_execution_matrix(self) -> None:
        runtime = runtime_from_initialized_jax(self.jax)
        for layout in ("z-first", "xyz"):
            for tridiagonal in ("thomas", "pcr"):
                config = Poisson3DConfig(
                    nx=8,
                    ny=8,
                    nz=8,
                    lx=8.0,
                    ly=8.0,
                    lz=8.0,
                    dtype="float32",
                    method="transpose",
                    tridiag=tridiagonal,
                    data_layout=layout,
                    discretization="cell-centered-compatible",
                )
                solver = Poisson3DSolver(config, runtime=runtime)
                values = self.jnp.arange(
                    np.prod(solver.local_input_shape),
                    dtype=self.jnp.float32,
                ).reshape(solver.local_input_shape)
                rhs = self.jnp.sin(values * 0.17)
                for execution in ("monolithic", "staged"):
                    with self.subTest(
                        layout=layout,
                        tridiagonal=tridiagonal,
                        execution=execution,
                    ):
                        pressure = solver.solve(rhs, execution=execution)
                        self.assertTrue(bool(self.jnp.isfinite(pressure).all()))
                        residual = float(np.asarray(solver.residual(rhs))[0])
                        self.assertLess(residual, 3.0e-5)

    def test_spike_methods_match_transpose(self) -> None:
        runtime = runtime_from_initialized_jax(self.jax)
        cases = (
            ("spike", "z-first", "thomas", "allgather", "dense"),
            ("spike", "z-first", "pcr", "alltoall", "selected-rows"),
            ("spike", "xyz", "thomas", "alltoall", "block-thomas"),
            ("spike", "xyz", "pcr", "allgather", "selected-rows"),
            ("spike-adaptive", "z-first", "thomas", "alltoall", "dense"),
            (
                "spike-adaptive",
                "z-first",
                "pcr",
                "allgather",
                "selected-rows",
            ),
            ("spike-adaptive", "xyz", "thomas", "allgather", "block-thomas"),
            ("spike-adaptive", "xyz", "pcr", "alltoall", "selected-rows"),
        )
        references = {}
        right_hand_sides = {}
        for layout in ("z-first", "xyz"):
            config = Poisson3DConfig(
                nx=8,
                ny=8,
                nz=8,
                lx=8.0,
                ly=8.0,
                lz=8.0,
                dtype="float32",
                method="transpose",
                tridiag="thomas",
                data_layout=layout,
                discretization="cell-centered-compatible",
            )
            solver = Poisson3DSolver(config, runtime=runtime)
            values = self.jnp.arange(
                np.prod(solver.local_input_shape),
                dtype=self.jnp.float32,
            ).reshape(solver.local_input_shape)
            rhs = self.jnp.sin(values * 0.17)
            if layout == "z-first":
                rhs = rhs.at[-1, -1].add(1.0)
            else:
                rhs = rhs.at[-1, ..., -1].add(1.0)
            right_hand_sides[layout] = rhs
            references[layout] = solver.solve(rhs)

        for method, layout, tridiagonal, collective, interface_solver in cases:
            with self.subTest(
                method=method,
                layout=layout,
                tridiagonal=tridiagonal,
                collective=collective,
                interface_solver=interface_solver,
            ):
                config = Poisson3DConfig(
                    nx=8,
                    ny=8,
                    nz=8,
                    lx=8.0,
                    ly=8.0,
                    lz=8.0,
                    dtype="float32",
                    method=method,
                    tridiag=tridiagonal,
                    data_layout=layout,
                    discretization="cell-centered-compatible",
                    spike_interface_collective=collective,
                    spike_interface_solver=interface_solver,
                )
                solver = Poisson3DSolver(config, runtime=runtime)
                rhs = right_hand_sides[layout]
                pressure = solver.solve(rhs)
                np.testing.assert_allclose(
                    pressure,
                    references[layout],
                    rtol=4.0e-5,
                    atol=4.0e-5,
                )
                residual = float(np.asarray(solver.residual(rhs))[0])
                self.assertLess(residual, 3.0e-5)


class CompatibleConfigTests(unittest.TestCase):
    def test_compatible_mode_accepts_spike_methods(self) -> None:
        for method in ("spike", "spike-adaptive"):
            Poisson3DConfig(
                method=method,
                discretization="cell-centered-compatible",
            ).validate()

    def test_compatible_mode_requires_nyquist_exclusion(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            Poisson3DConfig(
                nyquist_filter=False,
                discretization="cell-centered-compatible",
            ).validate()


if __name__ == "__main__":
    unittest.main()
