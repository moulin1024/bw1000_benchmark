"""Internal callable engine shared by the public API and legacy benchmark."""

from __future__ import annotations

import numpy as np


class Poisson3DEngine:
    """Bind precomputed factors to monolithic and staged solve callables."""

    def __init__(
        self,
        *,
        config,
        global_devices,
        local_devices,
        process_count,
        process_index,
        local_input_shape,
        global_input_shape,
        solve_monolithic,
        solve_staged,
        residual,
        pipeline_ops,
    ):
        self.config = config
        self.global_devices = global_devices
        self.local_devices = local_devices
        self.process_count = process_count
        self.process_index = process_index
        self.local_input_shape = local_input_shape
        self.global_input_shape = global_input_shape
        self._solve_monolithic = solve_monolithic
        self._solve_staged = solve_staged
        self._residual = residual
        self._pipeline_ops = pipeline_ops

    def _validate_rhs(self, rhs) -> None:
        try:
            shape = tuple(rhs.shape)
            dtype = np.dtype(rhs.dtype)
        except AttributeError as exc:
            raise TypeError("rhs must be an array with shape and dtype") from exc
        if shape != self.local_input_shape:
            raise ValueError(
                f"rhs shape {shape} does not match the expected local shape "
                f"{self.local_input_shape}"
            )
        expected_dtype = np.dtype(self.config.dtype)
        if dtype != expected_dtype:
            raise TypeError(
                f"rhs dtype {dtype} does not match configured dtype "
                f"{expected_dtype}"
            )

    def solve(self, rhs, *, execution=None):
        self._validate_rhs(rhs)
        selected_execution = execution or self.config.pipeline_execution
        if selected_execution == "staged":
            solve = self._solve_staged
        elif selected_execution == "monolithic":
            solve = self._solve_monolithic
        else:
            raise ValueError("execution must be 'monolithic' or 'staged'")
        return solve(rhs, *self._pipeline_ops)

    def residual(self, rhs):
        self._validate_rhs(rhs)
        return self._residual(rhs)
