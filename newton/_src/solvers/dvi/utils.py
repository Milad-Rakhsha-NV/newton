# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility classes and functions for DVI solver."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .numerical_solver import NumericalSolver


@dataclass
class DVITimings:
    """Timing data for DVI solver steps (times in milliseconds)."""

    total: float = 0.0
    joint_solve: float = 0.0
    contact_solve: float = 0.0
    position_solve: float = 0.0
    contact_position_solve: float = 0.0
    integration: float = 0.0

    def reset(self):
        """Reset all timings to zero."""
        self.total = 0.0
        self.joint_solve = 0.0
        self.contact_solve = 0.0
        self.position_solve = 0.0
        self.contact_position_solve = 0.0
        self.integration = 0.0


class DVITimer:
    """Timer utility for DVI solver with controlled output.

    Collects timing data and prints in a compact format.
    Prints initially for first few frames, then at lower frequency.
    Gracefully handles CUDA graph capture (disables sync during capture).
    """

    def __init__(
        self,
        enabled: bool = False,
        initial_prints: int = 2,
        print_interval: int = 100,
    ):
        self.enabled = enabled
        self.initial_prints = initial_prints
        self.print_interval = print_interval
        self.frame_count = 0
        self.timings = DVITimings()
        self._start_times: dict[str, float] = {}
        self._capturing = False

    def set_capturing(self, capturing: bool):
        """Set graph capture mode (disables sync during capture)."""
        self._capturing = capturing

    def start(self, name: str):
        """Start timing a named section."""
        if not self.enabled or self._capturing:
            return
        try:
            wp.synchronize()
        except RuntimeError:
            self._capturing = True
            return
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing a named section and return elapsed ms."""
        if not self.enabled or self._capturing:
            return 0.0
        try:
            wp.synchronize()
        except RuntimeError:
            self._capturing = True
            return 0.0
        start_time = self._start_times.get(name, time.perf_counter())
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return elapsed_ms

    def should_print(self) -> bool:
        """Check if timing should be printed this frame."""
        if not self.enabled or self._capturing:
            return False
        if self.frame_count < self.initial_prints:
            return True
        frames_after_initial = self.frame_count - self.initial_prints
        return frames_after_initial > 0 and frames_after_initial % self.print_interval == 0

    def print_timings(self):
        """Print timing results in a compact single-line format."""
        if not self.should_print():
            return

        t = self.timings
        print(
            f"[DVI F{self.frame_count:>4}] "
            f"Total: {t.total:>5.1f}ms | "
            f"Joint: {t.joint_solve:>5.2f}ms | "
            f"Contact: {t.contact_solve:>5.2f}ms | "
            f"Position: {t.position_solve:>5.2f}ms | "
            f"Integrator: {t.integration:>4.2f}ms"
        )

    def next_frame(self):
        """Advance to next frame and reset timings."""
        self.frame_count += 1
        self.timings.reset()


# Alpha threshold: values above this are considered "effectively no Baumgarte"
_BAUMGARTE_DISABLED_ALPHA = 1e5


def validate_solver_config(
    joint_solver: NumericalSolver,
    contact_solver: NumericalSolver,
    enable_position_correction: bool,
    enable_contact_position_correction: bool = False,
):
    """Validate solver configuration and enforce mutual exclusivity.

    Baumgarte stabilization (small alpha in the velocity-level solve) and
    position correction (post-stabilization projection) are **alternative**
    approaches to constraint drift.  Using both simultaneously causes
    double-correction that destabilizes the simulation.

    When position correction is enabled and a solver has a small alpha,
    this function **overrides** alpha to a large value and emits a clear
    warning so the user knows what happened.

    This applies independently to joints and contacts:
    - Joint position correction → disables joint Baumgarte
    - Contact position correction → disables contact Baumgarte

    Args:
        joint_solver: Numerical solver for joint constraints.
        contact_solver: Numerical solver for contact forces.
        enable_position_correction: Whether joint position correction is enabled.
        enable_contact_position_correction: Whether contact position correction is enabled.
    """
    if enable_position_correction:
        # Enforce: when joint position correction is on, joint Baumgarte must be off.
        cfg = joint_solver.config
        if cfg.alpha < _BAUMGARTE_DISABLED_ALPHA:
            old_alpha = cfg.alpha
            cfg.alpha = _BAUMGARTE_DISABLED_ALPHA * 10  # 1e6
            warnings.warn(
                f"Joint position correction is enabled but joint solver has "
                f"Baumgarte stabilization active (alpha={old_alpha}). "
                f"Using both simultaneously causes double-correction that "
                f"destabilizes the simulation.\n"
                f"  -> Automatically overriding joint solver alpha to "
                f"{cfg.alpha:.0e} (Baumgarte disabled).\n"
                f"  -> To silence this warning, set joint alpha >= "
                f"{_BAUMGARTE_DISABLED_ALPHA:.0e} when using position correction.",
                UserWarning,
                stacklevel=3,
            )

    if enable_contact_position_correction:
        # Enforce: when contact position correction is on, contact Baumgarte must be off.
        cfg = contact_solver.config
        if cfg.alpha < _BAUMGARTE_DISABLED_ALPHA:
            old_alpha = cfg.alpha
            cfg.alpha = _BAUMGARTE_DISABLED_ALPHA * 10  # 1e6
            warnings.warn(
                f"Contact position correction is enabled but contact solver has "
                f"Baumgarte stabilization active (alpha={old_alpha}). "
                f"Using both simultaneously causes double-correction that "
                f"destabilizes the simulation.\n"
                f"  -> Automatically overriding contact solver alpha to "
                f"{cfg.alpha:.0e} (Baumgarte disabled).\n"
                f"  -> To silence this warning, set contact alpha >= "
                f"{_BAUMGARTE_DISABLED_ALPHA:.0e} when using contact position correction.",
                UserWarning,
                stacklevel=3,
            )

    joint_cfg = joint_solver.config
    has_baumgarte = joint_cfg.alpha < _BAUMGARTE_DISABLED_ALPHA

    if not enable_position_correction and not has_baumgarte:
        warnings.warn(
            "No joint constraint stabilization is configured "
            "(alpha is very large and no position_correction). "
            "Joint constraints may drift over time. Consider either:\n"
            "  1. Setting alpha to a reasonable value (0.0-0.5) for Baumgarte stabilization, or\n"
            "  2. Setting position_correction on the joint_solver config.",
            UserWarning,
            stacklevel=3,
        )


def print_solver_config(
    joint_solver: NumericalSolver,
    contact_solver: NumericalSolver,
    enable_position_correction: bool,
    enable_contact_position_correction: bool = False,
    position_solver: NumericalSolver | None = None,
    contact_position_solver: NumericalSolver | None = None,
):
    """Print solver configuration in a clean format.

    Args:
        joint_solver: Numerical solver for joint constraints.
        contact_solver: Numerical solver for contact forces.
        position_solver: Numerical solver for position correction.
        enable_position_correction: Whether joint position correction is enabled.
        enable_contact_position_correction: Whether contact position correction is enabled.
    """

    def _print_single_solver(name: str, solver: NumericalSolver, is_joint: bool = False):
        """Print config for a solver, showing only params it uses."""
        solver_type = type(solver).__name__
        cfg = solver.config
        defaults = solver.default_params

        def fmt_val(param_name: str, val):
            if param_name in defaults and val == defaults[param_name]:
                return f"{val} (default)"
            return str(val)

        print(f"{name}: {solver_type}")

        if "reg" in defaults:
            print(f"  reg: {fmt_val('reg', cfg.reg)}")
        if "max_iterations" in defaults:
            max_iter = fmt_val("max_iterations", cfg.max_iterations)
            print(f"  max_iter: {max_iter}")
        if "tolerance" in defaults:
            print(f"  tolerance: {fmt_val('tolerance', cfg.tolerance)}")
        if "omega" in defaults:
            print(f"  omega: {fmt_val('omega', cfg.omega)}")
        if "relax" in defaults:
            print(f"  relax: {fmt_val('relax', cfg.relax)}")
        if "backtrack_iterations" in defaults:
            bt_iter = fmt_val("backtrack_iterations", cfg.backtrack_iterations)
            print(f"  backtrack_iter: {bt_iter}")

        if is_joint and cfg.alpha < 1e5:  # alpha < large means Baumgarte is active
            print(f"  alpha (Baumgarte damping): {cfg.alpha}")
        if cfg.recovery_speed > 0:
            print(f"  recovery_speed: {cfg.recovery_speed} m/s")

        if cfg.warm_start:
            print(f"  warm_start: enabled")

    print("\n" + "─" * 60)
    print("SolverDVI Configuration")
    print("─" * 60)

    _print_single_solver("Joint Solver", joint_solver, is_joint=True)
    _print_single_solver("Contact Solver", contact_solver)

    if enable_position_correction and position_solver is not None:
        _print_single_solver("Joint Position Solver", position_solver)

    if enable_contact_position_correction and contact_position_solver is not None:
        _print_single_solver("Contact Position Solver", contact_position_solver)

    print("─" * 60 + "\n")
