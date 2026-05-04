# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous-time LQR gains for wheel-legged virtual-leg (theta0, L0) coordinates."""

from __future__ import annotations

import math

import numpy as np

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.utils.wl_urdf_params import parse_wheellegged_urdf_inertial


def _print_lqr_runtime_matrices(A, B, Q, R, K) -> None:
    """Stdout dump for debugging (single-thread Riccati solve; same K for L/R legs)."""
    with np.printoptions(precision=8, suppress=True, linewidth=120):
        print(
            "\n========== vmc_lqr: LQR matrices (continuous-time, env init / one-shot) =========="
        )
        print(
            "[vmc_lqr] Note: CARE solved once sequentially here; sim steps only apply u=-Kx per env batch.\n"
        )
        print("[vmc_lqr] A (4x4):\n", A)
        print("[vmc_lqr] B (4x2):\n", B)
        print("[vmc_lqr] Q (4x4):\n", Q)
        print("[vmc_lqr] R (2x2):\n", R)
        print("[vmc_lqr] K (2x4), u = -K @ x:\n", K)
        print("================================================================================\n")


def _nominal_L0_theta(cfg) -> tuple:
    """Match ``LeggedRobotVMC.forward_kinematics`` at default standing pose."""
    th1 = cfg.init_state.default_joint_angles["lf0_Joint"]
    th2 = cfg.init_state.default_joint_angles["lf1_Joint"]
    offset = cfg.asset.offset
    l1 = cfg.asset.l1
    l2 = cfg.asset.l2
    end_x = offset + l1 * math.cos(th1) + l2 * math.cos(th1 + th2)
    end_y = l1 * math.sin(th1) + l2 * math.sin(th1 + th2)
    L0 = math.sqrt(end_x * end_x + end_y * end_y)
    theta0 = math.atan2(end_y, end_x) - math.pi / 2.0
    return L0, theta0


def resolve_wl_urdf_path(cfg) -> str:
    path = cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR)
    return path


def build_virtual_leg_lqr_gain(cfg, gravity: float = 9.81) -> np.ndarray:
    """Return ``K`` with shape ``(2, 4)`` for ``u = -K x``.

    State per leg (linearization around references from actions):

        x = [ theta0_err, theta0_dot, L0_err, L0_dot ]

    Control:

        u = [ virtual_torque_cmd (about theta0), radial_force_cmd ]

    Dynamics model (decoupled torque→pitch accel, force→radial accel):

        theta_ddot ≈ alpha_theta * theta0_err + tau / I_eff
        L_ddot ≈ F / m_eff

    where ``alpha_theta ≈ g / l_com`` approximates inverted-pendulum-like instability
    in sagittal plane; ``l_com`` combines URDF wheel radius and nominal virtual leg length.

    Parameters ``I_eff``, ``m_eff`` come from URDF ``base_link`` inertia / total mass.

    Requires ``scipy``. Algebraic Riccati equation is solved **once** on CPU (no parallel solver);
    runtime control uses the fixed ``K`` (tensor multiply across envs is not a parallel Riccati solve).
    """
    try:
        from scipy.linalg import solve_continuous_are
    except ImportError as e:
        raise ImportError(
            "vmc_lqr control_path requires scipy (pip install scipy)."
        ) from e

    urdf_path = resolve_wl_urdf_path(cfg)
    urdf = parse_wheellegged_urdf_inertial(urdf_path)

    L0_nom, _ = _nominal_L0_theta(cfg)
    r_w = urdf.get("wheel_radius")
    if r_w is None:
        r_w = 0.0675

    # Effective pendulum length for gravity coupling on posture coordinate.
    l_com = max(r_w + 0.55 * L0_nom, 0.08)

    # Pitch inertia scale: URDF base Iyy dominates; fallback from geometry.
    I_yy = urdf.get("base_inertia_iyy")
    if I_yy is None:
        I_yy = 0.23
    I_eff = float(I_yy) + 0.02 * urdf["total_mass"]

    # Radial inertia proxy (per-leg vertical dynamics mass scale).
    m_eff = max(urdf["total_mass"] * 0.35, 0.5)

    alpha_theta = gravity / l_com

    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 1] = 1.0
    A[1, 0] = alpha_theta
    A[2, 3] = 1.0

    B = np.zeros((4, 2), dtype=np.float64)
    B[1, 0] = 1.0 / I_eff
    B[3, 1] = 1.0 / m_eff

    lqr = cfg.control.lqr
    Q = np.diag(
        [
            float(lqr.q_theta),
            float(lqr.q_theta_dot),
            float(lqr.q_L),
            float(lqr.q_L_dot),
        ]
    )
    R = np.diag([float(lqr.r_torque), float(lqr.r_force)])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    _print_lqr_runtime_matrices(A, B, Q, R, K)

    return K.astype(np.float64)
