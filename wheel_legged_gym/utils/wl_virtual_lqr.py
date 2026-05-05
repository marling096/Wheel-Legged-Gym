# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""论文式连续时间 LQR 对象（虚拟腿坐标 + 车体纵向），与仿真中 ``U=-K X``、VMC 力矩映射一致。

参考 Chen 等 «轮腿式平衡机器人控制»（DOI 10.13976/j.cnki.xk.2023.2533）：
https://xk.sia.cn/cn/article/doi/10.13976/j.cnki.xk.2023.2533?viewType=HTML

默认 ``state_model=paper6`` 时状态量与原文一致::

    X = [ x , ẋ , θ , θ̇ , l , ẋ_l ]ᵀ

其中 ``θ,l`` 为单腿虚拟姿态与腿长（相对指令）；``x`` 为纵向位置误差积分，``ẋ`` 为纵向速度误差
（本体前向速度与指令之差）。亦可选用 ``legacy4`` 保留旧的四维虚拟腿模型。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.utils.wl_urdf_params import parse_wheellegged_urdf_inertial


STATE_DIM_PAPER6 = 6
STATE_DIM_LEGACY4 = 4

STATE_SPACE_DESCRIPTION_PAPER6 = """
━━━━━━━━ vmc_lqr 连续域状态空间（论文状态顺序，state_model=paper6）━━━━━━━━
状态向量 X ∈ R^6::

    X = [ x , ẋ , θ , θ̇ , l , ẋ_l ]ᵀ

说明（与 LeggedRobotVMC._compute_torques 堆叠顺序一致）::
    x      — 纵向位置误差积分  ∫ (v_x − v_x,cmd) dt  [m]
    ẋ      — 纵向速度误差      v_x − v_x,cmd       [m/s]（机体前向）
    θ      — 虚拟腿姿态误差    θ0 − θ0,ref        [rad]
    θ̇     — θ̇0                                      [rad/s]
    l      — 虚拟腿长误差      L0 − L0,ref        [m]
    ẋ_l    — Ḻ0                                      [m/s]

控制输入 U ∈ R^2::

    U = [ τ_v , F_r ]ᵀ   （绕 θ0 的虚拟等效力矩 [N·m]；沿腿长径向力 [N]）

车体–姿态耦合块（平面小角线性化，轮处无外加水平力时 F_w=0）::

    [ M+m   m·l ] [ ẍ  ] = [ 0                    ]
    [ m·l  I_θ  ] [ θ̈ ]   [ τ + m g l θ ]

故 ẍ、θ̇̈ 对 (θ, τ) affine；腿长通道解耦::

    ẋ_l = ẋ_l ,   ẍ_l = F_r / m_eff

拼装得到 Ẋ = A X + B U（见运行时打印的 A,B）。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

STATE_SPACE_DESCRIPTION_LEGACY4 = """
━━━━━━━━ vmc_lqr state_model=legacy4（旧四维虚拟腿对象）━━━━━━━━
    X = [ Δθ , θ̇ , ΔL , Ḻ ]ᵀ ,   U = [ τ_v , F_r ]ᵀ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def build_balance_subsystem_matrices(
    alpha_theta: float, I_drive: float
) -> Tuple[np.ndarray, np.ndarray]:
    """legacy4：平衡子系统 ẋ_B = A_B x_B + B_B τ_v."""
    I_drive = max(float(I_drive), 1e-6)
    A_B = np.array([[0.0, 1.0], [float(alpha_theta), 0.0]], dtype=np.float64)
    B_B = np.array([[0.0], [1.0 / I_drive]], dtype=np.float64)
    return A_B, B_B


def build_longitudinal_leg_subsystem_matrices(m_eff: float) -> Tuple[np.ndarray, np.ndarray]:
    """腿长子系统 ẋ_L = A_L x_L + B_L F_r，x_L = [ΔL, Ḻ]ᵀ."""
    m_eff = max(float(m_eff), 1e-6)
    A_L = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    B_L = np.array([[0.0], [1.0 / m_eff]], dtype=np.float64)
    return A_L, B_L


def assemble_legacy_four_state_matrices(
    A_B: np.ndarray,
    B_B: np.ndarray,
    A_L: np.ndarray,
    B_L: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """X=[Δθ,θ̇,ΔL,Ḻ]ᵀ（与旧代码堆叠顺序一致：先姿态块再腿长块）."""
    A = np.zeros((4, 4), dtype=np.float64)
    B = np.zeros((4, 2), dtype=np.float64)
    A[0:2, 0:2] = A_B
    A[2:4, 2:4] = A_L
    B[0:2, 0:1] = B_B
    B[2:4, 1:2] = B_L
    return A, B


def _build_paper_six_state_AB(
    *,
    M_cart: float,
    m_body: float,
    l_arm: float,
    I_theta: float,
    gravity: float,
    m_eff: float,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """构造 X=[x,ẋ,θ,θ̇,l,ẋ_l]ᵀ 的 A (6×6)、B (6×2)."""
    M_cart = max(float(M_cart), 1e-3)
    m_body = max(float(m_body), 1e-3)
    l_arm = max(float(l_arm), 1e-3)
    I_theta = max(float(I_theta), 1e-4)
    Kc = np.array(
        [
            [M_cart + m_body, m_body * l_arm],
            [m_body * l_arm, I_theta],
        ],
        dtype=np.float64,
    )
    rhs_tau = np.array([0.0, 1.0], dtype=np.float64)
    rhs_grav_theta = np.array([0.0, m_body * gravity * l_arm], dtype=np.float64)
    inv_tau = np.linalg.solve(Kc, rhs_tau)
    inv_g = np.linalg.solve(Kc, rhs_grav_theta)
    # ẍ = inv_g[0]*θ + inv_tau[0]*τ ;  θ̈ = inv_g[1]*θ + inv_tau[1]*τ
    A = np.zeros((6, 6), dtype=np.float64)
    B = np.zeros((6, 2), dtype=np.float64)
    A[0, 1] = 1.0
    A[1, 2] = float(inv_g[0])
    B[1, 0] = float(inv_tau[0])
    A[2, 3] = 1.0
    A[3, 2] = float(inv_g[1])
    B[3, 0] = float(inv_tau[1])
    A[4, 5] = 1.0
    B[5, 1] = 1.0 / max(float(m_eff), 1e-6)
    coup_diag = {
        "cart_pendulum_K11": round(float(Kc[0, 0]), 6),
        "cart_pendulum_K12": round(float(Kc[0, 1]), 6),
        "cart_pendulum_K22": round(float(Kc[1, 1]), 6),
        "ddot_x_from_theta": round(float(inv_g[0]), 8),
        "ddot_x_from_tau": round(float(inv_tau[0]), 8),
        "ddot_theta_from_theta": round(float(inv_g[1]), 8),
        "ddot_theta_from_tau": round(float(inv_tau[1]), 8),
    }
    return A, B, coup_diag


def _print_lqr_paper6(A, B, Q, R, K, diagnostics: Optional[Dict]) -> None:
    with np.printoptions(precision=8, suppress=True, linewidth=120):
        print(STATE_SPACE_DESCRIPTION_PAPER6.rstrip())
        print(
            "\n========== vmc_lqr: LQR (paper6, continuous-time, one-shot) =========="
        )
        if diagnostics:
            print("[vmc_lqr] Lumped parameters:")
            for k, v in diagnostics.items():
                print(f"          {k}: {v}")
            print("")
        print("[vmc_lqr] Ẋ = A X + B U")
        print("[vmc_lqr] A (6x6):\n", A)
        print("[vmc_lqr] B (6x2):\n", B)
        print("[vmc_lqr] Q (6x6):\n", Q)
        print("[vmc_lqr] R (2x2):\n", R)
        print("[vmc_lqr] K (2x6), U = -K @ X:\n", K)
        print("========================================================================\n")


def _print_lqr_legacy4(
    A_B, B_B, A_L, B_L, A, B, Q, R, K, diagnostics: Optional[Dict]
) -> None:
    with np.printoptions(precision=8, suppress=True, linewidth=120):
        print(STATE_SPACE_DESCRIPTION_LEGACY4.rstrip())
        print("\n========== vmc_lqr: LQR (legacy4, continuous-time, one-shot) ==========")
        if diagnostics:
            print("[vmc_lqr] Lumped parameters:")
            for k, v in diagnostics.items():
                print(f"          {k}: {v}")
            print("")
        print("[vmc_lqr] Balance: A_B, B_B")
        print(A_B)
        print(B_B)
        print("[vmc_lqr] Leg length: A_L, B_L")
        print(A_L)
        print(B_L)
        print("[vmc_lqr] A (4x4):\n", A)
        print("[vmc_lqr] B (4x2):\n", B)
        print("[vmc_lqr] Q (4x4):\n", Q)
        print("[vmc_lqr] R (2x2):\n", R)
        print("[vmc_lqr] K (2x4), U = -K @ X:\n", K)
        print("========================================================================\n")


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


def _alpha_theta_balance_linearization(
    mode: str,
    *,
    gravity: float,
    total_mass: float,
    L0_nom: float,
    r_w: float,
    I_yy_cm: float,
    base_com_z: Optional[float],
    com_height_offset_m: float,
    h_lever_min: float,
    h_lever_max: float,
    alpha_theta_cap: float,
) -> Tuple[float, Dict]:
    """legacy4 用：返回 α_θ 及诊断。"""
    if mode == "legacy":
        l_com = max(r_w + 0.55 * L0_nom, 0.08)
        alpha = gravity / l_com
        diag = {
            "balance_linearization": "legacy (g/l_com)",
            "l_com [m]": round(l_com, 6),
            "alpha_theta [1/s^2]": round(alpha, 6),
        }
        return float(alpha), diag

    z_com = abs(base_com_z) if base_com_z is not None else 0.12
    h_lever = float(np.clip(z_com + r_w + com_height_offset_m, h_lever_min, h_lever_max))
    M = max(total_mass, 1e-3)
    I_cm = max(float(I_yy_cm), 1e-4)
    I_pivot = I_cm + M * h_lever * h_lever
    alpha = (M * gravity * h_lever) / max(I_pivot, 1e-9)
    alpha = float(min(alpha, alpha_theta_cap))
    diag = {
        "balance_linearization": "paper (M g h / (I_yy_cm + M h^2))",
        "h_lever [m]": round(h_lever, 6),
        "I_yy_cm [kg*m^2]": round(I_cm, 6),
        "I_pivot [kg*m^2]": round(I_pivot, 6),
        "M_tot [kg]": round(M, 4),
        "alpha_theta [1/s^2]": round(alpha, 6),
    }
    return alpha, diag


def _h_lever_for_coupled_model(
    mode: str,
    *,
    gravity: float,
    total_mass: float,
    L0_nom: float,
    r_w: float,
    I_yy_cm: float,
    base_com_z: Optional[float],
    com_height_offset_m: float,
    h_lever_min: float,
    h_lever_max: float,
    alpha_theta_cap: float,
) -> Tuple[float, Dict]:
    """paper6 车体–摆耦合块用的摆臂长 l（与 paper legacy α 推导共用几何）。"""
    alpha_dummy, diag = _alpha_theta_balance_linearization(
        mode if mode != "legacy" else "paper",
        gravity=gravity,
        total_mass=total_mass,
        L0_nom=L0_nom,
        r_w=r_w,
        I_yy_cm=I_yy_cm,
        base_com_z=base_com_z,
        com_height_offset_m=com_height_offset_m,
        h_lever_min=h_lever_min,
        h_lever_max=h_lever_max,
        alpha_theta_cap=alpha_theta_cap,
    )
    _ = alpha_dummy
    h_lever = float(diag["h_lever [m]"])
    return h_lever, diag


def build_virtual_leg_decoupled_state_matrices(
    cfg, gravity: float = 9.81
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """构造 ``A,B``（不求 Riccati）。

    Returns:
        ``paper6``: A (6,6), B (6,2), meta。
        ``legacy4``: A (4,4), B (4,2), meta（含 A_B,B_B,A_L,B_L）。
    """
    urdf_path = resolve_wl_urdf_path(cfg)
    urdf = parse_wheellegged_urdf_inertial(urdf_path)

    L0_nom, _ = _nominal_L0_theta(cfg)
    r_w = urdf.get("wheel_radius")
    if r_w is None:
        r_w = 0.0675

    lqr = cfg.control.lqr
    state_model = getattr(lqr, "state_model", "paper6")
    bal_mode = getattr(lqr, "balance_linearization", "paper")
    if bal_mode not in ("paper", "legacy"):
        bal_mode = "paper"

    base_xyz = urdf.get("base_com_xyz")
    base_com_z = float(base_xyz[2]) if base_xyz is not None else None

    I_yy = urdf.get("base_inertia_iyy")
    if I_yy is None:
        I_yy = 0.23

    total_mass = float(urdf["total_mass"])
    wheel_mass_sum = float(urdf.get("wheel_mass_sum") or 0.0)
    M_cart = max(
        wheel_mass_sum,
        float(getattr(lqr, "M_cart_floor_kg", 1.2)),
    )
    m_body = max(total_mass - wheel_mass_sum, float(getattr(lqr, "m_body_floor_kg", 0.8)))

    m_frac = float(getattr(lqr, "radial_mass_fraction", 0.45))
    m_eff = max(
        total_mass * m_frac,
        float(getattr(lqr, "radial_mass_min_kg", 0.5)),
    )

    if state_model == "legacy4":
        alpha_theta, bal_diag = _alpha_theta_balance_linearization(
            bal_mode,
            gravity=gravity,
            total_mass=total_mass,
            L0_nom=L0_nom,
            r_w=float(r_w),
            I_yy_cm=float(I_yy),
            base_com_z=base_com_z,
            com_height_offset_m=float(getattr(lqr, "com_height_offset_m", 0.0)),
            h_lever_min=float(getattr(lqr, "h_lever_min_m", 0.07)),
            h_lever_max=float(getattr(lqr, "h_lever_max_m", 0.55)),
            alpha_theta_cap=float(getattr(lqr, "alpha_theta_cap", 120.0)),
        )
        if bal_mode == "legacy":
            I_drive = float(I_yy) + 0.02 * total_mass
        else:
            I_drive = float(I_yy) + float(
                getattr(lqr, "inertia_drive_margin", 0.02)
            ) * total_mass
        I_drive = max(I_drive, 0.05)
        A_B, B_B = build_balance_subsystem_matrices(alpha_theta, I_drive)
        A_L, B_L = build_longitudinal_leg_subsystem_matrices(m_eff)
        A, B = assemble_legacy_four_state_matrices(A_B, B_B, A_L, B_L)
        meta = {
            **bal_diag,
            "state_model": "legacy4",
            "L0_nom [m]": round(L0_nom, 6),
            "r_wheel [m]": round(float(r_w), 6),
            "I_drive [kg*m^2]": round(float(I_drive), 6),
            "m_eff [kg]": round(float(m_eff), 6),
            "radial_mass_fraction": m_frac,
            "A_B": A_B,
            "B_B": B_B,
            "A_L": A_L,
            "B_L": B_L,
        }
        return A, B, meta

    # ----- paper6 -----
    h_lever, h_diag = _h_lever_for_coupled_model(
        bal_mode,
        gravity=gravity,
        total_mass=total_mass,
        L0_nom=L0_nom,
        r_w=float(r_w),
        I_yy_cm=float(I_yy),
        base_com_z=base_com_z,
        com_height_offset_m=float(getattr(lqr, "com_height_offset_m", 0.0)),
        h_lever_min=float(getattr(lqr, "h_lever_min_m", 0.07)),
        h_lever_max=float(getattr(lqr, "h_lever_max_m", 0.55)),
        alpha_theta_cap=float(getattr(lqr, "alpha_theta_cap", 120.0)),
    )
    l_arm = h_lever
    I_theta = max(
        float(I_yy) + m_body * l_arm * l_arm,
        float(getattr(lqr, "I_theta_floor_kg_m2", 0.08)),
    )
    A, B, coup_diag = _build_paper_six_state_AB(
        M_cart=M_cart,
        m_body=m_body,
        l_arm=l_arm,
        I_theta=I_theta,
        gravity=gravity,
        m_eff=m_eff,
    )
    meta = {
        **h_diag,
        **coup_diag,
        "state_model": "paper6",
        "M_cart [kg]": round(M_cart, 4),
        "m_body [kg]": round(m_body, 4),
        "l_arm [m]": round(l_arm, 6),
        "I_theta [kg*m^2]": round(I_theta, 6),
        "L0_nom [m]": round(L0_nom, 6),
        "r_wheel [m]": round(float(r_w), 6),
        "m_eff [kg]": round(float(m_eff), 6),
        "radial_mass_fraction": m_frac,
    }
    return A, B, meta


def build_virtual_leg_lqr_gain(cfg, gravity: float = 9.81) -> np.ndarray:
    """Return ``K`` for ``U = -K X``（paper6 时为 ``(2,6)``，legacy4 为 ``(2,4)``）。"""
    try:
        from scipy.linalg import solve_continuous_are
    except ImportError as e:
        raise ImportError(
            "vmc_lqr control_path requires scipy (pip install scipy)."
        ) from e

    A, B, meta = build_virtual_leg_decoupled_state_matrices(cfg, gravity=gravity)
    lqr = cfg.control.lqr

    if meta.get("state_model") == "legacy4":
        Q = np.diag(
            [
                float(lqr.q_theta),
                float(lqr.q_theta_dot),
                float(lqr.q_L),
                float(lqr.q_L_dot),
            ]
        )
        A_B = meta["A_B"]
        B_B = meta["B_B"]
        A_L = meta["A_L"]
        B_L = meta["B_L"]
        diag_print = {k: v for k, v in meta.items() if k not in ("A_B", "B_B", "A_L", "B_L")}
    else:
        Q = np.diag(
            [
                float(getattr(lqr, "q_x", 0.25)),
                float(getattr(lqr, "q_x_dot", 1.5)),
                float(lqr.q_theta),
                float(lqr.q_theta_dot),
                float(lqr.q_L),
                float(lqr.q_L_dot),
            ]
        )
        A_B = B_B = A_L = B_L = None
        diag_print = dict(meta)

    R = np.diag([float(lqr.r_torque), float(lqr.r_force)])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    if meta.get("state_model") == "legacy4":
        _print_lqr_legacy4(A_B, B_B, A_L, B_L, A, B, Q, R, K, diagnostics=diag_print)
    else:
        _print_lqr_paper6(A, B, Q, R, K, diagnostics=diag_print)

    return K.astype(np.float64)


# 兼容旧文档字符串引用
STATE_SPACE_DESCRIPTION = STATE_SPACE_DESCRIPTION_PAPER6
