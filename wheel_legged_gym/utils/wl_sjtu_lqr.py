"""SJTU wheel-legged LQR model ported from ``HerKules_VOCAL_SJ_LQR_v4_with_data.m``.

State order:
    X = [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr, theta_b, dtheta_b]

Input order:
    U = [T_wl, T_wr, T_bl, T_br]

The model follows the five linear equations in the MATLAB script and evaluates
their Jacobians numerically for the current URDF parameters.
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR


STATE_ORDER = (
    "s",
    "ds",
    "phi",
    "dphi",
    "theta_ll",
    "dtheta_ll",
    "theta_lr",
    "dtheta_lr",
    "theta_b",
    "dtheta_b",
)
INPUT_ORDER = ("T_wl", "T_wr", "T_bl", "T_br")


@dataclass(frozen=True)
class SJTUParams:
    R_w: float
    R_l: float
    l_l: float
    l_r: float
    l_wl: float
    l_wr: float
    l_bl: float
    l_br: float
    l_c: float
    m_w: float
    m_l: float
    m_b: float
    I_w: float
    I_ll: float
    I_lr: float
    I_b: float
    I_z: float
    g: float = 9.81


def _vec3(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.replace(",", " ").split()[:3]], dtype=np.float64)


def _read_urdf_maps(urdf_path: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    root = ET.parse(urdf_path).getroot()
    links: Dict[str, dict] = {}
    joints: Dict[str, dict] = {}
    for link in root.findall("link"):
        name = link.get("name") or ""
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        inertia_el = inertial.find("inertia")
        origin_el = inertial.find("origin")
        if mass_el is None or inertia_el is None:
            continue
        links[name] = {
            "mass": float(mass_el.attrib["value"]),
            "com": _vec3(origin_el.attrib.get("xyz", "0 0 0")) if origin_el is not None else np.zeros(3),
            "inertia": {k: float(inertia_el.attrib[k]) for k in ("ixx", "iyy", "izz") if k in inertia_el.attrib},
            "wheel_radius": None,
        }
        cyl = link.find("./collision/geometry/cylinder")
        if cyl is not None and "radius" in cyl.attrib:
            links[name]["wheel_radius"] = float(cyl.attrib["radius"])

    for joint in root.findall("joint"):
        name = joint.get("name") or ""
        origin_el = joint.find("origin")
        joints[name] = {
            "origin": _vec3(origin_el.attrib.get("xyz", "0 0 0")) if origin_el is not None else np.zeros(3),
        }
    return links, joints


def _virtual_leg_ik(l0: float, theta0: float, *, offset: float, l1: float, l2: float) -> Tuple[float, float]:
    end_x = float(l0) * math.cos(float(theta0) + math.pi / 2.0)
    end_y = float(l0) * math.sin(float(theta0) + math.pi / 2.0)
    rel_x = end_x - offset
    rel_y = end_y
    cos_q2 = (rel_x * rel_x + rel_y * rel_y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_q2 = float(np.clip(cos_q2, -0.999, 0.999))
    q2 = math.acos(cos_q2)
    q1 = math.atan2(rel_y, rel_x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    return q1, q2


def _estimate_leg_geometry_from_urdf(
    links: Dict[str, dict],
    leg_length: float,
    leg_angle: float,
    *,
    offset: float,
    l1: float,
    l2: float,
) -> Tuple[float, float, float]:
    """Estimate ``l_w``, ``l_b`` and leg inertia around the leg COM.

    The URDF link COMs are projected onto the VMC virtual-leg axis. This preserves
    the SJTU model convention ``l_w + l_b ~= leg_length`` while still using the
    current robot masses and inertias.
    """
    q1, q2 = _virtual_leg_ik(leg_length, leg_angle, offset=offset, l1=l1, l2=l2)
    m0 = links["lf0_Link"]["mass"]
    m1 = links["lf1_Link"]["mass"]
    # Current URDF COM distances in the planar linkage coordinates.
    d0 = abs(float(links["lf0_Link"]["com"][0]))
    d1 = abs(float(links["lf1_Link"]["com"][1]))
    p0 = np.array([offset + d0 * math.cos(q1), d0 * math.sin(q1)], dtype=np.float64)
    p1 = np.array(
        [
            offset + l1 * math.cos(q1) + d1 * math.cos(q1 + q2),
            l1 * math.sin(q1) + d1 * math.sin(q1 + q2),
        ],
        dtype=np.float64,
    )
    wheel = np.array(
        [
            offset + l1 * math.cos(q1) + l2 * math.cos(q1 + q2),
            l1 * math.sin(q1) + l2 * math.sin(q1 + q2),
        ],
        dtype=np.float64,
    )
    leg_com = (m0 * p0 + m1 * p1) / max(m0 + m1, 1e-9)
    leg_axis = wheel / max(float(np.linalg.norm(wheel)), 1e-9)
    l_bl = float(np.clip(np.dot(leg_com, leg_axis), 1e-4, leg_length - 1e-4))
    l_wl = float(max(leg_length - l_bl, 1e-4))
    I0 = float(links["lf0_Link"]["inertia"].get("izz", 0.0))
    I1 = float(links["lf1_Link"]["inertia"].get("izz", 0.0))
    I_leg = I0 + m0 * float(np.sum((p0 - leg_com) ** 2)) + I1 + m1 * float(np.sum((p1 - leg_com) ** 2))
    return l_wl, l_bl, float(max(I_leg, 1e-6))


def params_from_urdf(
    urdf_path: str,
    *,
    leg_length: float = 0.22,
    leg_angle: float = -0.07,
    gravity: float = 9.81,
) -> SJTUParams:
    links, joints = _read_urdf_maps(urdf_path)
    wheel_l = links["l_wheel_Link"]
    wheel_r = links["r_wheel_Link"]
    base = links["base_link"]
    m_w = 0.5 * (wheel_l["mass"] + wheel_r["mass"])
    m_l = links["lf0_Link"]["mass"] + links["lf1_Link"]["mass"]
    R_w = float(wheel_l["wheel_radius"] or wheel_r["wheel_radius"] or 0.0675)
    R_l = abs(float(joints["l_wheel_Joint"]["origin"][1] - joints["r_wheel_Joint"]["origin"][1])) / 2.0
    hip_z = 0.5 * (float(joints["lf0_Joint"]["origin"][2]) + float(joints["rf0_Joint"]["origin"][2]))
    l_c = abs(hip_z - float(base["com"][2]))
    offset = float(joints["lf0_Joint"]["origin"][0])
    l1 = float(joints["lf1_Joint"]["origin"][0])
    l2 = abs(float(joints["l_wheel_Joint"]["origin"][1]))
    l_w, l_b, I_leg = _estimate_leg_geometry_from_urdf(
        links, leg_length, leg_angle, offset=offset, l1=l1, l2=l2
    )
    # Add lateral wheel/leg placement to yaw inertia; pitch uses base inertia about y.
    I_z = float(base["inertia"].get("izz", 0.0)) + 2.0 * (m_w + m_l) * R_l * R_l
    return SJTUParams(
        R_w=R_w,
        R_l=R_l,
        l_l=float(leg_length),
        l_r=float(leg_length),
        l_wl=l_w,
        l_wr=l_w,
        l_bl=l_b,
        l_br=l_b,
        l_c=float(max(l_c, 1e-4)),
        m_w=float(m_w),
        m_l=float(m_l),
        m_b=float(base["mass"]),
        I_w=float(0.5 * (wheel_l["inertia"].get("izz", 0.0) + wheel_r["inertia"].get("izz", 0.0))),
        I_ll=I_leg,
        I_lr=I_leg,
        I_b=float(base["inertia"].get("iyy", 0.0)),
        I_z=I_z,
        g=float(gravity),
    )


def build_sjtu_state_space(p: SJTUParams) -> Tuple[np.ndarray, np.ndarray]:
    R_w, R_l = p.R_w, p.R_l
    l_l, l_r = p.l_l, p.l_r
    l_wl, l_wr = p.l_wl, p.l_wr
    l_bl, l_br = p.l_bl, p.l_br
    l_c = p.l_c
    m_w, m_l, m_b = p.m_w, p.m_l, p.m_b
    I_w, I_ll, I_lr = p.I_w, p.I_ll, p.I_lr
    I_b, I_z, g = p.I_b, p.I_z, p.g

    M = np.zeros((5, 5), dtype=np.float64)
    G = np.zeros((5, 3), dtype=np.float64)
    U = np.zeros((5, 4), dtype=np.float64)

    M[0, 0] = I_w * l_l / R_w + m_w * R_w * l_l + m_l * R_w * l_bl
    M[0, 2] = m_l * l_wl * l_bl - I_ll
    G[0, 0] = (m_l * l_wl + m_b * l_l / 2.0) * g
    U[0, 0] = -(1.0 + l_l / R_w)
    U[0, 2] = 1.0

    M[1, 1] = I_w * l_r / R_w + m_w * R_w * l_r + m_l * R_w * l_br
    M[1, 3] = m_l * l_wr * l_br - I_lr
    G[1, 1] = (m_l * l_wr + m_b * l_r / 2.0) * g
    U[1, 1] = -(1.0 + l_r / R_w)
    U[1, 3] = 1.0

    wheel_linear = m_w * R_w * R_w + I_w + m_l * R_w * R_w + m_b * R_w * R_w / 2.0
    M[2, 0] = -wheel_linear
    M[2, 1] = -wheel_linear
    M[2, 2] = -(m_l * R_w * l_wl + m_b * R_w * l_l / 2.0)
    M[2, 3] = -(m_l * R_w * l_wr + m_b * R_w * l_r / 2.0)
    U[2, 0] = 1.0
    U[2, 1] = 1.0

    body_wheel = m_w * R_w * l_c + I_w * l_c / R_w + m_l * R_w * l_c
    M[3, 0] = body_wheel
    M[3, 1] = body_wheel
    M[3, 2] = m_l * l_wl * l_c
    M[3, 3] = m_l * l_wr * l_c
    M[3, 4] = -I_b
    G[3, 2] = m_b * g * l_c
    U[3, 0] = -l_c / R_w
    U[3, 1] = -l_c / R_w
    U[3, 2] = -1.0
    U[3, 3] = -1.0

    yaw_wheel = I_z * R_w / (2.0 * R_l) + I_w * R_l / R_w
    M[4, 0] = yaw_wheel
    M[4, 1] = -yaw_wheel
    M[4, 2] = I_z * l_l / (2.0 * R_l)
    M[4, 3] = -I_z * l_r / (2.0 * R_l)
    U[4, 0] = -R_l / R_w
    U[4, 1] = R_l / R_w

    J_A = -np.linalg.solve(M, G)
    J_B = -np.linalg.solve(M, U)

    A = np.zeros((10, 10), dtype=np.float64)
    B = np.zeros((10, 4), dtype=np.float64)
    for row in range(0, 10, 2):
        A[row, row + 1] = 1.0
    for p_col, theta_col in enumerate((4, 6, 8)):
        A[1, theta_col] = R_w * (J_A[0, p_col] + J_A[1, p_col]) / 2.0
        A[3, theta_col] = (
            R_w * (-J_A[0, p_col] + J_A[1, p_col]) / (2.0 * R_l)
            - l_l * J_A[2, p_col] / (2.0 * R_l)
            + l_r * J_A[3, p_col] / (2.0 * R_l)
        )
        A[5, theta_col] = J_A[2, p_col]
        A[7, theta_col] = J_A[3, p_col]
        A[9, theta_col] = J_A[4, p_col]
    for h in range(4):
        B[1, h] = R_w * (J_B[0, h] + J_B[1, h]) / 2.0
        B[3, h] = (
            R_w * (-J_B[0, h] + J_B[1, h]) / (2.0 * R_l)
            - l_l * J_B[2, h] / (2.0 * R_l)
            + l_r * J_B[3, h] / (2.0 * R_l)
        )
        B[5, h] = J_B[2, h]
        B[7, h] = J_B[3, h]
        B[9, h] = J_B[4, h]
    return A, B


def default_qr() -> Tuple[np.ndarray, np.ndarray]:
    Q = np.diag([1.0, 2.0, 12000.0, 200.0, 1000.0, 1.0, 1000.0, 1.0, 20000.0, 1.0])
    R = np.diag([0.25, 0.25, 1.5, 1.5])
    return Q, R


def solve_lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    try:
        from scipy.linalg import solve_continuous_are
    except ImportError as e:
        raise ImportError("SJTU LQR requires scipy (pip install scipy).") from e
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def build_sjtu_lqr_from_urdf(
    urdf_path: str,
    *,
    leg_length: float = 0.22,
    leg_angle: float = -0.07,
    gravity: float = 9.81,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, SJTUParams]:
    params = params_from_urdf(urdf_path, leg_length=leg_length, leg_angle=leg_angle, gravity=gravity)
    A, B = build_sjtu_state_space(params)
    if Q is None or R is None:
        Q_default, R_default = default_qr()
        Q = Q_default if Q is None else Q
        R = R_default if R is None else R
    K = solve_lqr_gain(A, B, Q, R)
    return A, B, Q, R, K, params


def main() -> None:
    default_urdf = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, "resources/robots/wl/urdf/wl.urdf")
    parser = argparse.ArgumentParser(description="Compute SJTU 10-state wheel-legged LQR K from current URDF.")
    parser.add_argument("--urdf", default=default_urdf)
    parser.add_argument("--leg-length", type=float, default=0.22)
    parser.add_argument("--leg-angle", type=float, default=-0.07)
    args = parser.parse_args()

    A, B, Q, R, K, params = build_sjtu_lqr_from_urdf(
        args.urdf, leg_length=args.leg_length, leg_angle=args.leg_angle
    )
    with np.printoptions(precision=8, suppress=True, linewidth=160):
        print("SJTU LQR state order:", STATE_ORDER)
        print("SJTU LQR input order:", INPUT_ORDER)
        print("Physical parameters from URDF:")
        for key, value in asdict(params).items():
            print(f"  {key}: {value:.8g}")
        print("A (10x10):\n", A)
        print("B (10x4):\n", B)
        print("Q (10x10):\n", Q)
        print("R (4x4):\n", R)
        print("K (4x10), U = -K @ X:\n", K)
        print("K copyable:")
        print(np.array2string(K, precision=8, separator=", ", suppress_small=False))


if __name__ == "__main__":
    main()
