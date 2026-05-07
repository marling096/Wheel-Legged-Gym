# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import (
    WheelLeggedCfg,
    WheelLeggedCfgPPO,
)


class WheelLeggedVMCCfg(WheelLeggedCfg):
    class env(WheelLeggedCfg.env):
        num_privileged_obs = (
            WheelLeggedCfg.env.num_observations + 7 * 11 + 3 + 6 * 7 + 3 + 3
        )

    class control(WheelLeggedCfg.control):
        action_scale_theta = 0.5
        action_scale_l0 = 0.1
        action_scale_vel = 10.0

        l0_offset = 0.175
        feedforward_force = 40.0  # [N]

        kp_theta = 50.0  # [N*m/rad]
        kd_theta = 3.0  # [N*m*s/rad]
        kp_l0 = 900.0  # [N/m]
        kd_l0 = 20.0  # [N*s/m]

        # PD Drive parameters:
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}  # [N*m/rad]
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.5}  # [N*m*s/rad]

        # 默认虚拟腿 PD + VMC（RL）。命令行 ``--control_path`` 默认为 vmc_pd，传入时会覆盖 cfg。
        control_path = "vmc_pd"

        # 在 VMC 力矩映射前叠加到虚拟腿摆矩 τ_leg（默认 0，不影响其它机器人）。
        vmc_leg_pitch_stab_kp = 0.0
        vmc_leg_pitch_stab_kd = 0.0
        vmc_leg_roll_stab_kp = 0.0
        vmc_leg_roll_stab_kd = 0.0

        class lqr:
            # sjtu10: X=[s,ds,phi,dphi,theta_ll,dtheta_ll,theta_lr,dtheta_lr,theta_b,dtheta_b]ᵀ
            # paper_ip6: X=[θ, θ̇, xb, ẋb, φ, φ̇]ᵀ, U=[wheel T, leg Tp]ᵀ（论文路径）
            # paper6: X = [x, ẋ, θ, θ̇, l, ẋ_l]ᵀ；legacy4: X = [Δθ, θ̇, ΔL, Ḻ]ᵀ
            state_model = "sjtu10"
            # upright-first tuning for paper6:
            # 1) weaken x/xdot coupling to avoid longitudinal error injecting large Fr
            # 2) emphasize theta/theta_dot stabilization
            # 3) increase force penalty to prevent aggressive radial-force oscillation
            q_x = 0.0
            q_x_dot = 0.0
            q_theta = 180.0
            q_theta_dot = 14.0
            q_L = 1000.0
            q_L_dot = 28.0
            r_torque = 1.2e-3
            r_force = 2.0e-4

            # Headless lqr_demo stand-still tuning at 0.18 m:
            # keep longitudinal tracking weak, prioritize leg/body pitch damping.
            q_paper_theta = 600.0
            q_paper_theta_dot = 45.0
            q_paper_x = 5.0
            q_paper_x_dot = 8.0
            q_paper_phi = 12000.0
            q_paper_phi_dot = 120.0
            r_paper_wheel_torque = 6.0
            r_paper_leg_torque = 0.8
            paper_velocity_ref_gain = 0.0
            paper_theta_sign = 1.0
            paper_pitch_sign = 1.0
            paper_wheel_torque_scale = 0.20
            paper_leg_torque_scale = 0.05
            paper_wheel_torque_limit = 10.0
            paper_wheel_pitch_kp = 0.0
            paper_wheel_pitch_kd = 0.0
            paper_leg_torque_limit = 60.0
            paper_leg_pd_blend = 1.0
            paper_leg_pd_kp = 90.0
            paper_leg_pd_kd = 5.0
            x_integral_limit = 2.0

            # Current-sim wheel inverted-pendulum model:
            # X=[x_i, vx, pitch, pitch_dot], U=[single-side wheel torque].
            q_sim_x = 0.2
            q_sim_x_dot = 1.0
            q_sim_pitch = 1200.0
            q_sim_pitch_dot = 50.0
            r_sim_wheel_torque = 1.0
            sim_wheel_torque_scale = 1.0
            sim_wheel_torque_limit = 5.0
            sim_wheel_force_scale = 1.0
            sim_wheel_l_com_m = 0.0
            sim_wheel_I_pitch_floor = 0.08
            sim_leg_pitch_kp = 0.0
            sim_leg_pitch_kd = 0.0
            sim_leg_torque_limit = 60.0

            # SJTU open-source 10-state model defaults, matching the MATLAB script.
            sjtu_leg_length_m = 0.22
            sjtu_leg_angle_rad = -0.07
            q_sjtu_s = 1.0
            q_sjtu_ds = 2.0
            q_sjtu_phi = 12000.0
            q_sjtu_dphi = 200.0
            q_sjtu_theta_l = 1000.0
            q_sjtu_dtheta_l = 1.0
            q_sjtu_theta_r = 1000.0
            q_sjtu_dtheta_r = 1.0
            q_sjtu_theta_b = 20000.0
            q_sjtu_dtheta_b = 1.0
            r_sjtu_wheel_l = 0.25
            r_sjtu_wheel_r = 0.25
            r_sjtu_leg_l = 1.5
            r_sjtu_leg_r = 1.5
            sjtu_wheel_torque_scale = 1.0
            sjtu_leg_torque_scale = 1.0
            sjtu_wheel_left_scale = 1.0
            sjtu_wheel_right_scale = 1.0
            sjtu_leg_left_scale = 1.0
            sjtu_leg_right_scale = 1.0
            sjtu_leg_map_ll = 1.0
            sjtu_leg_map_lr = 0.0
            sjtu_leg_map_rl = 0.0
            sjtu_leg_map_rr = 1.0
            sjtu_wheel_torque_limit = 5.0
            sjtu_leg_torque_limit = 60.0
            sjtu_leg_pd_blend = 0.0
            sjtu_leg_pd_kp = 90.0
            sjtu_leg_pd_kd = 5.0
            sjtu_wheel_pitch_kp = 0.0
            sjtu_wheel_pitch_kd = 0.0
            sjtu_leg_roll_kp = 0.0
            sjtu_leg_roll_kd = 0.0
            sjtu_balance_wheel_only = False
            sjtu_pitch_sign = 1.0
            sjtu_theta_sign = 1.0
            sjtu_yaw_sign = 1.0

            # LQR plant linearization (see wl_virtual_lqr.py; Chen et al. 2023 IP + decoupled length).
            balance_linearization = "paper"  # "paper" | "legacy"
            com_height_offset_m = 0.0  # [m] added to |base_com_z| + r_wheel for h_lever
            h_lever_min_m = 0.07
            h_lever_max_m = 0.55
            alpha_theta_cap = 120.0  # [1/s^2] numerical cap on A[1,0]
            inertia_drive_margin = 0.02  # I_drive += margin * M_tot (paper mode B torque gain)
            radial_mass_fraction = 0.45  # m_eff = max(M_tot * frac, radial_mass_min_kg)
            radial_mass_min_kg = 0.5
            # paper6 车体–摆质量拆分（URDF wheel 总质量为 M_cart 下界）
            M_cart_floor_kg = 1.2
            m_body_floor_kg = 0.8
            I_theta_floor_kg_m2 = 0.08

    class normalization(WheelLeggedCfg.normalization):
        class obs_scales(WheelLeggedCfg.normalization.obs_scales):
            l0 = 5.0
            l0_dot = 0.25

    class noise(WheelLeggedCfg.noise):
        class noise_scales(WheelLeggedCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1


class WheelLeggedVMCCfgPPO(WheelLeggedCfgPPO):

    class algorithm(WheelLeggedCfgPPO.algorithm):
        kl_decay = (
            WheelLeggedCfgPPO.algorithm.desired_kl - 0.002
        ) / WheelLeggedCfgPPO.runner.max_iterations

    class runner(WheelLeggedCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged_vmc"
