# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""swpu2026 专用 VMC：髋/膝为绕 X 轴转动，与 wl 的平面腿约定不同，需单独映射 θ1/θ2。"""

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC


class LeggedRobotVMCSwpu2026(LeggedRobotVMC):
    def leg_post_physics_step(self):
        # wl 使用膝角 +π/2 与固定 l1/l2 匹配 swpu2026 的 URDF 零位不一致；此处直接用关节角作为 2R 模型内角。
        self.theta1 = torch.cat(
            (self.dof_pos[:, 0].unsqueeze(1), -self.dof_pos[:, 3].unsqueeze(1)), dim=1
        )
        self.theta2 = torch.cat(
            (self.dof_pos[:, 1].unsqueeze(1), -self.dof_pos[:, 4].unsqueeze(1)), dim=1
        )
        theta1_dot = torch.cat(
            (self.dof_vel[:, 0].unsqueeze(1), -self.dof_vel[:, 3].unsqueeze(1)), dim=1
        )
        theta2_dot = torch.cat(
            (self.dof_vel[:, 1].unsqueeze(1), -self.dof_vel[:, 4].unsqueeze(1)), dim=1
        )

        self.L0, self.theta0 = self.forward_kinematics(self.theta1, self.theta2)
        self.L0 = torch.nan_to_num(self.L0, nan=0.2, posinf=2.0, neginf=0.05)
        self.theta0 = torch.nan_to_num(self.theta0, nan=0.0, posinf=3.14, neginf=-3.14)

        dt = 0.001
        L0_temp, theta0_temp = self.forward_kinematics(
            self.theta1 + theta1_dot * dt, self.theta2 + theta2_dot * dt
        )
        self.L0_dot = (L0_temp - self.L0) / dt
        self.theta0_dot = (theta0_temp - self.theta0) / dt
        self.L0_dot = torch.nan_to_num(self.L0_dot, nan=0.0, posinf=1e3, neginf=-1e3)
        self.theta0_dot = torch.nan_to_num(
            self.theta0_dot, nan=0.0, posinf=1e3, neginf=-1e3
        )
