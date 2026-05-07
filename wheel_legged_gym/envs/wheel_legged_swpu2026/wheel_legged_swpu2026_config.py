# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from wheel_legged_gym.envs.wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)


class WheelLeggedSwpu2026Cfg(WheelLeggedVMCFlatCfg):
    """Flat terrain + swpu2026 URDF. Joint names differ from wl (wheel: left/right_wheel_joint)."""

    class init_state(WheelLeggedVMCFlatCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            "lf0_Joint": 0.5,
            "lf1_Joint": 0.35,
            "left_wheel_joint": 0.0,
            "rf0_Joint": -0.5,
            "rf1_Joint": -0.35,
            "right_wheel_joint": 0.0,
        }

    class asset(WheelLeggedVMCFlatCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/swpu2026/urdf/swpu2026_isaac.urdf"
        foot_name = "Wheel"
        # URDF 关节原点间距近似（髋-膝 / 膝-轮），用于 VMC 雅可比；与 wl 默认 0.15/0.25 不一致。
        offset = 0.02
        l1 = 0.210
        l2 = 0.254

    class control(WheelLeggedVMCFlatCfg.control):
        # 虚拟腿几何修正后略加强刚度，并在 VMC 映射前抵抗俯仰/横滚（否则易出现“跪着不倒、不直立”）。
        kp_theta = 65.0
        kd_theta = 4.0
        vmc_leg_pitch_stab_kp = 72.0
        vmc_leg_pitch_stab_kd = 5.0
        vmc_leg_roll_stab_kp = 55.0
        vmc_leg_roll_stab_kd = 4.0

    class rewards(WheelLeggedVMCFlatCfg.rewards):
        class scales(WheelLeggedVMCFlatCfg.rewards.scales):
            orientation = -18.0


class WheelLeggedSwpu2026CfgPPO(WheelLeggedVMCFlatCfgPPO):
    class algorithm(WheelLeggedVMCFlatCfgPPO.algorithm):
        learning_rate = 3.0e-4
        extra_learning_rate = 3.0e-4
        value_loss_coef = 0.5

    class runner(WheelLeggedVMCFlatCfgPPO.runner):
        experiment_name = "wheel_legged_swpu2026"
