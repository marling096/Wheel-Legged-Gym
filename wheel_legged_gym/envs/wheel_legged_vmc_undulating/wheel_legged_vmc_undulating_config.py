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

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
)
from wheel_legged_gym.envs.wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)


class WheelLeggedVMCUndulatingCfg(WheelLeggedVMCFlatCfg):
    """与 ``wheel_legged_vmc_flat`` 相同机器人与控制默认，地面为三角网格连续减速带。"""

    class env(WheelLeggedVMCFlatCfg.env):
        # ~4GB VRAM：4096 并行环境 + 单块 trimesh 起伏地（PhysX/GPU 缓冲）易 OOM；flat 为 plane 不占此项
        num_envs = 1024
        fail_to_terminal_time_s = 0.2
        min_base_height = 0.08
        min_leg_length = 0.11

    class terrain(WheelLeggedVMCCfg.terrain):
        mesh_type = "heightfield"  # 比 trimesh 碰撞检测更快，兼容同样高度场函数
        measure_heights = True
        curriculum = False
        selected = False
        undulating_terrain = True
        horizontal_scale = 0.10
        # 3×3 格 × 3m + 2×1.8m 边框 = 12.6m 边长 = 158.8 m² ≈ 原 3136 m² 的 1/20
        border_size = 1.8
        terrain_length = 1.0
        terrain_width = 1.0
        num_rows = 1
        num_cols = 2
        # 连续圆顶减速带：沿 x 方向反复出现，脊线横跨 y 方向。
        undulating_profile = "speed_bumps"
        undulating_amplitude = 0.13
        undulating_bump_spacing = 1.15
        undulating_bump_width = 0.85
        # 轻微横向起伏用于破除完全直线地形，但主视觉仍是多条并排减速带。
        undulating_lateral_amplitude = 0.015
        undulating_wavelength_y = 4.0
        undulating_cross_weight = 0.0
        undulating_randomize_cell = True
        undulating_wavelength_jitter = 0.12
        undulating_amplitude_jitter = 0.12

    class viewer(WheelLeggedVMCFlatCfg.viewer):
        draw_terrain_contours = False  # 每帧绘制大量等高线会严重拖慢仿真
        terrain_contour_range = 3.0
        terrain_contour_stride = 3
        terrain_contour_max_lines = 400
        terrain_contour_height_offset = 0.025

    class rewards(WheelLeggedVMCFlatCfg.rewards):
        min_vmc_leg_length = 0.12

        class scales(WheelLeggedVMCFlatCfg.rewards.scales):
            base_height = 2.0
            vmc_nominal_leg = -0.6
            vmc_leg_collapse = -25.0


class WheelLeggedVMCUndulatingCfgPPO(WheelLeggedVMCFlatCfgPPO):
    class runner(WheelLeggedVMCFlatCfgPPO.runner):
        experiment_name = "wheel_legged_vmc_undulating"
        max_iterations = 1000
