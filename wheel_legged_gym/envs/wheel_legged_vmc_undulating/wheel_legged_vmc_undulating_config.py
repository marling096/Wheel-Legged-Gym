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
    """与 ``wheel_legged_vmc_flat`` 相同机器人与控制默认，地面为三角网格起伏波面。"""

    class env(WheelLeggedVMCFlatCfg.env):
        # ~4GB VRAM：4096 并行环境 + 单块 trimesh 起伏地（PhysX/GPU 缓冲）易 OOM；flat 为 plane 不占此项
        num_envs = 1024
        fail_to_terminal_time_s = 0.2
        min_base_height = 0.08
        min_leg_length = 0.11

    class terrain(WheelLeggedVMCCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        curriculum = False
        selected = False
        undulating_terrain = True
        # 更粗水平网格 + 更小边界：显著减少 heightfield 像素与三角面片（见 terrain.Terrain 中 tot_rows/tot_cols）
        horizontal_scale = 0.15
        border_size = 10.0
        # 波幅与波长（米）；cross 为 y 向正弦叠加权重；每块子地形可随机相位/波长
        undulating_amplitude = 0.07
        undulating_wavelength_x = 3.5
        undulating_wavelength_y = 3.5
        undulating_cross_weight = 0.40
        undulating_randomize_cell = True
        undulating_wavelength_jitter = 0.30
        num_rows = 5
        num_cols = 5

    class viewer(WheelLeggedVMCFlatCfg.viewer):
        draw_terrain_contours = True
        terrain_contour_range = 4.5
        terrain_contour_stride = 3
        terrain_contour_max_lines = 420
        terrain_contour_height_offset = 0.015

    class rewards(WheelLeggedVMCFlatCfg.rewards):
        min_vmc_leg_length = 0.12

        class scales(WheelLeggedVMCFlatCfg.rewards.scales):
            base_height = 2.0
            vmc_nominal_leg = -0.6
            vmc_leg_collapse = -25.0


class WheelLeggedVMCUndulatingCfgPPO(WheelLeggedVMCFlatCfgPPO):
    class runner(WheelLeggedVMCFlatCfgPPO.runner):
        experiment_name = "wheel_legged_vmc_undulating"
        max_iterations = 500
