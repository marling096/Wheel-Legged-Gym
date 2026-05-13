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

import numpy as np
from numpy.random import choice
from scipy import interpolate
from typing import TYPE_CHECKING

from isaacgym import terrain_utils

if TYPE_CHECKING:
    from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Terrain:
    def __init__(self, cfg: "LeggedRobotCfg.terrain", num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if getattr(cfg, "undulating_terrain", False):
            self.undulating_terrain_fill()
        elif cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = (
                terrain_utils.convert_heightfield_to_trimesh(
                    self.height_field_raw,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.slope_treshold,
                )
            )

    def undulating_terrain_fill(self):
        """Fill the heightfield with the configured undulating profile per cell."""
        cfg = self.cfg
        profile = getattr(cfg, "undulating_profile", "wave")
        amp = getattr(cfg, "undulating_amplitude", 0.06)
        wl_x = getattr(
            cfg,
            "undulating_bump_spacing",
            getattr(cfg, "undulating_wavelength_x", 4.0),
        )
        wl_y = getattr(cfg, "undulating_wavelength_y", 4.0)
        bump_width = getattr(cfg, "undulating_bump_width", 0.8)
        lateral_amp = getattr(cfg, "undulating_lateral_amplitude", 0.0)
        cross = getattr(cfg, "undulating_cross_weight", 0.45)
        rand_cell = getattr(cfg, "undulating_randomize_cell", True)
        wl_jit = getattr(cfg, "undulating_wavelength_jitter", 0.35)
        amp_jit = getattr(cfg, "undulating_amplitude_jitter", 0.0)
        patch_length = getattr(cfg, "undulating_patch_length", None)
        patch_width = getattr(cfg, "undulating_patch_width", None)
        patch_center_x = getattr(cfg, "undulating_patch_center_x", None)
        patch_center_y = getattr(cfg, "undulating_patch_center_y", None)
        patch_falloff = getattr(cfg, "undulating_patch_falloff", 0.0)
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale,
            )
            if rand_cell:
                phx = float(np.random.uniform(0.0, 2.0 * np.pi))
                phy = float(np.random.uniform(0.0, 2.0 * np.pi))
                jx = float(np.random.uniform(-wl_jit, wl_jit))
                jy = float(np.random.uniform(-wl_jit, wl_jit))
                ja = float(np.random.uniform(-amp_jit, amp_jit))
                wx = max(0.25, wl_x * (1.0 + jx))
                wy = max(0.25, wl_y * (1.0 + jy))
                amp_cell = max(0.0, amp * (1.0 + ja))
            else:
                phx = phy = 0.0
                wx, wy = wl_x, wl_y
                amp_cell = amp
            if profile in ["speed_bumps", "speed_bumps_gravel"]:
                speed_bump_terrain(
                    terrain,
                    amplitude=amp_cell,
                    spacing=wx,
                    bump_width=bump_width,
                    phase_x=phx,
                    lateral_amplitude=lateral_amp,
                    lateral_wavelength=wy,
                    phase_y=phy,
                    patch_length=patch_length,
                    patch_width=patch_width,
                    patch_center_x=patch_center_x,
                    patch_center_y=patch_center_y,
                    patch_falloff=patch_falloff,
                )
                if profile == "speed_bumps_gravel":
                    gravel_road_terrain(
                        terrain,
                        roughness=getattr(cfg, "gravel_roughness", 0.018),
                        block_size=getattr(cfg, "gravel_block_size", 0.18),
                        block_height_range=getattr(
                            cfg, "gravel_block_height_range", (-0.035, 0.055)
                        ),
                        block_slope=getattr(cfg, "gravel_block_slope", 0.012),
                        stone_density=getattr(cfg, "gravel_stone_density", 1.8),
                        stone_height_range=getattr(
                            cfg, "gravel_stone_height_range", (0.015, 0.065)
                        ),
                        stone_radius_range=getattr(
                            cfg, "gravel_stone_radius_range", (0.06, 0.20)
                        ),
                        rut_depth=getattr(cfg, "gravel_rut_depth", 0.012),
                        rut_width=getattr(cfg, "gravel_rut_width", 0.38),
                    )
            elif profile == "gravel":
                gravel_road_terrain(
                    terrain,
                    roughness=getattr(cfg, "gravel_roughness", 0.025),
                    block_size=getattr(cfg, "gravel_block_size", 0.18),
                    block_height_range=getattr(
                        cfg, "gravel_block_height_range", (-0.035, 0.060)
                    ),
                    block_slope=getattr(cfg, "gravel_block_slope", 0.014),
                    stone_density=getattr(cfg, "gravel_stone_density", 2.4),
                    stone_height_range=getattr(
                        cfg, "gravel_stone_height_range", (0.015, 0.075)
                    ),
                    stone_radius_range=getattr(
                        cfg, "gravel_stone_radius_range", (0.06, 0.22)
                    ),
                    rut_depth=getattr(cfg, "gravel_rut_depth", 0.018),
                    rut_width=getattr(cfg, "gravel_rut_width", 0.42),
                )
            else:
                undulating_wave_terrain(
                    terrain,
                    amplitude=amp_cell,
                    wavelength_x=wx,
                    wavelength_y=wy,
                    phase_x=phx,
                    phase_y=phy,
                    cross_weight=cross,
                )
            self.add_terrain_to_map(terrain, i, j)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        slope = difficulty * 0.5
        random_height = 0.05 + difficulty * 0.05
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.1
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        if choice < self.proportions[0]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.0)
        elif choice < self.proportions[1]:
            if (
                choice
                < self.proportions[0] + (self.proportions[1] - self.proportions[0]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
        elif choice < self.proportions[2]:
            if (
                choice
                < self.proportions[1] + (self.proportions[2] - self.proportions[1]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope * 0.5, platform_size=3.0
            )
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-random_height,
                max_height=random_height,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[4]:
            if choice < self.proportions[3]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.7, step_height=step_height, platform_size=4.0
            )
        elif choice < self.proportions[5]:
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[6]:
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.proportions[7]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = (
            np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def undulating_wave_terrain(
    terrain,
    amplitude,
    wavelength_x,
    wavelength_y,
    phase_x=0.0,
    phase_y=0.0,
    cross_weight=0.45,
):
    """Smooth rolling bumps in meters; writes ``terrain.height_field_raw`` (vertical-scale units)."""
    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale
    ix = np.arange(terrain.length, dtype=np.float64)[:, None]
    iy = np.arange(terrain.width, dtype=np.float64)[None, :]
    x_m = ix * hs
    y_m = iy * hs
    kx = 2.0 * np.pi / max(float(wavelength_x), 0.25)
    ky = 2.0 * np.pi / max(float(wavelength_y), 0.25)
    cw = float(cross_weight)
    norm = 1.0 + abs(cw)
    z_m = float(amplitude) * (
        np.sin(kx * x_m + phase_x) + cw * np.sin(ky * y_m + phase_y)
    ) / norm
    terrain.height_field_raw[:, :] = np.clip(
        np.rint(z_m / vs).astype(np.int16), -32768, 32767
    )


def speed_bump_terrain(
    terrain,
    amplitude,
    spacing,
    bump_width,
    phase_x=0.0,
    lateral_amplitude=0.0,
    lateral_wavelength=4.0,
    phase_y=0.0,
    patch_length=None,
    patch_width=None,
    patch_center_x=None,
    patch_center_y=None,
    patch_falloff=0.0,
):
    """Repeated rounded speed bumps in meters; ridges run across the y axis."""
    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale
    ix = np.arange(terrain.length, dtype=np.float64)[:, None]
    iy = np.arange(terrain.width, dtype=np.float64)[None, :]
    x_m = ix * hs
    y_m = iy * hs
    spacing = max(float(spacing), 0.25)
    half_width = max(float(bump_width) * 0.5, hs)
    phase_offset = (float(phase_x) / (2.0 * np.pi)) * spacing
    nearest_center = np.rint((x_m - phase_offset) / spacing) * spacing + phase_offset
    dist = np.abs(x_m - nearest_center)
    bump = np.where(
        dist <= half_width,
        0.5 * (1.0 + np.cos(np.pi * dist / half_width)),
        0.0,
    )
    z_m = float(amplitude) * bump
    if lateral_amplitude != 0.0:
        ky = 2.0 * np.pi / max(float(lateral_wavelength), 0.25)
        z_m = z_m + float(lateral_amplitude) * np.sin(ky * y_m + phase_y)
    if patch_length is not None or patch_width is not None:
        center_x = (
            terrain.length * hs * 0.5
            if patch_center_x is None
            else float(patch_center_x)
        )
        center_y = (
            terrain.width * hs * 0.5 if patch_center_y is None else float(patch_center_y)
        )
        half_length = (
            terrain.length * hs if patch_length is None else float(patch_length)
        ) * 0.5
        half_width_patch = (
            terrain.width * hs if patch_width is None else float(patch_width)
        ) * 0.5
        dx_edge = half_length - np.abs(x_m - center_x)
        dy_edge = half_width_patch - np.abs(y_m - center_y)
        inside_margin = np.minimum(dx_edge, dy_edge)
        falloff = max(float(patch_falloff), 0.0)
        if falloff > 0.0:
            patch_mask = np.clip(inside_margin / falloff, 0.0, 1.0)
        else:
            patch_mask = (inside_margin >= 0.0).astype(np.float64)
        z_m = z_m * patch_mask
    terrain.height_field_raw[:, :] = np.clip(
        np.rint(z_m / vs).astype(np.int16), -32768, 32767
    )


def gravel_road_terrain(
    terrain,
    roughness=0.025,
    block_size=0.18,
    block_height_range=(-0.035, 0.060),
    block_slope=0.014,
    stone_density=2.4,
    stone_height_range=(0.015, 0.075),
    stone_radius_range=(0.06, 0.22),
    rut_depth=0.018,
    rut_width=0.42,
):
    """Dense blocky gravel road with low-poly facets and shallow ruts."""
    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale
    ix = np.arange(terrain.length, dtype=np.float64)[:, None]
    iy = np.arange(terrain.width, dtype=np.float64)[None, :]
    x_m = ix * hs
    y_m = iy * hs
    z_m = terrain.height_field_raw.astype(np.float64) * vs

    coarse_step = max(2, int(0.45 / hs))
    coarse_shape = (
        max(2, int(np.ceil(terrain.length / coarse_step)) + 1),
        max(2, int(np.ceil(terrain.width / coarse_step)) + 1),
    )
    coarse_x = np.linspace(0.0, terrain.length - 1, coarse_shape[0])
    coarse_y = np.linspace(0.0, terrain.width - 1, coarse_shape[1])
    fine_x = np.arange(terrain.length, dtype=np.float64)
    fine_y = np.arange(terrain.width, dtype=np.float64)
    coarse_noise = np.random.uniform(-1.0, 1.0, coarse_shape)
    interp = interpolate.RectBivariateSpline(
        coarse_x, coarse_y, coarse_noise, kx=1, ky=1
    )
    z_m += float(roughness) * interp(fine_x, fine_y)

    width_m = terrain.width * hs
    length_m = terrain.length * hs
    block_cells = max(1, int(round(float(block_size) / hs)))
    block_rows = int(np.ceil(terrain.length / block_cells))
    block_cols = int(np.ceil(terrain.width / block_cells))
    block_min, block_max = block_height_range
    block_heights = np.random.uniform(block_min, block_max, (block_rows, block_cols))
    block_heights += np.random.normal(0.0, float(roughness) * 0.35, block_heights.shape)
    block_heights = np.clip(block_heights, block_min, block_max)
    block_heights = np.repeat(
        np.repeat(block_heights, block_cells, axis=0), block_cells, axis=1
    )
    block_heights = block_heights[: terrain.length, : terrain.width]

    local_x = (np.arange(terrain.length) % block_cells).astype(np.float64)
    local_y = (np.arange(terrain.width) % block_cells).astype(np.float64)
    local_x = (local_x / max(block_cells - 1, 1) - 0.5)[:, None]
    local_y = (local_y / max(block_cells - 1, 1) - 0.5)[None, :]
    slope_x = np.random.uniform(-1.0, 1.0, (block_rows, block_cols))
    slope_y = np.random.uniform(-1.0, 1.0, (block_rows, block_cols))
    slope_x = np.repeat(np.repeat(slope_x, block_cells, axis=0), block_cells, axis=1)
    slope_y = np.repeat(np.repeat(slope_y, block_cells, axis=0), block_cells, axis=1)
    slope_x = slope_x[: terrain.length, : terrain.width]
    slope_y = slope_y[: terrain.length, : terrain.width]
    facets = float(block_slope) * (slope_x * local_x + slope_y * local_y)
    z_m += block_heights + facets

    area_m2 = width_m * length_m
    num_stones = max(1, int(float(stone_density) * area_m2))
    h_min, h_max = stone_height_range
    r_min, r_max = stone_radius_range
    for _ in range(num_stones):
        cx = float(np.random.uniform(0.0, length_m))
        cy = float(np.random.uniform(0.0, width_m))
        radius = float(np.random.uniform(r_min, r_max))
        height = float(np.random.uniform(h_min, h_max))
        if np.random.uniform() < 0.18:
            height *= -0.45
        ix0 = max(0, int((cx - radius) / hs))
        ix1 = min(terrain.length, int((cx + radius) / hs) + 1)
        iy0 = max(0, int((cy - radius) / hs))
        iy1 = min(terrain.width, int((cy + radius) / hs) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        dx = x_m[ix0:ix1] - cx
        dy = y_m[:, iy0:iy1] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        stone = np.where(
            dist <= radius,
            0.5 * height * (1.0 + np.cos(np.pi * dist / radius)),
            0.0,
        )
        z_m[ix0:ix1, iy0:iy1] += stone

    if rut_depth > 0.0:
        center_y = 0.5 * width_m
        track_half_gap = min(0.42, width_m * 0.18)
        sigma = max(float(rut_width) * 0.5, hs)
        ruts = np.exp(-0.5 * ((y_m - (center_y - track_half_gap)) / sigma) ** 2)
        ruts += np.exp(-0.5 * ((y_m - (center_y + track_half_gap)) / sigma) ** 2)
        longitudinal = 0.65 + 0.35 * np.sin(2.0 * np.pi * x_m / 2.8)
        z_m -= float(rut_depth) * longitudinal * ruts

    terrain.height_field_raw[:, :] = np.clip(
        np.rint(z_m / vs).astype(np.int16), -32768, 32767
    )


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
