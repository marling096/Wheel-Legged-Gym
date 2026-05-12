from wheel_legged_gym.envs.swpu2026_vmc_flat.swpu2026_vmc_flat_config import (
    Swpu2026VMCFlatCfg,
    Swpu2026VMCFlatCfgPPO,
)


class Swpu2026VMCUndulatingCfg(Swpu2026VMCFlatCfg):
    """SWPU2026 VMC task on repeated rounded speed-bump terrain."""

    class env(Swpu2026VMCFlatCfg.env):
        num_envs = 1024

    class terrain(Swpu2026VMCFlatCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        curriculum = False
        selected = False
        undulating_terrain = True
        horizontal_scale = 0.10
        border_size = 8.0

        undulating_profile = "speed_bumps"
        undulating_amplitude = 0.13
        undulating_bump_spacing = 1.15
        undulating_bump_width = 0.85
        undulating_lateral_amplitude = 0.015
        undulating_wavelength_y = 4.0
        undulating_cross_weight = 0.0
        undulating_randomize_cell = True
        undulating_wavelength_jitter = 0.12
        undulating_amplitude_jitter = 0.12

        num_rows = 5
        num_cols = 5

    class viewer(Swpu2026VMCFlatCfg.viewer):
        draw_terrain_contours = True
        terrain_contour_range = 6.0
        terrain_contour_stride = 2
        terrain_contour_max_lines = 1100
        terrain_contour_height_offset = 0.025


class Swpu2026VMCUndulatingCfgPPO(Swpu2026VMCFlatCfgPPO):
    class runner(Swpu2026VMCFlatCfgPPO.runner):
        experiment_name = "swpu2026_vmc_undulating"
        max_iterations = 500
