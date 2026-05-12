from wheel_legged_gym.envs.swpu2026_vmc_undulating.swpu2026_vmc_undulating_config import (
    Swpu2026VMCUndulatingCfg,
    Swpu2026VMCUndulatingCfgPPO,
)


class Swpu2026VMCGravelCfg(Swpu2026VMCUndulatingCfg):
    """SWPU2026 VMC task on dense blocky gravel terrain."""

    class terrain(Swpu2026VMCUndulatingCfg.terrain):
        undulating_profile = "gravel"
        undulating_randomize_cell = True
        gravel_roughness = 0.012
        gravel_block_size = 0.20
        gravel_block_height_range = (-0.030, 0.055)
        gravel_block_slope = 0.012
        gravel_stone_density = 0.8
        gravel_stone_height_range = (0.006, 0.030)
        gravel_stone_radius_range = (0.045, 0.12)
        gravel_rut_depth = 0.010
        gravel_rut_width = 0.38


class Swpu2026VMCGravelCfgPPO(Swpu2026VMCUndulatingCfgPPO):
    class runner(Swpu2026VMCUndulatingCfgPPO.runner):
        experiment_name = "swpu2026_vmc_gravel"
        max_iterations = 500
