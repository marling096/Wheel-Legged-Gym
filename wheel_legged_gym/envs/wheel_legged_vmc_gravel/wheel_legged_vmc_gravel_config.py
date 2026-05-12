from wheel_legged_gym.envs.wheel_legged_vmc_undulating.wheel_legged_vmc_undulating_config import (
    WheelLeggedVMCUndulatingCfg,
    WheelLeggedVMCUndulatingCfgPPO,
)


class WheelLeggedVMCGravelCfg(WheelLeggedVMCUndulatingCfg):
    """Wheel-legged VMC task on dense blocky gravel terrain."""

    class terrain(WheelLeggedVMCUndulatingCfg.terrain):
        undulating_profile = "gravel"
        undulating_randomize_cell = True
        gravel_roughness = 0.014
        gravel_block_size = 0.18
        gravel_block_height_range = (-0.035, 0.065)
        gravel_block_slope = 0.014
        gravel_stone_density = 1.0
        gravel_stone_height_range = (0.006, 0.035)
        gravel_stone_radius_range = (0.045, 0.13)
        gravel_rut_depth = 0.012
        gravel_rut_width = 0.40


class WheelLeggedVMCGravelCfgPPO(WheelLeggedVMCUndulatingCfgPPO):
    class runner(WheelLeggedVMCUndulatingCfgPPO.runner):
        experiment_name = "wheel_legged_vmc_gravel"
        max_iterations = 500
