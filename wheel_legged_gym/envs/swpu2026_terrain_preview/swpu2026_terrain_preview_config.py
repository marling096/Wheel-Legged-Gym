import copy

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)
from wheel_legged_gym.envs.swpu2026_vmc_gravel.swpu2026_vmc_gravel_config import (
    Swpu2026VMCGravelCfg,
)
from wheel_legged_gym.envs.swpu2026_vmc_undulating.swpu2026_vmc_undulating_config import (
    Swpu2026VMCUndulatingCfg,
)


def _make_preview_source_terrain(cfg_cls):
    terrain = copy.deepcopy(cfg_cls().terrain)
    terrain.num_rows = 5
    terrain.num_cols = 2
    terrain.terrain_length = 4.0
    terrain.terrain_width = 4.0
    terrain.border_size = 0.0
    terrain.curriculum = False
    terrain.selected = False
    return terrain


class Swpu2026TerrainPreviewCfg(LeggedRobotCfg):
    def __init__(self):
        super().__init__()
        self.terrain.preview_task_names = [
            "swpu2026_vmc_undulating",
            "swpu2026_vmc_gravel",
        ]
        self.terrain.preview_terrain_cfgs = [
            _make_preview_source_terrain(Swpu2026VMCUndulatingCfg),
            _make_preview_source_terrain(Swpu2026VMCGravelCfg),
        ]

    class env(LeggedRobotCfg.env):
        num_envs = 1
        num_observations = 0
        num_privileged_obs = None
        obs_history_length = 1
        num_actions = 0
        env_spacing = 0.0
        episode_length_s = 60.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "heightfield"
        measure_heights = False
        curriculum = False
        selected = False
        num_rows = 1
        num_cols = 1
        border_size = 0.5
        preview_concat_axis = "y"
        preview_task_names = []
        preview_terrain_cfgs = []

    class viewer(LeggedRobotCfg.viewer):
        horizontal_fov = 90.0
        width = 1600
        height = 900


class Swpu2026TerrainPreviewCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "swpu2026_terrain_preview"
        max_iterations = 1
