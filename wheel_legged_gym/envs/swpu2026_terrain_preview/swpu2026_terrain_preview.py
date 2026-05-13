from isaacgym import gymapi
from isaacgym import terrain_utils
import numpy as np
import torch
from types import SimpleNamespace

from wheel_legged_gym.envs.base.base_task import BaseTask
from wheel_legged_gym.utils.terrain import Terrain


class TerrainPreviewTask(BaseTask):
    """Viewer-only terrain preview task without robot actors."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self._parse_cfg(cfg)
        self.height_samples = None
        self.preview_terrain = None
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.reset_buf.zero_()
        self._set_preview_camera()

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.sim.dt
        self.task_terrain_names = list(getattr(cfg.terrain, "preview_task_names", []))

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_preview_terrains()
        self._create_envs()

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _create_preview_terrains(self):
        preview_cfgs = getattr(self.cfg.terrain, "preview_terrain_cfgs", None)
        if not preview_cfgs:
            raise ValueError("terrain preview requires preview_terrain_cfgs")
        source_terrains = [Terrain(terrain_cfg, 1) for terrain_cfg in preview_cfgs]
        first_cfg = preview_cfgs[0]
        mesh_type = first_cfg.mesh_type
        horizontal_scale = float(first_cfg.horizontal_scale)
        vertical_scale = float(first_cfg.vertical_scale)
        slope_treshold = float(first_cfg.slope_treshold)
        if any(
            terrain_cfg.mesh_type != mesh_type for terrain_cfg in preview_cfgs[1:]
        ):
            raise ValueError("preview terrains must use the same mesh type")
        if any(
            float(terrain_cfg.horizontal_scale) != horizontal_scale
            or float(terrain_cfg.vertical_scale) != vertical_scale
            for terrain_cfg in preview_cfgs[1:]
        ):
            raise ValueError("preview terrains must share horizontal/vertical scale")

        concat_axis_name = getattr(self.cfg.terrain, "preview_concat_axis", "y")
        if concat_axis_name not in ("x", "y"):
            raise ValueError("preview_concat_axis must be 'x' or 'y'")
        concat_axis = 0 if concat_axis_name == "x" else 1

        cores = []
        for terrain in source_terrains:
            border = int(getattr(terrain, "border", 0))
            core = terrain.height_field_raw
            if border > 0:
                core = core[border:-border, border:-border]
            cores.append(core)

        ref_shape = list(cores[0].shape)
        other_axis = 1 - concat_axis
        if any(core.shape[other_axis] != ref_shape[other_axis] for core in cores[1:]):
            raise ValueError("preview terrains must match on the non-concatenated axis")

        core_map = np.concatenate(cores, axis=concat_axis)
        outer_border = int(round(float(self.cfg.terrain.border_size) / horizontal_scale))
        total_rows = core_map.shape[0] + 2 * outer_border
        total_cols = core_map.shape[1] + 2 * outer_border
        height_field_raw = np.zeros((total_rows, total_cols), dtype=np.int16)
        height_field_raw[
            outer_border : outer_border + core_map.shape[0],
            outer_border : outer_border + core_map.shape[1],
        ] = core_map

        preview_cfg = SimpleNamespace(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            border_size=outer_border * horizontal_scale,
            slope_treshold=slope_treshold,
            static_friction=first_cfg.static_friction,
            dynamic_friction=first_cfg.dynamic_friction,
            restitution=first_cfg.restitution,
            mesh_type=mesh_type,
        )
        preview_terrain = SimpleNamespace(
            cfg=preview_cfg,
            border=outer_border,
            tot_rows=total_rows,
            tot_cols=total_cols,
            height_field_raw=height_field_raw,
            heightsamples=height_field_raw,
        )
        if mesh_type == "trimesh":
            (
                preview_terrain.vertices,
                preview_terrain.triangles,
            ) = terrain_utils.convert_heightfield_to_trimesh(
                height_field_raw,
                horizontal_scale,
                vertical_scale,
                slope_treshold,
            )
        self.preview_terrain = preview_terrain
        self.preview_total_length = total_rows * horizontal_scale
        self.preview_total_width = total_cols * horizontal_scale

        if mesh_type == "heightfield":
            self._add_heightfield(preview_terrain)
        elif mesh_type == "trimesh":
            self._add_trimesh(preview_terrain)
        elif mesh_type == "plane":
            self._add_ground_plane(preview_cfg)
        else:
            raise ValueError(f"Unsupported terrain mesh type for preview: {mesh_type}")

    def _create_envs(self):
        lower = gymapi.Vec3(-1.0, -1.0, -1.0)
        upper = gymapi.Vec3(1.0, 1.0, 1.0)
        self.envs = []
        self.env_origins = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float
        )
        for _ in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, lower, upper, 1)
            self.envs.append(env_handle)

    def _add_ground_plane(self, terrain_cfg):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = terrain_cfg.static_friction
        plane_params.dynamic_friction = terrain_cfg.dynamic_friction
        plane_params.restitution = terrain_cfg.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _add_heightfield(self, terrain, x_offset=0.0, y_offset=0.0):
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = terrain.cfg.horizontal_scale
        hf_params.row_scale = terrain.cfg.horizontal_scale
        hf_params.vertical_scale = terrain.cfg.vertical_scale
        hf_params.nbRows = terrain.tot_cols
        hf_params.nbColumns = terrain.tot_rows
        hf_params.transform.p.x = -terrain.cfg.border_size + x_offset
        hf_params.transform.p.y = -terrain.cfg.border_size + y_offset
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = terrain.cfg.static_friction
        hf_params.dynamic_friction = terrain.cfg.dynamic_friction
        hf_params.restitution = terrain.cfg.restitution
        self.gym.add_heightfield(self.sim, terrain.heightsamples, hf_params)

    def _add_trimesh(self, terrain, x_offset=0.0, y_offset=0.0):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = terrain.vertices.shape[0]
        tm_params.nb_triangles = terrain.triangles.shape[0]
        tm_params.transform.p.x = -terrain.cfg.border_size + x_offset
        tm_params.transform.p.y = -terrain.cfg.border_size + y_offset
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = terrain.cfg.static_friction
        tm_params.dynamic_friction = terrain.cfg.dynamic_friction
        tm_params.restitution = terrain.cfg.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            terrain.vertices.flatten(order="C"),
            terrain.triangles.flatten(order="C"),
            tm_params,
        )

    def _set_preview_camera(self):
        if self.viewer is None or self.preview_terrain is None:
            return
        start_x = -self.preview_terrain.cfg.border_size
        end_x = (
            self.preview_terrain.tot_rows * self.preview_terrain.cfg.horizontal_scale
            - self.preview_terrain.cfg.border_size
        )
        center_x = 0.5 * (start_x + end_x)
        center_y = 0.5 * self.preview_total_width - self.preview_terrain.cfg.border_size
        length = max(1.0, self.preview_total_length)
        width = max(1.0, self.preview_total_width)
        cam_pos = np.array(
            [
                center_x - 0.45 * length,
                center_y - 1.10 * width,
                max(length, width) * 0.95,
            ],
            dtype=np.float64,
        )
        cam_target = np.array([center_x, center_y, 0.0], dtype=np.float64)
        self.set_camera(cam_pos, cam_target)

    def step(self, actions):
        self.render()
        self.gym.simulate(self.sim)
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
        )

    def reset_idx(self, env_ids):
        return
