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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from isaacgym import gymapi
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import math

# 脚本直接运行时可在 __main__ 中改为 True；若从其它模块 import play()，请先设置该变量
USE_KEYBOARD_TELEOP = False

# RL play 模式（非 --lqr_demo）每步写入的默认高度指令
PLAY_DEFAULT_HEIGHT_CMD = 0.18
# lqr_demo 目标高度（按 RL 可达区先设为 0.10）
LQR_DEMO_HEIGHT_CMD = 0.10


def nominal_virtual_leg_from_cfg(env_cfg):
    """Compute nominal (L0, theta0) in the same coordinates used by VMC env."""
    th1 = float(env_cfg.init_state.default_joint_angles["lf0_Joint"])
    th2 = float(env_cfg.init_state.default_joint_angles["lf1_Joint"]) + math.pi / 2.0
    end_x = (
        float(env_cfg.asset.offset)
        + float(env_cfg.asset.l1) * math.cos(th1)
        + float(env_cfg.asset.l2) * math.cos(th1 + th2)
    )
    end_y = float(env_cfg.asset.l1) * math.sin(th1) + float(env_cfg.asset.l2) * math.sin(
        th1 + th2
    )
    L0 = math.sqrt(end_x * end_x + end_y * end_y)
    theta0 = math.atan2(end_y, end_x) - math.pi / 2.0
    return L0, theta0


def install_keyboard_teleop(env, env_cfg):
    """Subscribe viewer keys and drive env.commands (vx, heading target, height).

    Commands layout matches LeggedRobot._resample_commands:
      [:, 0] lin_vel_x, [:, 1] ang_vel_yaw (overwritten from heading when heading_command),
      [:, 2] height, [:, 3] heading target [rad].

    Requires viewer (non-headless). Callbacks run inside BaseTask.render during env.step().
    """

    vx_min, vx_max = env_cfg.commands.ranges.lin_vel_x
    h_min, h_max = env_cfg.commands.ranges.height
    height_min = float(env_cfg.commands.ranges.height[0])
    height_max = float(env_cfg.commands.ranges.height[1])
    # 单次按下直接给定指令幅值（不再逐级换挡）
    vx_forward = float(np.clip(vx_max, vx_min, vx_max))
    vx_reverse = float(np.clip(vx_min, vx_min, vx_max))
    dheading = 0.5  # rad / 次，目标航向单次跳动（仍属姿态指令而非速度档位）

    forward = quat_apply(env.base_quat[0:1], env.forward_vec[0:1])
    state = {
        "vx": float(torch.clip(env.commands[0, 0], vx_min, vx_max).cpu()),
        "height": float(torch.clip(env.commands[0, 2], h_min, h_max).cpu()),
        "heading": float(torch.atan2(forward[:, 1], forward[:, 0]).squeeze().cpu()),
    }

    def wrap_pi(a):
        return float(np.arctan2(np.sin(a), np.cos(a)))

    def bump_heading(delta):
        state["heading"] = wrap_pi(state["heading"] + delta)

    callbacks = {}

    def bind(key_code, action_name, fn):
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key_code, action_name)
        callbacks[action_name] = fn

    def on_press(evt, fn):
        if evt.value > 0:
            fn()

    bind(
        gymapi.KEY_W,
        "teleop_vx_fwd",
        lambda e: on_press(e, lambda: state.update({"vx": vx_forward})),
    )
    bind(
        gymapi.KEY_S,
        "teleop_vx_rev",
        lambda e: on_press(e, lambda: state.update({"vx": vx_reverse})),
    )
    bind(
        gymapi.KEY_UP,
        "teleop_vx_fwd_arw",
        lambda e: on_press(e, lambda: state.update({"vx": vx_forward})),
    )
    bind(
        gymapi.KEY_DOWN,
        "teleop_vx_rev_arw",
        lambda e: on_press(e, lambda: state.update({"vx": vx_reverse})),
    )
    bind(
        gymapi.KEY_A,
        "teleop_heading_left",
        lambda e: on_press(e, lambda: bump_heading(dheading)),
    )
    bind(
        gymapi.KEY_D,
        "teleop_heading_right",
        lambda e: on_press(e, lambda: bump_heading(-dheading)),
    )
    bind(
        gymapi.KEY_LEFT,
        "teleop_heading_left_arw",
        lambda e: on_press(e, lambda: bump_heading(dheading)),
    )
    bind(
        gymapi.KEY_RIGHT,
        "teleop_heading_right_arw",
        lambda e: on_press(e, lambda: bump_heading(-dheading)),
    )
    bind(
        gymapi.KEY_Q,
        "teleop_height_high",
        lambda e: on_press(e, lambda: state.update({"height": h_max})),
    )
    bind(
        gymapi.KEY_E,
        "teleop_height_low",
        lambda e: on_press(e, lambda: state.update({"height": h_min})),
    )
    bind(
        gymapi.KEY_SPACE,
        "teleop_stop_vx",
        lambda e: on_press(e, lambda: state.update({"vx": 0.0})),
    )

    env.viewer_keyboard_callbacks = callbacks
    env._keyboard_teleop_state = state

    print(
        "键盘遥操作 (需聚焦仿真窗口):\n"
        f"  W/S 或 ↑/↓ : 线速度指令一次到位 "
        f"(前进={vx_forward:.2f} m/s, 后退={vx_reverse:.2f} m/s，取自 cfg.commands.ranges.lin_vel_x)\n"
        "  A/D 或 ←/→ : 目标航向单次转动一步（heading_command=True 时生效）\n"
        f"  Q/E        : 高度指令为高限{height_max:.2f} / 低限{height_min:.2f}（cfg.commands.ranges.height）\n"
        "  Space      : 线速度归零\n"
        "  V          : 切换 viewer 同步（原有） Esc : 退出\n"
    )


def apply_keyboard_commands(env):
    st = getattr(env, "_keyboard_teleop_state", None)
    if st is None:
        return
    vx = st["vx"]
    h = st["heading"]
    ht = st["height"]
    env.commands[:, 0] = vx
    env.commands[:, 2] = ht
    env.commands[:, 3] = h


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    lqr_demo = getattr(args, "lqr_demo", False)
    if lqr_demo and hasattr(env_cfg.control, "control_path"):
        # make_env 内 update_cfg_from_args 会用 args.control_path 覆盖 cfg，此处同步 args
        args.control_path = "vmc_lqr"
    if USE_KEYBOARD_TELEOP:
        # Avoid LeggedRobot._post_physics_step_callback periodically overwriting commands.
        env_cfg.commands.resampling_time = 1e9
    # override some parameters for testing
    env_cfg.env.episode_length_s = 20
    env_cfg.env.fail_to_terminal_time_s = 3
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4)  # reduced for inference to save GPU memory
    if lqr_demo:
        env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.friction_range = [0.1, 0.2]
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    if lqr_demo:
        env_cfg.commands.resampling_time = 1e9

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if USE_KEYBOARD_TELEOP:
        if getattr(args, "headless", False) or env.viewer is None:
            print("USE_KEYBOARD_TELEOP 需要可视化窗口（不要使用 --headless）。")
        else:
            install_keyboard_teleop(env, env_cfg)
    obs, obs_history = env.get_observations()

    policy = None
    ppo_runner = None
    latent = None
    if lqr_demo:
        if getattr(args, "headless", False):
            print(
                "[play] --lqr_demo 建议去掉 --headless，否则无法打开窗口观察单机器人。"
            )
        if not hasattr(env_cfg.control, "control_path"):
            raise ValueError(
                "--lqr_demo 仅适用于带 vmc_lqr 控制路径的任务（如 wheel_legged_vmc / wheel_legged_vmc_flat）。"
            )
        print(
            "[play] LQR 演示：单环境、不加载 RL 策略；"
            "虚拟腿由 LQR+K 矩阵控制，动作为零（仅站姿/指令 hold）。"
        )
        print(f"[play] LQR 演示目标高度: {LQR_DEMO_HEIGHT_CMD} m（RL 可达区）")
        actions_zero = torch.zeros(
            env.num_envs, env.num_actions, device=env.device, dtype=torch.float
        )
        # Refresh virtual-leg kinematics without stepping simulation to avoid
        # introducing an uncontrolled fall before references are set.
        if hasattr(env, "leg_post_physics_step"):
            env.leg_post_physics_step()

        # Initialize references from runtime standing kinematics first; fallback to nominal cfg.
        theta_scale = float(env.cfg.control.action_scale_theta)
        l0_nominal, theta_nominal = nominal_virtual_leg_from_cfg(env_cfg)
        if hasattr(env, "theta0"):
            theta_runtime = env.theta0[0].detach().cpu().numpy()
            theta_nominal = float(theta_runtime.mean())
        actions_zero[:, 0] = float(theta_nominal) / max(theta_scale, 1e-6)
        actions_zero[:, 3] = float(theta_nominal) / max(theta_scale, 1e-6)

        # Map height command to L0 reference channels with a first-order compensation from
        # nominal base-height deviation (height->L0 gain ~0.6 around nominal stand).
        l0_offset = float(env.cfg.control.l0_offset)
        l0_scale = float(env.cfg.control.action_scale_l0)
        if hasattr(env, "L0"):
            l0_nominal = float(env.L0[0].mean().item())
        nominal_base_h = float(env_cfg.init_state.pos[2])
        l0_height_gain = 0.6
        l0_target = l0_nominal + l0_height_gain * (LQR_DEMO_HEIGHT_CMD - nominal_base_h)
        l0_action_from_height = (l0_target - l0_offset) / max(l0_scale, 1e-6)
        l0_action_from_height = float(l0_action_from_height)
        actions_zero[:, 1] = l0_action_from_height
        actions_zero[:, 4] = l0_action_from_height
        print(
            "[play] LQR 演示将高度命令映射到动作 L0 参考: "
            f"nominal_base_h={nominal_base_h:.3f}, L0_nom={l0_nominal:.3f}, "
            f"theta_nom={theta_nominal:.3f}, L0*={l0_target:.3f}, "
            f"action_l0={l0_action_from_height:.3f} (l0_offset={l0_offset:.3f}, scale={l0_scale:.3f})"
        )
        # Outer-loop height servo for lqr_demo only: slowly trim L0 references so that
        # base height tracks LQR_DEMO_HEIGHT_CMD despite model mismatch.
        h_trim_i = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
        h_kp = 2.8
        h_ki = 0.9
        h_trim_limit = 1.2
        l0_action_base = torch.full(
            (env.num_envs,), l0_action_from_height, device=env.device, dtype=torch.float
        )
        ff_base = float(env.cfg.control.feedforward_force)
        ff_trim_i = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
        ff_kp = 320.0
        ff_ki = 120.0
        ff_trim_limit = 0.35
        ff_min, ff_max = 20.0, 260.0
        # Height-priority direct force loop (demo only): inject additive radial force.
        env._lqr_height_force_enabled = True
        env._lqr_height_force_add = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float
        )
        h_force_kp = 900.0
        h_force_ki = 240.0
        h_force_kd = 120.0
        h_force_max = 220.0
        enable_theta_auto_trim = (
            getattr(env.cfg.control.lqr, "state_model", "paper6") != "legacy4"
        )
        theta_trim_i = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
        theta_trim_ki = 0.015
        theta_trim_limit = 0.35
    else:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        if EXPORT_POLICY:
            path = os.path.join(
                WHEEL_LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "policies",
            )
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print("Exported policy as jit script to: ", path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging (reduced from 21 due to smaller num_envs for inference)
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    CoM_offset_compensate = True and not USE_KEYBOARD_TELEOP and not lqr_demo
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)

    for i in range(1000 * int(env.max_episode_length)):
        if lqr_demo:
            actions = actions_zero
        elif ppo_runner.alg.actor_critic.is_sequence:
            actions, latent = policy(obs, obs_history)
        else:
            actions = policy(obs.detach())

        if USE_KEYBOARD_TELEOP and getattr(env, "_keyboard_teleop_state", None) is not None:
            apply_keyboard_commands(env)
        elif lqr_demo:
            env.commands[:, 0] = 0.0
            env.commands[:, 2] = LQR_DEMO_HEIGHT_CMD
            env.commands[:, 3] = 0.0
        else:
            env.commands[:, 0] = 2.5
            env.commands[:, 2] = PLAY_DEFAULT_HEIGHT_CMD  # + 0.07 * np.sin(i * 0.01)
            env.commands[:, 3] = 0

        if CoM_offset_compensate:
            if i > 200 and i < 600:
                vel_cmd[:] = 2.5 * np.clip((i - 200) * 0.05, 0, 1)
            else:
                vel_cmd[:] = 0
            vel_err_intergral += (
                (vel_cmd - env.base_lin_vel[:, 0])
                * env.dt
                * ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5)
            )
            vel_err_intergral = torch.clip(vel_err_intergral, -0.5, 0.5)
            env.commands[:, 0] = vel_cmd + vel_err_intergral

        obs, _, rews, dones, infos, obs_history = env.step(actions)
        if lqr_demo and enable_theta_auto_trim and hasattr(env, "theta0"):
            # Auto-trim theta reference to reduce static bias between linear model
            # equilibrium and the actual simulator equilibrium.
            theta_ref = actions_zero[:, [0, 3]] * env.cfg.control.action_scale_theta
            theta_err_lr = env.theta0 - theta_ref
            theta_err_mean = theta_err_lr.mean(dim=1)
            theta_trim_i += theta_err_mean * env.dt
            theta_trim_i = torch.clip(theta_trim_i, -theta_trim_limit, theta_trim_limit)
            theta_ref_new = torch.clip(
                theta_ref + theta_trim_ki * theta_trim_i.unsqueeze(1),
                -theta_trim_limit,
                theta_trim_limit,
            )
            actions_zero[:, 0] = theta_ref_new[:, 0] / env.cfg.control.action_scale_theta
            actions_zero[:, 3] = theta_ref_new[:, 1] / env.cfg.control.action_scale_theta
        if lqr_demo:
            h_err = LQR_DEMO_HEIGHT_CMD - env.base_height
            h_trim_i += h_err * env.dt
            h_trim_i = torch.clip(h_trim_i, -h_trim_limit, h_trim_limit)
            # Height-priority outer loop: directly servo L0 reference from measured base height error.
            l0_action_cmd = l0_action_base + h_kp * h_err + h_ki * h_trim_i
            l0_action_cmd = torch.clip(l0_action_cmd, -6.0, 6.0)
            actions_zero[:, 1] = l0_action_cmd
            actions_zero[:, 4] = l0_action_cmd
            ff_trim_i += h_err * env.dt
            ff_trim_i = torch.clip(ff_trim_i, -ff_trim_limit, ff_trim_limit)
            ff_cmd = ff_base + ff_kp * h_err + ff_ki * ff_trim_i
            ff_cmd = torch.clip(ff_cmd, ff_min, ff_max)
            env.cfg.control.feedforward_force = float(ff_cmd.mean().item())
            # Direct radial-force compensation from height error/vertical speed.
            h_force_add = h_force_kp * h_err + h_force_ki * h_trim_i - h_force_kd * env.base_lin_vel[:, 2]
            h_force_add = torch.clip(h_force_add, -h_force_max, h_force_max)
            env._lqr_height_force_add = h_force_add
        if lqr_demo and i % 200 == 0:
            base_h = env.base_height[robot_index].item()
            cmd_h = env.commands[robot_index, 2].item()
            l0_l = env.L0[robot_index, 0].item() if hasattr(env, "L0") else float("nan")
            theta_ref = actions_zero[robot_index, 0].item() * env.cfg.control.action_scale_theta
            l0_ref = actions_zero[robot_index, 1].item() * env.cfg.control.action_scale_l0 + env.cfg.control.l0_offset
            theta_err = (
                (env.theta0[robot_index] - theta_ref).abs().mean().item()
                if hasattr(env, "theta0")
                else float("nan")
            )
            l0_err = (
                (env.L0[robot_index] - l0_ref).abs().mean().item()
                if hasattr(env, "L0")
                else float("nan")
            )
            print(
                f"[lqr_demo] step={i:05d} base_h={base_h:.3f} cmd_h={cmd_h:.3f} "
                f"L0={l0_l:.3f} l0_act={actions_zero[robot_index, 1].item():.3f} "
                f"ff={env.cfg.control.feedforward_force:.1f} "
                f"hF={env._lqr_height_force_add[robot_index].item():.1f} "
                f"theta_ref={theta_ref:.3f} "
                f"|theta_err|={theta_err:.3f} |L0_err|={l0_err:.3f}"
            )
        elif (not lqr_demo) and i % 200 == 0:
            print(
                f"[rl_play] step={i:05d} base_h={env.base_height[robot_index].item():.3f} "
                f"cmd_h={env.commands[robot_index, 2].item():.3f} "
                f"vx={env.base_lin_vel[robot_index, 0].item():.3f}"
            )
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    WHEEL_LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale
                    + env.default_dof_pos[robot_index, joint_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_yaw": env.commands[robot_index, 1].item(),
                    "command_height": env.commands[robot_index, 2].item(),
                    "base_height": env.base_height[robot_index].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            if CoM_offset_compensate:
                logger.log_states({"command_x": vel_cmd[robot_index].item()})
            else:
                logger.log_states({"command_x": env.commands[robot_index, 0].item()})
            if latent is not None:
                logger.log_states(
                    {
                        "est_lin_vel_x": latent[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": latent[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": latent[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
                if latent.shape[1] > 3 and env_cfg.noise.add_noise:
                    logger.log_states(
                        {
                            "base_vel_yaw_obs": obs[robot_index, 2].item()
                            / env.cfg.normalization.obs_scales.ang_vel,
                            "dof_pos_obs": obs[robot_index, 9 + joint_index].item()
                            / env.cfg.normalization.obs_scales.dof_pos
                            + env.default_dof_pos[robot_index, joint_index].item(),
                            "dof_vel_obs": obs[robot_index, 15 + joint_index].item()
                            / env.cfg.normalization.obs_scales.dof_vel,
                        }
                    )
                    logger.log_states(
                        {
                            "base_vel_yaw_est": latent[robot_index, 3 + 2].item()
                            / env.cfg.normalization.obs_scales.ang_vel,
                            "dof_pos_est": latent[
                                robot_index, 3 + 9 + joint_index
                            ].item()
                            / env.cfg.normalization.obs_scales.dof_pos
                            + env.default_dof_pos[robot_index, joint_index].item(),
                            "dof_vel_est": latent[
                                robot_index, 3 + 15 + joint_index
                            ].item()
                            / env.cfg.normalization.obs_scales.dof_vel,
                        }
                    )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            episode_info = infos.get("episode")
            if episode_info:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(episode_info, num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    # True：用键盘改 env.commands（需在仿真窗口聚焦）；转向需 cfg.commands.heading_command=True
    USE_KEYBOARD_TELEOP = True
    args = get_args()
    play(args)
