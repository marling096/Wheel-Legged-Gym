from isaacgym import gymtorch
import torch

from wheel_legged_gym.envs import *  # noqa: F401,F403
from wheel_legged_gym.utils import get_args, task_registry


def _set_root_height(env, heights):
    env.root_states[:, :3] = env.base_init_state[:3]
    env.root_states[:, :3] += env.env_origins
    env.root_states[:, 2] = torch.tensor(heights, device=env.device)
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )


def _set_default_dofs(env):
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def main():
    args = get_args()
    args.task = "swpu2026_vmc_flat"
    args.headless = True
    args.num_envs = 2

    env, _ = task_registry.make_env(name=args.task, args=args)
    env.reset()

    dof_props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    print("loaded DOF properties:")
    for i, name in enumerate(env.dof_names):
        print(
            f"  {i} {name}: "
            f"driveMode={int(dof_props['driveMode'][i])}, "
            f"stiffness={float(dof_props['stiffness'][i])}, "
            f"damping={float(dof_props['damping'][i])}, "
            f"effort={float(dof_props['effort'][i])}, "
            f"velocity={float(dof_props['velocity'][i])}"
        )

    _set_root_height(env, [0.31, 2.00])
    _set_default_dofs(env)

    wheel_ids = list(env.wheel_dof_indices)
    torques = torch.zeros_like(env.torques)
    torques[:, wheel_ids] = 20.0

    start_pos = env.dof_pos[:, wheel_ids].clone()
    start_root = env.root_states[:, :3].clone()
    steps = 60
    for _ in range(steps):
        env.gym.set_dof_actuation_force_tensor(
            env.sim, gymtorch.unwrap_tensor(torques)
        )
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)

    end_pos = env.dof_pos[:, wheel_ids].clone()
    end_vel = env.dof_vel[:, wheel_ids].clone()
    root_delta = env.root_states[:, :3] - start_root
    contact_norm = torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1)

    labels = ["ground", "air"]
    print("swpu2026 constant wheel torque check")
    print(f"wheel_dof_indices={wheel_ids}, applied_torque_nm=20.0, steps={steps}")
    for i, label in enumerate(labels):
        print(
            f"{label}: "
            f"wheel_pos_delta={end_pos[i].detach().cpu().tolist()}, "
            f"wheel_vel={end_vel[i].detach().cpu().tolist()}, "
            f"root_delta={root_delta[i].detach().cpu().tolist()}, "
            f"wheel_contact_norm={contact_norm[i].detach().cpu().tolist()}"
        )

    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
