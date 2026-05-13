from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, task_registry
import torch


def show_terrain_preview(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device, dtype=torch.float
    )
    preview_names = getattr(env_cfg.terrain, "preview_task_names", [])
    if preview_names:
        print("[terrain_preview] showing terrains from:", ", ".join(preview_names))
    print("[terrain_preview] press Esc to exit, V to toggle viewer sync.")
    try:
        while True:
            env.step(actions)
    except KeyboardInterrupt:
        print("\n[terrain_preview] interrupted from terminal.")


if __name__ == "__main__":
    args = get_args()
    if args.task == "anymal_c_flat":
        args.task = "swpu2026_terrain_preview"
    if args.headless:
        raise ValueError("terrain preview requires a viewer; do not use --headless")
    show_terrain_preview(args)
