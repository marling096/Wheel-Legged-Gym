#!/usr/bin/env python3
"""Visualize the terrain heightfields used by the SWPU2026 VMC tasks."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import types

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np


TASK_CONFIGS = {
    "swpu2026_vmc_gravel": (
        "wheel_legged_gym.envs.swpu2026_vmc_gravel.swpu2026_vmc_gravel_config",
        "Swpu2026VMCGravelCfg",
    ),
    "swpu2026_vmc_undulating": (
        "wheel_legged_gym.envs.swpu2026_vmc_undulating.swpu2026_vmc_undulating_config",
        "Swpu2026VMCUndulatingCfg",
    ),
}

ROOT_DIR = Path(__file__).resolve().parents[2]
ENVS_DIR = ROOT_DIR / "wheel_legged_gym" / "envs"
TERRAIN_PATH = ROOT_DIR / "wheel_legged_gym" / "utils" / "terrain.py"


def _load_module(module_name, path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_package(package_name, path):
    if package_name in sys.modules:
        return sys.modules[package_name]
    package = types.ModuleType(package_name)
    package.__path__ = [str(path)]
    sys.modules[package_name] = package
    parent_name, _, child_name = package_name.rpartition(".")
    if parent_name in sys.modules:
        setattr(sys.modules[parent_name], child_name, package)
    return package


def _load_task_config(task_name):
    import wheel_legged_gym

    _ensure_package("wheel_legged_gym.envs", ENVS_DIR)
    _ensure_package("wheel_legged_gym.envs.base", ENVS_DIR / "base")
    _ensure_package("wheel_legged_gym.envs.wheel_legged", ENVS_DIR / "wheel_legged")
    _ensure_package(
        "wheel_legged_gym.envs.wheel_legged_vmc",
        ENVS_DIR / "wheel_legged_vmc",
    )
    _ensure_package(
        "wheel_legged_gym.envs.wheel_legged_vmc_flat",
        ENVS_DIR / "wheel_legged_vmc_flat",
    )
    _ensure_package(
        "wheel_legged_gym.envs.swpu2026_vmc_flat",
        ENVS_DIR / "swpu2026_vmc_flat",
    )
    _ensure_package(
        "wheel_legged_gym.envs.swpu2026_vmc_undulating",
        ENVS_DIR / "swpu2026_vmc_undulating",
    )
    _ensure_package(
        "wheel_legged_gym.envs.swpu2026_vmc_gravel",
        ENVS_DIR / "swpu2026_vmc_gravel",
    )

    setattr(wheel_legged_gym, "envs", sys.modules["wheel_legged_gym.envs"])

    _load_module(
        "wheel_legged_gym.envs.base.base_config",
        ENVS_DIR / "base" / "base_config.py",
    )
    _load_module(
        "wheel_legged_gym.envs.base.legged_robot_config",
        ENVS_DIR / "base" / "legged_robot_config.py",
    )
    _load_module(
        "wheel_legged_gym.envs.wheel_legged.wheel_legged_config",
        ENVS_DIR / "wheel_legged" / "wheel_legged_config.py",
    )
    _load_module(
        "wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc_config",
        ENVS_DIR / "wheel_legged_vmc" / "wheel_legged_vmc_config.py",
    )
    _load_module(
        "wheel_legged_gym.envs.wheel_legged_vmc_flat.wheel_legged_vmc_flat_config",
        ENVS_DIR / "wheel_legged_vmc_flat" / "wheel_legged_vmc_flat_config.py",
    )
    _load_module(
        "wheel_legged_gym.envs.swpu2026_vmc_flat.swpu2026_vmc_flat_config",
        ENVS_DIR / "swpu2026_vmc_flat" / "swpu2026_vmc_flat_config.py",
    )
    _load_module(
        "wheel_legged_gym.envs.swpu2026_vmc_undulating.swpu2026_vmc_undulating_config",
        ENVS_DIR / "swpu2026_vmc_undulating" / "swpu2026_vmc_undulating_config.py",
    )
    _load_module(
        "wheel_legged_gym.envs.swpu2026_vmc_gravel.swpu2026_vmc_gravel_config",
        ENVS_DIR / "swpu2026_vmc_gravel" / "swpu2026_vmc_gravel_config.py",
    )

    module_name, class_name = TASK_CONFIGS[task_name]
    return getattr(sys.modules[module_name], class_name)()


Terrain = _load_module("_swpu2026_terrain", TERRAIN_PATH).Terrain


def _terrain_height_map(cfg, include_border):
    terrain = Terrain(cfg.terrain, cfg.env.num_envs)
    heights = terrain.height_field_raw.astype(np.float64) * cfg.terrain.vertical_scale

    if include_border or terrain.border == 0:
        return terrain, heights

    border = terrain.border
    return terrain, heights[border:-border, border:-border]


def _plot_task(task_name, cfg, include_border, surface_stride, output_dir):
    terrain, heights = _terrain_height_map(cfg, include_border)
    stride = max(1, int(surface_stride))

    x_size = heights.shape[0] * cfg.terrain.horizontal_scale
    y_size = heights.shape[1] * cfg.terrain.horizontal_scale
    x = np.arange(heights.shape[0]) * cfg.terrain.horizontal_scale
    y = np.arange(heights.shape[1]) * cfg.terrain.horizontal_scale
    x_surface, y_surface = np.meshgrid(x[::stride], y[::stride], indexing="ij")
    z_surface = heights[::stride, ::stride]

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    fig.suptitle(task_name, fontsize=13)

    ax_map = fig.add_subplot(1, 2, 1)
    image = ax_map.imshow(
        heights.T,
        origin="lower",
        extent=[0.0, x_size, 0.0, y_size],
        aspect="equal",
        cmap="terrain",
    )
    ax_map.set_title("Height map")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.grid(color="white", alpha=0.25, linewidth=0.5)
    colorbar = fig.colorbar(image, ax=ax_map, fraction=0.046, pad=0.04)
    colorbar.set_label("height [m]")

    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_3d.plot_surface(
        x_surface,
        y_surface,
        z_surface,
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax_3d.set_title("Surface")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("height [m]")
    ax_3d.view_init(elev=32, azim=-125)

    print(
        f"{task_name}: profile={getattr(cfg.terrain, 'undulating_profile', 'unknown')}, "
        f"rows={cfg.terrain.num_rows}, cols={cfg.terrain.num_cols}, "
        f"cell={terrain.env_length:.1f}x{terrain.env_width:.1f} m, "
        f"height=[{heights.min():.3f}, {heights.max():.3f}] m"
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{task_name}_terrain.png")
        fig.savefig(out_path, dpi=160)
        print(f"saved: {out_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Show the terrain heightfields used by SWPU2026 VMC tasks."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_CONFIGS.keys()),
        action="append",
        help="Task to visualize. Repeat the option for multiple tasks. Default: both.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="NumPy random seed used for randomized terrain cells.",
    )
    parser.add_argument(
        "--include-border",
        action="store_true",
        help="Include the flat simulator border around the playable terrain.",
    )
    parser.add_argument(
        "--surface-stride",
        type=int,
        default=2,
        help="Downsample stride for the 3D surface plot.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for saving PNG previews.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening the interactive Matplotlib window.",
    )
    args = parser.parse_args()

    tasks = args.task or sorted(TASK_CONFIGS.keys())
    np.random.seed(args.seed)

    for task_name in tasks:
        cfg = _load_task_config(task_name)
        _plot_task(
            task_name=task_name,
            cfg=cfg,
            include_border=args.include_border,
            surface_stride=args.surface_stride,
            output_dir=args.output_dir,
        )

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
