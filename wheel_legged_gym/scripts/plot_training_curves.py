#!/usr/bin/env python3
"""从单次训练的 TensorBoard 事件文件导出奖励 / 损失 / 控制误差曲线图."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load_scalar(ea: EventAccumulator, tag: str):
    try:
        events = ea.Scalars(tag)
    except KeyError:
        return None, None
    steps = np.array([e.step for e in events], dtype=np.float64)
    vals = np.array([e.value for e in events], dtype=np.float64)
    return steps, vals


def main():
    parser = argparse.ArgumentParser(
        description="读取 TensorBoard 日志目录并生成 training_curves.png"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="单次实验目录（内含 events.out.tfevents.*），例如 logs/<experiment>/<日期>_run/",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 PNG 路径，默认写入 log_dir/training_curves.png",
    )
    args = parser.parse_args()

    log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    if not os.path.isdir(log_dir):
        print(f"目录不存在: {log_dir}", file=sys.stderr)
        sys.exit(1)

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])
    if not scalar_tags:
        print(f"未在 {log_dir} 中找到任何标量事件。", file=sys.stderr)
        sys.exit(1)

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "DejaVu Sans",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)

    # —— 奖励 ——
    ax = axes[0]
    tags_reward = sorted(t for t in scalar_tags if t.startswith("Episode/rew_"))
    plotted = False
    steps, vals = _load_scalar(ea, "Train/mean_reward")
    if steps is not None:
        ax.plot(steps, vals, label="Train/mean_reward（回合均值近似）", linewidth=2.0)
        plotted = True
    for tag in tags_reward[:14]:
        steps, vals = _load_scalar(ea, tag)
        if steps is None:
            continue
        ax.plot(steps, vals, label=tag.replace("Episode/", ""), alpha=0.75, linewidth=1.0)
        plotted = True
    ax.set_ylabel("数值")
    ax.set_title("奖励：回合奖励分项（Episode/*）与 Train/mean_reward")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="upper left", fontsize=8, ncol=2)

    # —— 损失与策略指标 ——
    ax = axes[1]
    loss_tags = [
        "Loss/value_function",
        "Loss/surrogate",
        "Loss/encoder",
        "Policy/mean_kl",
        "Policy/mean_noise_std",
    ]
    for tag in loss_tags:
        steps, vals = _load_scalar(ea, tag)
        if steps is None:
            continue
        ax.plot(steps, vals, label=tag.replace("Loss/", "").replace("Policy/", ""))
    ax.set_ylabel("损失 / 指标")
    ax.set_title("损失（价值 / 替代 / 编码器）与 KL、探索噪声标准差")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # —— 控制跟踪误差（训练中 rollout 即时误差均值）——
    ax = axes[2]
    ctrl_tags = [
        "Control/mean_abs_lin_vel_x_error",
        "Control/mean_abs_yaw_vel_error",
        "Control/mean_abs_height_error",
    ]
    ctrl_plotted = False
    for tag in ctrl_tags:
        steps, vals = _load_scalar(ea, tag)
        if steps is None:
            continue
        ax.plot(steps, vals, label=tag.replace("Control/", ""))
        ctrl_plotted = True
    if not ctrl_plotted:
        ax.text(
            0.5,
            0.5,
            "无 Control/* 标签（请使用更新后的代码重新训练）",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("迭代次数（与学习迭代一致）")
    ax.set_ylabel("误差（物理单位：m/s、rad/s、m）")
    ax.set_title("控制效果：指令与状态的平均绝对误差（越小越好）")
    ax.grid(True, alpha=0.3)
    if ctrl_plotted:
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"训练曲线\n{log_dir}", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_path = args.output or os.path.join(log_dir, "training_curves.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
