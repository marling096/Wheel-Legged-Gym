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

import os
from collections import defaultdict
from multiprocessing import Process

import numpy as np


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self, save_path=None, title=None, show=True, blocking=False):
        if not self.state_log:
            return
        if blocking:
            self._plot(save_path=save_path, title=title, show=show)
            return
        self.plot_process = Process(
            target=self._plot,
            kwargs={"save_path": save_path, "title": title, "show": show},
        )
        self.plot_process.start()

    def start_live_plot(self, save_path=None, title=None):
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt

        plt.ion()
        self._live_fig, self._live_axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        self._live_fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.08, hspace=0.10)
        if title:
            self._live_fig.suptitle(title, fontsize=12)

        self._live_save_path = save_path
        self._live_lines = {}

        def on_close(event):
            if self._live_save_path:
                os.makedirs(os.path.dirname(os.path.abspath(self._live_save_path)), exist_ok=True)
                self._live_fig.savefig(self._live_save_path, dpi=200)
                print(f"[play_plot] saved performance plot: {self._live_save_path}")

        self._live_fig.canvas.mpl_connect("close_event", on_close)
        plt.show(block=False)
        plt.pause(0.01)

    def update_live_plot(self):
        import matplotlib.pyplot as plt

        if not hasattr(self, "_live_fig") or not plt.fignum_exists(self._live_fig.number):
            return

        log = self.state_log
        if log.get("time"):
            time = np.asarray(log["time"], dtype=np.float64)
        else:
            return

        axs = self._live_axs

        def _update_tracking(ax, actual_key, command_key, ylabel, plot_title):
            if actual_key not in self._live_lines:
                a_line, = ax.plot([], [], label="actual", linewidth=1.5)
                c_line, = ax.plot([], [], label="command", linewidth=1.2)
                ax.set(xlabel="time [s]", ylabel=ylabel, title=plot_title)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                self._live_lines[actual_key] = a_line
                self._live_lines[command_key] = c_line
            if log.get(actual_key):
                self._live_lines[actual_key].set_data(time, log[actual_key])
            if log.get(command_key):
                self._live_lines[command_key].set_data(time, log[command_key])
            ax.relim()
            ax.autoscale_view()

        def _update_attitude(ax, key, plot_title):
            if key not in self._live_lines:
                line, = ax.plot([], [], label=key, linewidth=1.5)
                ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
                ax.set(xlabel="time [s]", ylabel="angle [rad]", title=plot_title)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                self._live_lines[key] = line
            if log.get(key):
                self._live_lines[key].set_data(time, log[key])
            ax.relim()
            ax.autoscale_view()

        _update_tracking(axs[0, 0], "base_vel_x", "command_x", "forward velocity [m/s]", "Forward Velocity Tracking")
        _update_tracking(axs[0, 1], "base_height", "command_height", "base height [m]", "Base Height Tracking")
        _update_attitude(axs[1, 0], "pitch", "Pitch Angle")
        _update_attitude(axs[1, 1], "roll", "Roll Angle")

        self._live_fig.canvas.draw_idle()
        plt.pause(0.001)

    def close_live_plot(self, save_path=None):
        import matplotlib.pyplot as plt

        path = save_path or getattr(self, "_live_save_path", None)
        if hasattr(self, "_live_fig") and plt.fignum_exists(self._live_fig.number):
            if path:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                self._live_fig.savefig(path, dpi=200)
                print(f"[play_plot] saved performance plot: {path}")
            plt.close(self._live_fig)

    def _plot(self, save_path=None, title=None, show=True):
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt

        log = self.state_log
        if log.get("time"):
            time = np.asarray(log["time"], dtype=np.float64)
        else:
            time = None
            for value in log.values():
                if value:
                    time = np.linspace(0.0, (len(value) - 1) * self.dt, len(value))
                    break
            if time is None:
                return

        fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        fig.set_constrained_layout_pads(
            w_pad=0.04,
            h_pad=0.04,
            wspace=0.08,
            hspace=0.10,
        )
        if title:
            fig.suptitle(title, fontsize=12)

        def _plot_tracking(ax, actual_key, command_key, ylabel, plot_title):
            if log.get(actual_key):
                ax.plot(time, log[actual_key], label="actual", linewidth=1.5)
            if log.get(command_key):
                ax.plot(time, log[command_key], label="command", linewidth=1.2)
            ax.set(xlabel="time [s]", ylabel=ylabel, title=plot_title)
            ax.grid(True, alpha=0.3)
            if ax.lines:
                ax.legend(fontsize=8)

        def _plot_attitude(ax, key, plot_title):
            if log.get(key):
                ax.plot(time, log[key], label=key, linewidth=1.5)
            ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set(xlabel="time [s]", ylabel="angle [rad]", title=plot_title)
            ax.grid(True, alpha=0.3)
            if len(ax.lines) > 1:
                ax.legend(fontsize=8)

        _plot_tracking(
            axs[0, 0],
            "base_vel_x",
            "command_x",
            "forward velocity [m/s]",
            "Forward Velocity Tracking",
        )
        _plot_tracking(
            axs[0, 1],
            "base_height",
            "command_height",
            "base height [m]",
            "Base Height Tracking",
        )
        _plot_attitude(axs[1, 0], "pitch", "Pitch Angle")
        _plot_attitude(axs[1, 1], "roll", "Roll Angle")

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200)
            print(f"[play_plot] saved performance plot: {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None and self.plot_process.is_alive():
            self.plot_process.kill()
