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

import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import os
import csv


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self._rt_enabled = False
        self._rt_update_interval = 1
        self._rt_window_steps = 1
        self._rt_plt = None
        self._rt_fig = None
        self._rt_axes = None
        self._rt_lines = {}

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

    def plot_states(self):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ModuleNotFoundError:
            print("matplotlib is not installed; skipping state plots.")
            return
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def init_realtime_plot(self, update_interval_steps=10, window_seconds=20.0):
        try:
            import matplotlib.pyplot as plt
            import matplotlib
        except ModuleNotFoundError:
            print("matplotlib is not installed; skipping realtime plots.")
            return
        self._rt_plt = plt
        backend = str(matplotlib.get_backend()).lower()
        if "agg" in backend:
            print(
                f"matplotlib backend is '{matplotlib.get_backend()}', realtime plot window may not appear. "
                "Try setting MPLBACKEND=TkAgg and ensure tkinter is installed."
            )
        self._rt_plt.ion()
        self._rt_fig, self._rt_axes = self._rt_plt.subplots(2, 2, figsize=(12, 8))
        self._rt_update_interval = max(1, int(update_interval_steps))
        self._rt_window_steps = max(2, int(window_seconds / max(self.dt, 1e-6)))
        self._rt_enabled = True

        ax = self._rt_axes
        self._rt_lines = {
            "base_vel_x": ax[0, 0].plot([], [], label="real vx")[0],
            "command_x": ax[0, 0].plot([], [], label="cmd vx")[0],
            "base_height": ax[0, 1].plot([], [], label="real h")[0],
            "command_height": ax[0, 1].plot([], [], label="cmd h")[0],
            "base_pitch": ax[1, 0].plot([], [], label="pitch")[0],
            "base_roll": ax[1, 0].plot([], [], label="roll")[0],
            "track_error_vx": ax[1, 1].plot([], [], label="vx err")[0],
            "track_error_height": ax[1, 1].plot([], [], label="h err")[0],
            "overall_score": ax[1, 1].plot([], [], label="overall score")[0],
        }

        ax[0, 0].set_title("Velocity Tracking")
        ax[0, 0].set_xlabel("time [s]")
        ax[0, 0].set_ylabel("m/s")
        ax[0, 0].legend()
        ax[0, 1].set_title("Height Tracking")
        ax[0, 1].set_xlabel("time [s]")
        ax[0, 1].set_ylabel("m")
        ax[0, 1].legend()
        ax[1, 0].set_title("Body Posture")
        ax[1, 0].set_xlabel("time [s]")
        ax[1, 0].set_ylabel("rad")
        ax[1, 0].legend()
        ax[1, 1].set_title("Errors & Score")
        ax[1, 1].set_xlabel("time [s]")
        ax[1, 1].set_ylabel("value")
        ax[1, 1].legend()
        self._rt_fig.tight_layout()
        self._rt_fig.show()
        self._rt_plt.show(block=False)
        self._rt_fig.canvas.draw_idle()
        self._rt_plt.pause(0.001)

    def update_realtime_plot(self, step):
        if (not self._rt_enabled) or (self._rt_fig is None):
            return
        if step % self._rt_update_interval != 0:
            return

        log = self.state_log
        n = len(log["base_vel_x"])
        if n < 2:
            return
        start = max(0, n - self._rt_window_steps)
        t = np.arange(start, n) * self.dt

        def _tail(key):
            v = log[key]
            if not v:
                return None
            return np.asarray(v[start:n], dtype=float)

        vx = _tail("base_vel_x")
        cmd_vx = _tail("command_x")
        bh = _tail("base_height")
        cmd_h = _tail("command_height")
        pitch = _tail("base_pitch")
        roll = _tail("base_roll")
        e_vx = _tail("track_error_vx")
        e_h = _tail("track_error_height")

        if vx is not None:
            self._rt_lines["base_vel_x"].set_data(t, vx)
        if cmd_vx is not None:
            self._rt_lines["command_x"].set_data(t, cmd_vx)
        if bh is not None:
            self._rt_lines["base_height"].set_data(t, bh)
        if cmd_h is not None:
            self._rt_lines["command_height"].set_data(t, cmd_h)
        if pitch is not None:
            self._rt_lines["base_pitch"].set_data(t, pitch)
        if roll is not None:
            self._rt_lines["base_roll"].set_data(t, roll)
        if e_vx is not None:
            self._rt_lines["track_error_vx"].set_data(t, e_vx)
        if e_h is not None:
            self._rt_lines["track_error_height"].set_data(t, e_h)

        if (e_vx is not None) and (e_h is not None):
            score = 0.5 * (1.0 / (1.0 + np.abs(e_vx)) + 1.0 / (1.0 + 20.0 * np.abs(e_h)))
            self._rt_lines["overall_score"].set_data(t, score)

        for ax in self._rt_axes.reshape(-1):
            ax.relim()
            ax.autoscale_view()

        self._rt_fig.canvas.draw_idle()
        self._rt_plt.pause(0.001)

    def close_realtime_plot(self):
        if self._rt_plt is None:
            return
        self._rt_plt.ioff()

    def save_performance_curves(self, output_dir, prefix="play_performance"):
        os.makedirs(output_dir, exist_ok=True)
        log = self.state_log
        n = len(log["base_vel_x"])
        if n == 0:
            print("No state logs to save.")
            return None, None
        time = np.arange(n) * self.dt

        def _arr(key):
            if key not in log or len(log[key]) == 0:
                return np.full(n, np.nan, dtype=float)
            arr = np.asarray(log[key], dtype=float)
            if arr.shape[0] < n:
                pad = np.full(n - arr.shape[0], np.nan, dtype=float)
                arr = np.concatenate([arr, pad], axis=0)
            return arr[:n]

        cmd_vx = _arr("command_x")
        real_vx = _arr("base_vel_x")
        cmd_h = _arr("command_height")
        real_h = _arr("base_height")
        pitch = _arr("base_pitch")
        roll = _arr("base_roll")
        evx = _arr("track_error_vx")
        eh = _arr("track_error_height")
        score_vx = 1.0 / (1.0 + np.abs(evx))
        score_h = 1.0 / (1.0 + 20.0 * np.abs(eh))
        score = 0.5 * (score_vx + score_h)

        csv_path = os.path.join(output_dir, f"{prefix}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time_s",
                    "command_vx",
                    "base_vel_x",
                    "command_height",
                    "base_height",
                    "base_pitch",
                    "base_roll",
                    "track_error_vx",
                    "track_error_height",
                    "score_vx",
                    "score_height",
                    "score_overall",
                ]
            )
            for i in range(n):
                writer.writerow(
                    [
                        float(time[i]),
                        float(cmd_vx[i]),
                        float(real_vx[i]),
                        float(cmd_h[i]),
                        float(real_h[i]),
                        float(pitch[i]),
                        float(roll[i]),
                        float(evx[i]),
                        float(eh[i]),
                        float(score_vx[i]),
                        float(score_h[i]),
                        float(score[i]),
                    ]
                )

        png_path = None
        try:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs[0, 0].plot(time, real_vx, label="real vx")
            axs[0, 0].plot(time, cmd_vx, label="cmd vx")
            axs[0, 0].set_title("Velocity Tracking")
            axs[0, 0].set_xlabel("time [s]")
            axs[0, 0].set_ylabel("m/s")
            axs[0, 0].legend()

            axs[0, 1].plot(time, real_h, label="real h")
            axs[0, 1].plot(time, cmd_h, label="cmd h")
            axs[0, 1].set_title("Height Tracking")
            axs[0, 1].set_xlabel("time [s]")
            axs[0, 1].set_ylabel("m")
            axs[0, 1].legend()

            axs[1, 0].plot(time, pitch, label="pitch")
            axs[1, 0].plot(time, roll, label="roll")
            axs[1, 0].set_title("Body Posture")
            axs[1, 0].set_xlabel("time [s]")
            axs[1, 0].set_ylabel("rad")
            axs[1, 0].legend()

            axs[1, 1].plot(time, evx, label="vx err")
            axs[1, 1].plot(time, eh, label="h err")
            axs[1, 1].plot(time, score, label="overall score")
            axs[1, 1].set_title("Errors & Score")
            axs[1, 1].set_xlabel("time [s]")
            axs[1, 1].set_ylabel("value")
            axs[1, 1].legend()

            fig.tight_layout()
            png_path = os.path.join(output_dir, f"{prefix}.png")
            fig.savefig(png_path, dpi=160)
            plt.close(fig)
        except ModuleNotFoundError:
            print("matplotlib is not installed; saved CSV only.")

        return csv_path, png_path

    def _plot(self):
        import matplotlib.pyplot as plt

        nb_rows = 4
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(15, 12))
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot joint targets and real positions
        a = axs[1, 0]
        if log["dof_pos_obs"]:
            a.plot(time, log["dof_pos_obs"], label="obs")
        if log["dof_pos_est"]:
            a.plot(time, log["dof_pos_est"], label="est")
        if log["dof_pos"]:
            a.plot(time, log["dof_pos"], label="real")
        if log["dof_pos_target"]:
            a.plot(time, log["dof_pos_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Position [rad]", title="DOF Position")
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel_obs"]:
            a.plot(time, log["dof_vel_obs"], label="obs")
        if log["dof_vel_est"]:
            a.plot(time, log["dof_vel_est"], label="est")
        if log["dof_vel"]:
            a.plot(time, log["dof_vel"], label="real")
        if log["dof_vel_target"]:
            a.plot(time, log["dof_vel_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity")
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label="real")
        if log["est_lin_vel_x"]:
            a.plot(time, log["est_lin_vel_x"], label="est")
        if log["command_x"]:
            a.plot(time, log["command_x"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label="real")
        if log["est_lin_vel_y"]:
            a.plot(time, log["est_lin_vel_y"], label="est")
        if log["command_y"]:
            a.plot(time, log["command_y"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity y")
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw_obs"]:
            a.plot(time, log["base_vel_yaw_obs"], label="obs")
        if log["base_vel_yaw_est"]:
            a.plot(time, log["base_vel_yaw_est"], label="est")
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label="real")
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label="commanded")
        a.set(
            xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base velocity yaw"
        )
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]:
            a.plot(time, log["base_vel_z"], label="real")
        if log["est_lin_vel_z"]:
            a.plot(time, log["est_lin_vel_z"], label="est")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity z")
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        # if log["contact_forces_z"]:
        #     forces = np.array(log["contact_forces_z"])
        #     for i in range(forces.shape[1]):
        #         a.plot(time, forces[:, i], label=f"force {i}")
        # a.set(xlabel="time [s]", ylabel="Forces z [N]", title="Vertical Contact forces")
        if log["base_height"]:
            a.plot(time, log["base_height"], label="real")
        if log["command_height"]:
            a.plot(time, log["command_height"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base height [m]", title="Base Height")
        a.legend()
        # plot torque/vel curves
        a = axs[2, 1]
        if log["dof_vel"] != [] and log["dof_torque"] != []:
            a.plot(log["dof_vel"], log["dof_torque"], "x", label="real")
        a.set(
            xlabel="Joint vel [rad/s]",
            ylabel="Joint Torque [Nm]",
            title="Torque/velocity curves",
        )
        a.legend()
        # plot torques
        a = axs[2, 2]
        if log["dof_torque"] != []:
            a.plot(time, log["dof_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="Joint Torque [Nm]", title="Torque")
        a.legend()

        # posture (balance)
        a = axs[3, 0]
        if log["base_pitch"]:
            a.plot(time, log["base_pitch"], label="pitch")
        if log["base_roll"]:
            a.plot(time, log["base_roll"], label="roll")
        a.set(xlabel="time [s]", ylabel="angle [rad]", title="Body Posture")
        a.legend()

        # tracking errors
        a = axs[3, 1]
        if log["track_error_vx"]:
            a.plot(time, log["track_error_vx"], label="vx error")
        if log["track_error_height"]:
            a.plot(time, log["track_error_height"], label="height error")
        a.set(xlabel="time [s]", ylabel="error", title="Tracking Errors")
        a.legend()

        # performance tracking (normalized score, higher is better)
        a = axs[3, 2]
        if log["track_error_vx"] or log["track_error_height"]:
            vx_err = np.abs(np.array(log["track_error_vx"])) if log["track_error_vx"] else np.array([])
            h_err = (
                np.abs(np.array(log["track_error_height"]))
                if log["track_error_height"]
                else np.array([])
            )
            # Simple bounded score in [0, 1], for visual controllability comparison.
            vx_score = 1.0 / (1.0 + vx_err) if vx_err.size > 0 else None
            h_score = 1.0 / (1.0 + 20.0 * h_err) if h_err.size > 0 else None
            if vx_score is not None:
                a.plot(time[: len(vx_score)], vx_score, label="vx score")
            if h_score is not None:
                a.plot(time[: len(h_score)], h_score, label="height score")
            if vx_score is not None and h_score is not None:
                total = 0.5 * (vx_score + h_score)
                a.plot(time[: len(total)], total, label="overall score")
        a.set(xlabel="time [s]", ylabel="score", title="Controllability Score")
        a.set_ylim(0.0, 1.05)
        a.legend()
        fig.tight_layout()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
