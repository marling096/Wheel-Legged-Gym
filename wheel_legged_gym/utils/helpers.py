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
import copy
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    def _model_files(run_dir):
        files = [
            f
            for f in os.listdir(run_dir)
            if "model" in f and f.endswith(".pt") and os.path.isfile(os.path.join(run_dir, f))
        ]
        files.sort(key=lambda m: "{0:0>15}".format(m))
        return files

    try:
        runs = [
            r
            for r in os.listdir(root)
            if os.path.isdir(os.path.join(root, r)) and r != "exported"
        ]
        runs.sort()
    except Exception as e:
        raise ValueError("Cannot list experiment directory: " + root) from e

    if not runs:
        raise ValueError(
            "No run folders under "
            + root
            + ". Train first or fix --experiment_name / logs path."
        )

    if load_run == -1 or load_run == "-1":
        load_run_dir = None
        for r in reversed(runs):
            cand = os.path.join(root, r)
            if _model_files(cand):
                load_run_dir = cand
                break
        if load_run_dir is None:
            raise ValueError(
                "No model_*.pt checkpoints found in any run under "
                + root
                + ". Either wait until training saves (see save_interval), "
                "or use --load_run <subdir_name> pointing to a folder that contains checkpoints."
            )
        load_run = load_run_dir
    else:
        raw = str(load_run).strip()
        expanded = os.path.expanduser(raw)
        candidates = []
        if os.path.isabs(expanded):
            candidates.append(os.path.normpath(expanded))
        candidates.append(os.path.normpath(os.path.join(root, expanded)))
        if not os.path.isabs(expanded):
            candidates.append(os.path.normpath(os.path.abspath(expanded)))
        tail = os.path.basename(os.path.normpath(expanded))
        if tail:
            candidates.append(os.path.normpath(os.path.join(root, tail)))

        load_run_resolved = None
        for c in candidates:
            if c and os.path.isdir(c):
                load_run_resolved = c
                break
        if load_run_resolved is None:
            raise ValueError(
                "Could not resolve load_run={0!r} under experiment root={1!r}. "
                "Use the run folder name only (e.g. May04_18-06-29_), "
                "or a path relative to the current working directory "
                "(e.g. logs/<experiment>/<run>), "
                "or an absolute path.".format(raw, root)
            )
        load_run = load_run_resolved

    if checkpoint == -1:
        models = _model_files(load_run)
        if not models:
            raise ValueError(
                "No model_*.pt in "
                + load_run
                + ". Specify --checkpoint N or choose another --load_run."
            )
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.seed is not None:
            env_cfg.seed = args.seed
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if hasattr(env_cfg.control, "control_path"):
            env_cfg.control.control_path = args.control_path
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "anymal_c_flat",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--control_path",
            "type": str,
            "default": "vmc_pd",
            "help": "wheel_legged_vmc*: virtual-leg loop — default vmc_pd (RL); use vmc_lqr for LQR. Overrides cfg.",
        },
        {
            "name": "--lqr_demo",
            "action": "store_true",
            "default": False,
            "help": "play.py only: single robot, no RL checkpoint; model LQR virtual-leg control (forces vmc_lqr).",
        },
        {
            "name": "--runtime_log_interval",
            "type": int,
            "default": 50,
            "help": "play.py only: print and CSV-log runtime state every N control steps; <=0 disables it.",
        },
        {
            "name": "--runtime_log_path",
            "type": str,
            "help": "play.py only: CSV path for runtime state diagnostics. Defaults to logs/<experiment>/runtime_state_<task>.csv.",
        },
        {
            "name": "--play_match_train",
            "action": "store_true",
            "default": False,
            "help": "play.py only: keep cfg noise/domain_rand as in training (skip play defaults that zero randomization).",
        },
        {
            "name": "--play_height_assist",
            "type": float,
            "default": 0.0,
            "help": "play.py only: add gain*(cmd_height-base_height) to L0 action channels (1,4) after policy; 0 disables. Try 2~8 for wheel_legged_vmc*.",
        },
        {
            "name": "--play_height_assist_warmup",
            "type": int,
            "default": 300,
            "help": "play.py only: warmup steps before enabling play_height_assist.",
        },
        {
            "name": "--play_plot_interval",
            "type": int,
            "default": 20,
            "help": "play.py only: update live plot every N steps (default 20). Set 1 for per-step updates, <=0 to disable.",
        },
        {
            "name": "--play_num_envs",
            "type": int,
            "default": 0,
            "help": "play.py only: override number of envs for play (0=auto: min(cfg.num_envs,4) for plane, min(cfg.num_envs,2) for trimesh).",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Run name suffix in log folder; omit or empty uses timestamp YYYYmmdd_HHMMSS_mmm.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
        {
            "name": "--exptid",
            "type": str,
            "default": "",
            "help": "Appended to log folder after run_name; empty adds random _XXXX.",
        },
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, "memory_a"):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_1.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(
            f"hidden_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )
        self.register_buffer(
            f"cell_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_lstm_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
