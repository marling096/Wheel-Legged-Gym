from wheel_legged_gym.envs.wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)


class Swpu2026VMCFlatCfg(WheelLeggedVMCFlatCfg):
    class terrain(WheelLeggedVMCFlatCfg.terrain):
        static_friction = 1.5
        dynamic_friction = 1.2

    class sim(WheelLeggedVMCFlatCfg.sim):
        class physx(WheelLeggedVMCFlatCfg.sim.physx):
            num_position_iterations = 8
            num_velocity_iterations = 4

    class init_state(WheelLeggedVMCFlatCfg.init_state):
        pos = [0.0, 0.0, 0.31]
        default_joint_angles = {
            "lf0_Joint": 0.0,
            "lf1_Joint": 0.0,
            "left_wheel_joint": 0.0,
            "rf0_Joint": 0.0,
            "rf1_Joint": 0.0,
            "right_wheel_joint": 0.0,
        }

    class asset(WheelLeggedVMCFlatCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/swpu2026/urdf/swpu2026.urdf"
        name = "Swpu2026"
        foot_name = "Wheel"
        forward_vec = [0.0, -1.0, 0.0]

        offset = 0.0
        l1 = 0.20999687979586745
        l2 = 0.24999843359509277
        wheel_radius = 0.06995

        hip_dof_indices = [0, 3]
        knee_dof_indices = [1, 4]
        wheel_dof_indices = [2, 5]

        hip_angle_signs = [-1.0, 1.0]
        knee_angle_signs = [-1.0, -1.0]
        hip_torque_signs = [-1.0, 1.0]
        knee_torque_signs = [-1.0, -1.0]
        wheel_vel_signs = [1.0, 1.0]

        # swpu2026's leg links are exported in the base Y-Z plane with X-axis joints.
        # These offsets map its zero-pose link vectors into the VMC 2-D leg plane.
        hip_angle_offsets = [2.8109182017715604, 2.8109182017715604]
        knee_angle_offsets = [-2.158280908636402, -2.158280908636402]

        penalize_contacts_on = ["lf", "rf", "base"]
        terminate_after_contacts_on = ["base"]
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        friction = 2.0
        dof_velocity_limits = {
            "f0": 100.0,
            "f1": 100.0,
            "wheel": 100.0,
        }

    class commands(WheelLeggedVMCFlatCfg.commands):
        class ranges(WheelLeggedVMCFlatCfg.commands.ranges):
            height = [0.295, 0.305]

    class domain_rand(WheelLeggedVMCFlatCfg.domain_rand):
        randomize_friction = False
        friction_range = [2.0, 2.0]
        randomize_restitution = False
        randomize_base_mass = False
        randomize_inertia = False
        randomize_base_com = False
        push_robots = False
        randomize_Kp = False
        randomize_Kd = False
        randomize_motor_torque = False
        randomize_default_dof_pos = False
        randomize_action_delay = False

    class control(WheelLeggedVMCFlatCfg.control):
        action_limits = [
            [-0.25, 0.0, 0.0, -0.25, 0.0, 0.0],
            [0.25, 0.2, 0.0, 0.25, 0.2, 0.0],
        ]
        action_scale_vel = 0.0
        command_wheel_vel_gain = 1.0
        l0_offset = 0.245
        feedforward_force = 60.0
        kp_theta = 55.0
        kd_theta = 3.0
        kp_l0 = 1200.0
        kd_l0 = 25.0
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 6.0}

    class rewards(WheelLeggedVMCFlatCfg.rewards):
        base_height_target = 0.30
        tracking_sigma = 0.12

        class scales(WheelLeggedVMCFlatCfg.rewards.scales):
            base_height = 3.0
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 0.5
            tracking_wheel_vel = 3.0
            tracking_ang_vel = 0.5
            orientation = -30.0
            ang_vel_xy = -0.25
            torques = -5.0e-5
            action_rate = -0.02
            action_smooth = -0.02
            vmc_leg_action = -0.3
            vmc_leg_motion = -0.08
            vmc_leg_symmetry = -4.0
            vmc_nominal_leg = -25.0

    class normalization(WheelLeggedVMCFlatCfg.normalization):
        clip_actions = 3.0


class Swpu2026VMCFlatCfgPPO(WheelLeggedVMCFlatCfgPPO):
    class runner(WheelLeggedVMCFlatCfgPPO.runner):
        experiment_name = "swpu2026_vmc_flat"
        max_iterations = 500
