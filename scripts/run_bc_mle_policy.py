import os
import sys
import time
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer

from policy.policy import STATE_DIM, ACTION_DIM
from policy.actor_critic_policy import Actor as Policy

from planner.rrt_core import in_goal, get_qpos_indices, get_qvel_indices, get_ctrl_indices


DEBUG_START_POSE = np.array([0, 0.2])
DEFAULT_POLICY_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "bc_mle_policy.pth",
)

def parse_args():
    p = argparse.ArgumentParser(description="Run a trained policy in MuJoCo")
    default_xml = os.path.join(os.path.dirname(__file__), "../scenes", "point_robot_nav.xml")
    p.add_argument("--xml_path", type=str, default=default_xml, help="Path to MuJoCo XML")
    p.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH, help="Path to saved policy weights")
    p.add_argument("--randomize_start", action="store_true", help="Randomize starting pose")
    p.add_argument("--debug_start", action="store_true", help="Debug starting pose")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for randomized start")
    p.add_argument("--speed", type=float, default=5.0, help="Playback speed multiplier")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.xml_path):
        raise FileNotFoundError(f"XML not found: {args.xml_path}")
    if not os.path.exists(args.policy_path):
        raise FileNotFoundError(f"Policy not found: {args.policy_path}")

    # Load Gaussian MLE BC policy (CPU for simplicity)
    state_dict = torch.load(
        args.policy_path,
        map_location="cpu",
        weights_only=True,
    )
    policy = Policy(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    policy.load_state_dict(state_dict)
    policy.eval()

    # Build sim
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    qpos_idx = get_qpos_indices(model)
    qvel_idx = get_qvel_indices(model)
    ctrl_idx = get_ctrl_indices(model)

    # Optionally randomize start
    if args.randomize_start:
        rng = np.random.default_rng(args.seed)
        start_pose = np.array([rng.uniform(-0.13, 1.03), rng.uniform(-0.3, 0.3)], dtype=float)
        data.qpos[qpos_idx] = start_pose
        data.qvel[qvel_idx] = 0.0
        mujoco.mj_forward(model, data)

    if args.debug_start:
        start_pose = DEBUG_START_POSE
        data.qpos[qpos_idx] = start_pose
        data.qvel[qvel_idx] = 0.0
        mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Build state: [x, y, vx, vy]
            pose = data.qpos[qpos_idx].copy()
            vel = data.qvel[qvel_idx].copy()
            state = np.concatenate([pose, vel], axis=0).astype(np.float32)
            with torch.no_grad():
                # Sample an action from the learned Gaussian policy
                tensor_state = torch.from_numpy(state).unsqueeze(0)
                action_tensor, _ = policy.sample_action(tensor_state)
                action = action_tensor.squeeze(0).cpu().numpy()
            data.ctrl[ctrl_idx] = action

            mujoco.mj_step(model, data)
            viewer.sync()

            # check if goal
            if in_goal(pose):
                print("Reached goal.")
                break

            time.sleep(max(0.0, model.opt.timestep / max(1e-6, args.speed)))

        time.sleep(2)


if __name__ == "__main__":
    main()


