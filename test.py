import time
import numpy as np
import mujoco
import mujoco.viewer

from planner.rrt_planner import RRTPlanner
from planner.rrt_core import get_ctrl_indices, get_qpos_indices, get_qvel_indices

XML = "/home/kchen/MLAI/point-robot-imitation-learning/point_robot_nav.xml"
STEPS = 5  # must match planner.steps_per_action

# Planner + one randomized-start trajectory (node-level actions)
planner = RRTPlanner(xml_path=XML, steps_per_action=STEPS, time_limit_seconds=30.0)
traj = planner.collect(num_trajectories=1, seed=123, randomize_start=True, min_plan_len=1,
                       show_progress=False, per_step=False)[0]
states = traj["states"]    # [K+1, 4]
actions = traj["actions"]  # [K, 2]

# Build sim
m = planner.model
d = mujoco.MjData(m)
qi = get_qpos_indices(m)
vi = get_qvel_indices(m)
ci = get_ctrl_indices(m)

# Initialize to recorded start state
s0 = states[0]
print(s0)
print(states)
d.qpos[qi] = s0[:len(qi)]
d.qvel[vi] = s0[len(qi):len(qi)+len(vi)]
mujoco.mj_forward(m, d)

# rrt_core-style viewer loop
with mujoco.viewer.launch_passive(m, d) as viewer:
    it = 0
    while viewer.is_running():
        for _ in range(STEPS):
            d.ctrl[ci] = actions[it]
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)
        it += 1
        if it >= len(actions):
            break