import time
import numpy as np
import mujoco

try:
    # Prefer absolute import when used as a package
    from planner.rrt_core import (
        Tree,
        KDTreeIndex,
        get_qpos_indices,
        get_qvel_indices,
        get_ctrl_indices,
        set_qpos_values,
        set_qvel_values,
        set_ctrl_values,
        get_qpos_values,
        get_qvel_values,
        in_goal,
    )
except Exception:
    # Fallback for running this module directly
    from rrt_core import (
        Tree,
        KDTreeIndex,
        get_qpos_indices,
        get_qvel_indices,
        get_ctrl_indices,
        set_qpos_values,
        set_qvel_values,
        set_ctrl_values,
        get_qpos_values,
        get_qvel_values,
        in_goal,
    )

# Optional progress bar (fallback to no-op if tqdm not installed)
try:
    from tqdm import tqdm as _tqdm
except Exception:
    class _tqdm:  # type: ignore
        def __init__(self, total=None, desc=None, dynamic_ncols=True):
            self.total = total
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_postfix_str(self, s):
            pass
        def close(self):
            pass

class RRTPlanner:
    def __init__(self, xml_path, steps_per_action=5, time_limit_seconds=30.0, kdtree_rebuild_every=64):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.steps_per_action = steps_per_action
        self.time_limit_seconds = time_limit_seconds
        self.kdtree_rebuild_every = kdtree_rebuild_every
        # Cached indices for efficiency
        self.qpos_idx = get_qpos_indices(self.model)
        self.qvel_idx = get_qvel_indices(self.model)
        self.ctrl_idx = get_ctrl_indices(self.model)

    def _build_plan(self, data, goal_fn, rng, progress_print_every=None, plot_tree=False):
        t0 = time.time()
        mujoco.mj_step(self.model, data)
        pose = get_qpos_values(data, self.qpos_idx)
        vel = get_qvel_values(data, self.qvel_idx)
        root = Tree([*pose, *vel], [0, 0])
        nn_index = KDTreeIndex(rebuild_every=self.kdtree_rebuild_every)
        nn_index.add(root)
        progress = 0
        while time.time() - t0 < self.time_limit_seconds:
            progress += 1
            if progress_print_every is not None and progress % progress_print_every == 0:
                print(f"Progress: {progress} iters in {time.time() - t0:.2f}s")
            rand_config = [
                rng.uniform(-0.13, 1.03),
                rng.uniform(-0.3, 0.3)
            ]
            nn = nn_index.nearest(rand_config)
            if nn is None:
                continue
            rand_control = rng.uniform(-1, 1, len(self.ctrl_idx))
            set_qpos_values(data, self.qpos_idx, nn.configuration[:2])
            set_qvel_values(data, self.qvel_idx, nn.configuration[2:])
            is_valid = True
            for _ in range(self.steps_per_action):
                set_ctrl_values(data, self.ctrl_idx, rand_control)
                mujoco.mj_step(self.model, data)
                is_valid &= len(data.contact) == 0
                if not is_valid:
                    break
            if is_valid:
                pose = get_qpos_values(data, self.qpos_idx)
                vel = get_qvel_values(data, self.qvel_idx)
                newnode = Tree([*pose, *vel], rand_control)
                nn.add_child(newnode)
                nn_index.add(newnode)
                if goal_fn(pose):
                    if plot_tree:
                        root.plot_tree()
                    return newnode.get_plan()
        if plot_tree:
            root.plot_tree()
        return []

    def plan_once(self, seed=None, start_pose=None, goal_fn=None, progress_print_every=None, plot_tree=False):
        if goal_fn is None:
            goal_fn = in_goal
        rng = np.random.default_rng(seed)
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        if start_pose is not None:
            set_qpos_values(data, self.qpos_idx, np.asarray(start_pose))
            set_qvel_values(data, self.qvel_idx, np.zeros_like(self.qvel_idx, dtype=float))
        plan = self._build_plan(
            data=data,
            goal_fn=goal_fn,
            rng=rng,
            progress_print_every=progress_print_every,
            plot_tree=plot_tree
        )
        return plan

    def execute_plan(self, plan, per_step=False):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        states = []
        actions = []

        # set initial state
        plan_start_pose = plan[0].configuration[:2]
        plan_start_vel = plan[0].configuration[2:]
        set_qpos_values(data, self.qpos_idx, plan_start_pose)
        set_qvel_values(data, self.qvel_idx, plan_start_vel)
        mujoco.mj_forward(self.model, data)

        # execute plan
        if per_step:
            # Log at per-physics-step granularity: states length = total_steps + 1, actions length = total_steps
            # initial state
            pose = get_qpos_values(data, self.qpos_idx).copy()
            vel = get_qvel_values(data, self.qvel_idx).copy()
            states.append(np.concatenate([pose, vel]))
            for node in plan:
                for _ in range(self.steps_per_action):
                    set_ctrl_values(data, self.ctrl_idx, node.control)
                    mujoco.mj_step(self.model, data)
                    # state after step; action used during this step
                    pose = get_qpos_values(data, self.qpos_idx).copy()
                    vel = get_qvel_values(data, self.qvel_idx).copy()
                    states.append(np.concatenate([pose, vel]))
                    actions.append(np.asarray(node.control).copy())
            return np.asarray(states), np.asarray(actions)
        else:
            # Original behavior: one action per plan node (applied for steps_per_action steps)
            for node in plan:
                pose = get_qpos_values(data, self.qpos_idx).copy()
                vel = get_qvel_values(data, self.qvel_idx).copy()
                states.append(np.concatenate([pose, vel])) # s_t
                actions.append(np.asarray(node.control).copy()) # a_t
                for _ in range(self.steps_per_action):
                    set_ctrl_values(data, self.ctrl_idx, node.control)
                    mujoco.mj_step(self.model, data)
            pose = get_qpos_values(data, self.qpos_idx).copy()
            vel = get_qvel_values(data, self.qvel_idx).copy()
            states.append(np.concatenate([pose, vel]))
            return np.asarray(states), np.asarray(actions)

    def collect(self, num_trajectories, seed=None, randomize_start=False, start_sampler=None, goal_fn=None, min_plan_len=1, progress_every=10, show_progress=True, per_step=False):
        if goal_fn is None:
            goal_fn = in_goal
        rng = np.random.default_rng(seed)
        trajectories = []
        successes = 0
        attempts = 0
        pbar = _tqdm(total=num_trajectories, desc="Collect (RRTPlanner)", dynamic_ncols=True) if show_progress else None
        while len(trajectories) < num_trajectories:
            attempts += 1
            if start_sampler is not None:
                start_pose = start_sampler(rng)
            elif randomize_start:
                start_pose = np.array([rng.uniform(-0.13, 1.03), rng.uniform(-0.3, 0.3)])
            else:
                start_pose = None
            plan = self.plan_once(
                seed=rng.integers(0, 2**31 - 1),
                start_pose=start_pose,
                goal_fn=goal_fn,
                progress_print_every=None,
                plot_tree=False
            )
            if len(plan) >= min_plan_len:
                states, actions = self.execute_plan(plan, per_step=per_step)
                trajectories.append({"states": states, "actions": actions})
                successes += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"succ/att {successes}/{attempts}")
            if progress_every is not None and attempts % progress_every == 0:
                print(f"Collected {len(trajectories)}/{num_trajectories} trajectories "
                      f"(success rate: {successes}/{attempts})")
        if pbar is not None:
            pbar.close()
        return trajectories


if __name__ == "__main__":
    XML_PATH = "/home/kchen/MLAI/point-robot-imitation-learning/point_robot_nav.xml"
    planner = RRTPlanner(xml_path=XML_PATH, steps_per_action=5, time_limit_seconds=30.0)
    plan = planner.plan_once(seed=42)
    print("Plan length:", len(plan))
    plan_batch = planner.collect(num_trajectories=3, seed=123, randomize_start=True, min_plan_len=1, progress_every=1, show_progress=True,
        per_step=True
    )
    print("Collected trajectories:", len(plan_batch))
