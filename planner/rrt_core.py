import re
import sys
import json
import glob
import time

import numpy as np
from scipy.spatial import KDTree

import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# def get_objq_indices(model, obj_name):
#     jnt = model.joint(model.body(obj_name).jntadr[0])
#     qpos_inds = np.array(range(jnt.qposadr[0], jnt.qposadr[0] + len(jnt.qpos0)))
#     return qpos_inds


class Tree():

    def __init__(self, q, u):
        self.children = []
        self.configuration = q
        self.control = u
        self.parent = None

    def add_child(self, child_ptr):
        child_ptr.parent = self
        self.children.append(child_ptr)

    def make_point_list(self):  #Run from the root
        #Make list of each point config
        res = [self]
        for child in self.children:
            res += child.make_point_list()
        return res

    def find_nearest_neighbor(self, target):
        point_list = self.make_point_list()
        min_val = float("inf")
        min_tree = None
        for point in point_list:
            dist = np.linalg.norm(np.subtract(target, point.configuration[:2]))
            if dist < min_val:
                min_val = dist
                min_tree = point
        if min_tree == None:
            print("Min tree returned None!")
        return min_tree

    def get_plan(self):
        plan = [self]
        if self.parent is None:
            return plan
        return self.parent.get_plan() + plan

    #No node has two parents as a node is a control off of a starting point in continuous space. There are no loops either because we are not doing any reconnecting. 
    def plot_tree(self):
        #Plot each point with a scatterplot
        #Then plot a line between
        data = self.make_point_list()
        points = [[], []]
        for i in range(len(data)):
            #Make [x1, x2], [y1, y2] structure to plot lines
            points[0].append(data[i].configuration[0])
            points[1].append(data[i].configuration[1])
            if len(data[i].children) > 0:
                for j in range(len(data[i].children)):
                    plt.plot([data[i].configuration[0], data[i].children[j].configuration[0]], [data[i].configuration[1], data[i].children[j].configuration[1]]) #Plot lines here
        plt.scatter(points[0], points[1], s=0.1) #Scatter here
        plt.savefig('tree.png')
        plt.show()


class KDTreeIndex:
    def __init__(self, rebuild_every=64):
        self.points = []
        self.nodes = []
        self.kdtree = None
        self.rebuild_every = rebuild_every
        self.num_since_rebuild = 0

    def add(self, node):
        # store only x, y for nearest-neighbor in configuration space
        self.points.append(node.configuration[:2])
        self.nodes.append(node)
        self.num_since_rebuild += 1

    def _maybe_rebuild(self, force=False):
        if self.kdtree is None or self.num_since_rebuild >= self.rebuild_every or force:
            if len(self.points) > 0:
                self.kdtree = KDTree(np.asarray(self.points))
            else:
                self.kdtree = None
            self.num_since_rebuild = 0

    def nearest(self, q):
        self._maybe_rebuild()
        if self.kdtree is None:
            return None
        _, ii = self.kdtree.query([q], k=1)
        return self.nodes[ii[0]]


def get_qpos_indices(model, joints=['joint_x', 'joint_y']):
    qpos_inds = np.array([model.joint(j).qposadr[0] for j in joints])
    return qpos_inds


def get_qvel_indices(model, joints=['joint_x', 'joint_y']):
    qvel_inds = np.array([model.joint(j).dofadr[0] for j in joints])
    return qvel_inds


def set_qpos_values(mdata, joint_inds, joint_vals):
    mdata.qpos[joint_inds] = joint_vals


def get_qpos_values(mdata, joint_inds):
    return mdata.qpos[joint_inds]


def set_qvel_values(mdata, joint_inds, joint_vels):
    mdata.qvel[joint_inds] = joint_vels


def get_qvel_values(mdata, joint_inds):
    return mdata.qvel[joint_inds]


def get_ctrl_indices(model, motors=['actuator_x', 'actuator_y']):
    ctrl_inds = np.array([model.actuator(motor).id for motor in motors])
    return ctrl_inds


def set_ctrl_values(mdata, ctrl_inds, ctrl_vals):
    mdata.ctrl[ctrl_inds] = ctrl_vals


def colliding_body_pairs(contact, model):
    pairs = [
        (
            model.body(model.geom(c.geom1).bodyid[0]).name,
            model.body(model.geom(c.geom2).bodyid[0]).name
        ) for c in contact
    ]
    return pairs


def nearest_neighbor(q, points, index=True):
    tree = KDTree(points)
    dd, ii = tree.query([q], k=1)
    if index:
        return ii[0]  # index
    else:
        return points[ii[0]]  # point


def is_in_collision(
    model,
    mdata,
    joint_inds,
    joint_vals,
):
    # set the robot configuration to a certain state
    set_qpos_values(mdata, joint_inds, joint_vals)
    # check collision
    mujoco.mj_step1(model, mdata)
    # cols = colliding_body_pairs(mdata.contact, model)
    return len(mdata.contact) > 1


def in_goal(pose):
    """
    x:-0.13 to 1.03
    y:-0.3 to 0.3
    """
    return 0.8 < pose[0] < 1.1 and -0.3 < pose[1] < 0.3


def KRRT(m, d, STEPS):
    t0 = time.time()
    T_max = 30

    mujoco.mj_step(m, d)
    qi = get_qpos_indices(m)
    vi = get_qvel_indices(m)
    ci = get_ctrl_indices(m)
    pose = get_qpos_values(d, qi)
    vel = get_qvel_values(d, vi)
    root = Tree([*pose, *vel], [0, 0])
    # KD-tree index for fast nearest-neighbor lookups
    nn_index = KDTreeIndex(rebuild_every=64)
    nn_index.add(root)
    progress = 0
    while time.time() - t0 < T_max:
        progress += 1
        if progress % 100 == 0:
            print(f"Progress: {progress} iters in {time.time() - t0} seconds")
        rand_config = [
            np.random.uniform(-0.13, 1.03),
            np.random.uniform(-0.3, 0.3)
        ]
        nn = nn_index.nearest(rand_config)
        rand_control = np.random.uniform(-1, 1, len(ci))
        set_qpos_values(d, qi, nn.configuration[:2])
        set_qvel_values(d, vi, nn.configuration[2:])
        is_valid = True
        for i in range(STEPS):
            set_ctrl_values(d, ci, rand_control)
            mujoco.mj_step(m, d)
            is_valid &= len(d.contact) == 0
            if not is_valid:
                break
        if is_valid:
            pose = get_qpos_values(d, qi)
            vel = get_qvel_values(d, vi)
            newnode = Tree([*pose, *vel], rand_control)
            nn.add_child(newnode)
            nn_index.add(newnode)
            if in_goal(pose):
                root.plot_tree()
                return newnode.get_plan()
    root.plot_tree()
    return []


if __name__ == '__main__':
    m = mujoco.MjModel.from_xml_path('/home/kchen/MLAI/point-robot-imitation-learning/point_robot_nav.xml')
    d = mujoco.MjData(m)
    ctrl_inds = get_ctrl_indices(m)

    STEPS = 5
    # plan loop
    plan = KRRT(m, d, STEPS)
    print("Plan :", len(plan))

    # visualize planning trees
    #TODO

    # reset simulation
    mujoco.mj_resetData(m, d)

    # execute loop
    if len(plan) > 0:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            it = 0
            while viewer.is_running():
                for i in range(STEPS):
                    # control policy
                    set_ctrl_values(d, ctrl_inds, plan[it].control)

                    # physics
                    mujoco.mj_step(m, d)

                    # Pick up changes to the physics state, apply perturbations, update options from GUI.
                    viewer.sync()
                    time.sleep(m.opt.timestep)
                it += 1
                if it >= len(plan):
                    input('Close Me!')
                    sys.exit()
    else:
        print("No Plan Found!")
