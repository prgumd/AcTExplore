import math
import time

import cv2
import gym
import hydra
import numpy as np
import pybullet as p
import pybulletX as px
from gym import spaces
from scipy.spatial.transform import Rotation as R
import random

import misc.tacto as tacto
import matplotlib.pyplot as plt
from dataclasses import dataclass

from reconstruction import Recons
# import line_profiler
import threading
import os

@dataclass
class MinMax:
    min: float = None
    max: float = None


class TactoEnv(gym.Env):
    def __init__(self, rl_cfg, digit_base=None):
        ##### Action space #####
        self.action_space = spaces.Discrete(rl_cfg.action.action_num)
        self.trans_step = rl_cfg.action.trans_step
        self.rot_step = rl_cfg.action.rotate_step
        self.rl_cfg = rl_cfg
        self.camera_offset = 0
        self.action_to_direction = {
            0: np.array([self.trans_step, 0, 0, 0, 0, 0]),
            1: np.array([-self.trans_step, 0, 0, 0, 0, 0]),  # go forward
            2: np.array([0, self.trans_step, 0, 0, 0, 0]),
            3: np.array([0, -self.trans_step, 0, 0, 0, 0]),
            4: np.array([0, 0, self.trans_step, 0, 0, 0]),
            5: np.array([0, 0, -self.trans_step, 0, 0, 0]),
            6: np.array([0, 0, 0, self.rot_step, 0, 0]),
            7: np.array([0, 0, 0, -self.rot_step, 0, 0]),
            8: np.array([0, 0, 0, 0, self.rot_step, 0]),
            9: np.array([0, 0, 0, 0, -self.rot_step, 0]),
            10: np.array([0, 0, 0, 0, 0, self.rot_step]),
            11: np.array([0, 0, 0, 0, 0, -self.rot_step])
        }
        self.digit_bg = cv2.imread(rl_cfg.tacto.background)
        self.digit_bg = cv2.resize(self.digit_bg, (rl_cfg.tacto.width, rl_cfg.tacto.height))
        if rl_cfg.action.action_num == 13:
            self.action_to_direction[12] = None  # will fill again next to go back last touch}

        ##### Observation space #####
        ##### Observation space must be defined by spaces
        if rl_cfg.state.input_type == 'depth' or rl_cfg.state.input_type == 'TTA':
            self.observation_space = spaces.Box(low=-1, high=1, shape=(rl_cfg.tacto.height, rl_cfg.tacto.width),
                                                dtype=np.float32)
        elif rl_cfg.state.input_type == "TTS":
            self.observation_space = spaces.Box(low=-1, high=1, shape=(rl_cfg.tacto.height, rl_cfg.tacto.width * 5),
                                                dtype=np.float32)

        elif rl_cfg.state.input_type == "TIS":
            self.observation_space = spaces.Box(low=-1, high=1, shape=(rl_cfg.tacto.height, rl_cfg.tacto.width * 5, 3),
                                                dtype=np.float32)

        elif rl_cfg.state.input_type == "TDS":
            self.observation_space = spaces.Box(low=-1, high=1, shape=(rl_cfg.tacto.height, rl_cfg.tacto.width * 5),
                                                dtype=np.float32)

        ##### State variable #####
        self.state = None
        self.color = None
        self.depth = None
        self.pointcloud = None

        ##### Reward variable #####
        self.reward = 0
        self.accumulated_reward = 0
        self.reward_weight = rl_cfg.reward.weight
        self.diff_norm = 0

        ##### Short memory definition #####
        self.short_history = []
        self.short_mem_size = rl_cfg.state.short_mem_size
        self.no_touch_count = 0

        ##### For Ablation Study 
        self.novelty_buffer = []
        self.novelty_threshold = rl_cfg.reward.novelty_threshold * self.trans_step
        self.is_novel = False
        self.max_novel_size = rl_cfg.state.max_novel_size
        self.print_once = True

        ##### Pose memory #####
        self.curr_pos, self.curr_ori = None, None
        self.pose_action_history = []

        ##### Etc variable #####
        self.horizon_length = rl_cfg.termination.horizon_length
        self.horizon_counter = 0
        self._physics_client_id = -1
        self.is_touching = False
        self.metric_history = []

        ##### Load URDF files #####
        self.digits = tacto.Sensor(**rl_cfg.tacto)

        self._physics_client_id = px.init()
        p.setGravity(0, 0, 0)
        p.resetDebugVisualizerCamera(**rl_cfg.pybullet_camera)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # Add object to pybullet and tacto simulator
        obj_idx = random.randint(0, len(rl_cfg.environment.object.urdf_path) - 1)
        obj_path = rl_cfg.environment.object.urdf_path[obj_idx]
        self.object_name = obj_path.split("/")[-1].split(".")[0]
        self.obj = px.Body(urdf_path=obj_path, **rl_cfg.environment.pose)
        self.digits.add_body(self.obj)
        obj_AABB = p.getAABB(self.obj.id)
        # Create and initialize DIGIT
        if not digit_base:
            digit_size = [0.0323276, 0.027, 0.0340165]
            obj_center = np.mean(obj_AABB, axis=0)
            obj_center -= np.array([0, 0, digit_size[2]]) / 2
            self.digit_body = px.Body(**rl_cfg.digit, base_position=obj_center)
        else:
            self.digit_body = px.Body(**rl_cfg.digit, base_position=digit_base)

        self.digits.add_camera(self.digit_body.id, [-1])
        ##### Workspace boundary #####
        self.threshold = self._workspace_bounds()
        digit_start_pose = np.array(p.getBasePositionAndOrientation(self.digit_body.id))
        digit_start_pose[0] = np.array(digit_start_pose[0])
        digit_start_pose[0][0] = self.threshold["x"].max
        self.digit_body.set_base_pose(*digit_start_pose)  # moving digit to workspace boundary as init pose
        self.init_pose = p.getBasePositionAndOrientation(self.digit_body.id)

        ##### Reconstruction #####
        self.realtime = rl_cfg.visualization.realtime
        self.render = rl_cfg.visualization.render
        self.render_mode = rl_cfg.visualization.render
        self.recon = Recons(rl_cfg.environment, obj_idx, self.threshold, 0.002)

        # Generate the Gaussian kernel for compute area
        sigma = 10
        kernel = np.zeros((rl_cfg.tacto.height, rl_cfg.tacto.width))
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                x = i - kernel.shape[0] // 2
                y = j - kernel.shape[1] // 2
                kernel[i][j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        self.gaussian_kernel = kernel / np.sum(kernel)
        if not digit_base:
            self._go_first_touch()
        camera_thread = threading.Thread(target=self.rotate_camera)
        camera_thread.start()

    def _workspace_bounds(self):
        digit_AABB = p.getAABB(self.digit_body.id)
        obj_AABB = p.getAABB(self.obj.id)
        x_thres = MinMax()
        y_thres = MinMax()
        z_thres = MinMax()
        x_thres.min = min(obj_AABB[0][0], obj_AABB[1][0]) - ((digit_AABB[1][0] - digit_AABB[0][0]) * 2)
        x_thres.max = max(obj_AABB[0][0], obj_AABB[1][0]) + ((digit_AABB[1][0] - digit_AABB[0][0]) * 2)
        y_thres.min = min(obj_AABB[0][1], obj_AABB[1][1]) - ((digit_AABB[1][1] - digit_AABB[0][1]) * 2)
        y_thres.max = max(obj_AABB[0][1], obj_AABB[1][1]) + ((digit_AABB[1][1] - digit_AABB[0][1]) * 2)
        z_thres.min = min(obj_AABB[0][2], obj_AABB[1][2]) - ((digit_AABB[1][2] - digit_AABB[0][2]) * 2)
        z_thres.max = max(obj_AABB[0][2], obj_AABB[1][2]) + ((digit_AABB[1][2] - digit_AABB[0][2]) * 2)
        return {"x": x_thres, "y": y_thres, "z": z_thres}

    def _get_observation(self):
        """
        Get the observation state for the environment.

        Returns:
            observation (numpy.ndarray): The observation state.
        """
        observation = []
        self.color, self.depth, self.pointcloud = self.digits.render()
        ##### PointCloud Frame Convertion #####
        rot_x = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
        if len(self.pointcloud) > 0:
            self.pointcloud = np.matmul(self.pointcloud, rot_x) + np.array([0, 0, 0.03])

        ##### Update GUI #####
        self.depth = self.depth[0]
        ##### Update Short History Buffer #####
        self.obs = self.depth
        self.short_history.append(
            {"action": self.action, "reward": self.reward, "pose": (self.curr_pos, self.curr_ori), "obs": self.obs})

        self.is_touching = self._touch_indicator()

        ##### Compute is_touch when we use pointcloud data #####
        if len(self.short_history) > self.short_mem_size:
            self.short_history.pop(0)

        ##### Add loation for visualization #####
        if self.is_touching:
            self.recon.insert_data(self.short_history[-1], self.pointcloud)
        if self.realtime:
            self.recon.realtime_visualize(cv2.flip(self.obs, 1))

        ##### Compute State/Observation State #####
        if self.rl_cfg.state.input_type == 'depth':
            # cv2.waitKey(1)
            observation = np.clip(np.array(self.obs) / self.digits.zrange, 0, 1)
            # cv2.imshow("depth obs", (observation * 255).astype("uint8"))

        if self.rl_cfg.state.input_type == 'TTS':
            mem_depth = []
            for h in self.short_history[-5:]:
                normalized_depth = np.clip(h["obs"] / self.digits.zrange, 0, 1)
                mem_depth.append(normalized_depth)
            if len(mem_depth) != 5:
                mem_depth.append(np.repeat(np.zeros_like(mem_depth[0]), 5 - len(mem_depth), axis=1))
            observation = np.concatenate(mem_depth, axis=1)
            obs_color = cv2.applyColorMap((observation * 255).astype("uint8"), cv2.COLORMAP_PLASMA)
            cv2.imshow("concat_obs", obs_color)
            cv2.waitKey(1)
        if self.rl_cfg.state.input_type == 'TTA':
            mem_depth = []
            for h in self.short_history[-5:]:
                normalized_depth = np.clip(h["obs"] / self.digits.zrange, 0, 1)
                mem_depth.append(normalized_depth)
            observation = np.average(mem_depth, weights=[1 + i / 50 for i in range(len(mem_depth))], axis=0)
            cv2.imshow("mem_depth", (observation * 255).astype("uint8"))
            cv2.waitKey(1)

        return observation

    def _compute_reward(self):
        """
        Computes the reward based on the current state and configuration.

        Returns:
            float: The computed reward value.
        """
        ## Assumes self.is_touching is updated in compute_observation in advance
        if self.rl_cfg.reward.type == "TM":
            reward = 1 if self.is_touching else 0

            min_movement = 2 * [np.iinfo(int).max]
            for h in self.short_history:
                trans_diff = np.linalg.norm(np.subtract(self.curr_pos, h["pose"][0]))
                ori_diff = np.arccos(2 * np.vdot(self.curr_ori, h["pose"][1]) ** 2 - 1)
                if (h["reward"] >= 0 or h["reward"] == 0.15 * self.rl_cfg.reward.visited_state_penalty) and trans_diff < \
                        min_movement[0] and ori_diff < min_movement[1]:
                    # Close point which has already received a reward
                    min_movement[0] = trans_diff
                    min_movement[1] = ori_diff

            if min_movement[0] <= self.trans_step * 0.75 and min_movement[1] <= self.rot_step * 0.75:
                reward = self.rl_cfg.reward.visited_state_penalty

        if self.rl_cfg.reward.type == "AM":
            reward = self._compute_area()
            min_movement = 2 * [np.iinfo(int).max]
            for h in self.short_history[:-1]:
                trans_diff = np.linalg.norm(np.subtract(self.curr_pos, h["pose"][0]))
                ori_diff = np.arccos(min(1, 2 * np.vdot(self.curr_ori, h["pose"][1]) ** 2 - 1))

                if (h["reward"] > 0 or h["reward"] == 0.15 * self.rl_cfg.reward.visited_state_penalty) and trans_diff <= \
                        min_movement[0] and ori_diff <= min_movement[1]:
                    # Close point which has already received a reward
                    min_movement[0] = trans_diff
                    min_movement[1] = ori_diff

            if min_movement[0] <= self.trans_step * 0.75 and min_movement[1] <= self.rot_step * 0.75:
                reward = self.rl_cfg.reward.visited_state_penalty

        elif self.rl_cfg.reward.type == "AC":
            old_coverage = self.recon.coverage
            self.recon.compute_coverage()

        elif self.rl_cfg.reward.type == "AMB":
            reward_area = self._compute_area()
            reward_explore = 0
            min_movement = 2 * [np.iinfo(int).max]
            for h in self.short_history[:-1]:
                trans_diff = np.linalg.norm(np.subtract(self.curr_pos, h["pose"][0]))
                ori_diff = np.arccos(min(1, 2 * np.vdot(self.curr_ori, h["pose"][1]) ** 2 - 1))

                if (h["reward"] > 0 or h["reward"] == 0.15 * self.rl_cfg.reward.visited_state_penalty) and trans_diff <= \
                        min_movement[0] and ori_diff <= min_movement[1]:
                    # Close point which has already received a reward
                    min_movement[0] = trans_diff
                    min_movement[1] = ori_diff
            if min_movement[0] <= self.trans_step * 0.75 and min_movement[1] <= self.rot_step * 0.75:
                reward_area = self.rl_cfg.reward.visited_state_penalty

            if reward_area > 0:
                n_visit = 0  # we are counting this based on position as it's the same thing with image but harder to find similar state
                trans_thresh = self.trans_step * 2
                rot_thresh = self.rot_step * 6
                for h in self.pose_action_history:
                    pose = h[0]
                    action = h[1]
                    trans_diff = np.linalg.norm(np.subtract(self.curr_pos, pose[0]))
                    ori_diff = np.arccos(min(1, 2 * np.vdot(self.curr_ori, pose[1]) ** 2 - 1))
                    if trans_diff <= trans_thresh and ori_diff <= rot_thresh and action == self.action:
                        n_visit += 1
                reward_explore = (1 / np.sqrt(n_visit))
            reward = 0.15 * reward_area + 0.85 * reward_explore
            if self.no_touch_count >= self.rl_cfg.reward.no_touch_threshold:
                reward = self.rl_cfg.reward.visited_state_penalty
            if self.action == 12:
                reward = self.rl_cfg.reward.visited_state_penalty
            # print(
            #     f"reward #{self.horizon_counter:05}, total: {reward:2f}, area: {reward_area:2f}, exp: {reward_explore:2f}, act: {self.action}")

        self.accumulated_reward += reward
        return reward

    def step_global(self, action):
        """ 
        This function is for visualization
        """
        w_trans, w_rot = p.getBasePositionAndOrientation(self.digit_body.id)  # world coordinates
        delta = self.action_to_direction[action]
        new_pos = w_trans + np.array(delta[:3])
        self.digit_body.set_base_pose(new_pos, w_rot)

        for _ in range(4):
            p.stepSimulation()

        self.curr_pos, self.curr_ori = np.array(p.getBasePositionAndOrientation(self.digit_body.id))  # After simulation
        self.pose_action_history.append(((self.curr_pos, self.curr_ori), action))

        observation = self._get_observation()

        if self.is_touching and self.rl_cfg.action.action_num == 13:
            self.action_to_direction[12] = (self.curr_pos, self.curr_ori)
        ##### Compute Reward #####
        self.reward = self._compute_reward()

        info = self._get_info()
        self.recon.compute_coverage()

        ##### Compute termination condition #####
        termination = self._termination_compute()
        # for _ in range(4):
        #     p.stepSimulation()

        return observation, self.reward, termination, info

    def step(self, action):
        """
        Simulates one step in the environment.

        Args:
            action (int): The action to move the robot

        Returns:
            observation (object): The current observation/state of the environment.
            reward (float): The reward obtained from the action.
            termination (bool): Whether the episode is terminated or not.
            info (dict): Additional information about the environment.
        """
        # self.rotate_camera()
        ##### Apply Action #####
        self.action = action
        self.horizon_counter += 1
        if action == 12:
            new_pos, new_ori = self.action_to_direction[12]
        else:
            new_pos, new_ori = self._action_in_local_frame(self.digit_body.id, action)

        self.digit_body.set_base_pose(new_pos, new_ori)

        for _ in range(2):
            p.stepSimulation()

        self.curr_pos, self.curr_ori = np.array(p.getBasePositionAndOrientation(self.digit_body.id))  # After simulation
        self.pose_action_history.append(((self.curr_pos, self.curr_ori), action))

        ##### Compute State/Observation State #####
        observation = self._get_observation()

        if self.is_touching and self.rl_cfg.action.action_num == 13:
            self.action_to_direction[12] = (self.curr_pos, self.curr_ori)
        ##### Compute Reward #####
        self.reward = self._compute_reward()

        info = self._get_info()
        self.recon.compute_coverage()

        ##### Compute termination condition #####
        termination = self._termination_compute()
        for _ in range(5):
            p.stepSimulation()

        return observation, self.reward, termination, None, info

    def random_pose(self):
        """
        Generates a random pose.

        Returns:
            tuple: A tuple containing the random position (3D array) and random orientation (quaternion).
        """
        pos, ori = self.init_pose
        rnd_pos = np.array(pos)
        rnd_ori = np.array(ori)
        rnd_pos += (0.03 * np.random.random(3)) - 0.015  # 0.03 is sphere radius
        rnd_ori = R.from_quat(rnd_ori)
        rnd_ori = R.from_rotvec(rnd_ori.as_rotvec() + (np.pi / 10 * np.random.random(3) - np.pi / 20))
        return rnd_pos, rnd_ori.as_quat()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the observation and additional information.
        """
        ##### Regenerate Object 
        p.removeBody(self.obj.id)
        p.removeBody(self.digit_body.id)
        obj_idx = random.randint(0, len(self.rl_cfg.environment.object.urdf_path) - 1)
        obj_path = self.rl_cfg.environment.object.urdf_path[obj_idx]
        self.object_name = obj_path.split("/")[-1].split(".")[0]
        self.obj = px.Body(urdf_path=obj_path, **self.rl_cfg.environment.pose)

        ##### Set up digit new pose 
        # new_init_pose = self.random_pose()
        obj_AABB = p.getAABB(self.obj.id)
        digit_size = [0.0323276, 0.027, 0.0340165]
        obj_center = np.mean(obj_AABB, axis=0)
        obj_center -= np.array([0, 0, digit_size[2]]) / 2
        self.digit_body = px.Body(**self.rl_cfg.digit, base_position=obj_center)
        digit_start_pose = np.array(p.getBasePositionAndOrientation(self.digit_body.id))
        digit_start_pose[0] = np.array(digit_start_pose[0])
        digit_start_pose[0][0] = self.threshold["x"].max
        self.digit_body.set_base_pose(*digit_start_pose)  # moving digit to workspace boundary as init pose
        self.init_pose = p.getBasePositionAndOrientation(self.digit_body.id)
        self.digits = tacto.Sensor(**self.rl_cfg.tacto)
        self.digits.add_body(self.obj)
        self.digits.add_camera(self.digit_body.id, [-1])

        ##### Reset Variables
        self.horizon_counter = 0
        self.no_touch_count = 0
        self.accumulated_reward = 0
        self.novelty_buffer.clear()
        self.short_history.clear()
        self.is_novel = False
        self.is_touching = False
        self.print_once = True

        self._go_first_touch()

        self.threshold = self._workspace_bounds()
        self.recon.reset(obj_idx, self.threshold)
        self.metric_history = []

        ##### Get Observation #####
        observation = self._get_observation()

        print('#### Reset Finished ####')

        return observation, None

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_state(self):
        if len(self.short_history) > 1:
            weights = []
            for i in range(1, len(self.short_history) + 2, 1):
                weights.append(i / math.comb(len(self.short_history) + 1, 2))
        return self.digits.render()[1]

    def _get_info(self):
        return {"coverage": self.recon.coverage, "horizon_counter": self.horizon_counter,
                "acc_reward": self.accumulated_reward}

    def _go_first_touch(self):
        while not (self.is_touching):
            self.step(0)
            if self.horizon_counter == 1:
                self.digit_bg = self.color[0]

        self.short_history.clear()
        self.pose_action_history.clear()
        self.novelty_buffer.clear()
        self.no_touch_count = 0
        self.horizon_counter = 0
        print("First Touch")

    def _termination_compute(self):
        """
        Determines whether the evaluation should terminate based on certain conditions.

        Returns:
            bool: True if evaluation should terminate, False otherwise.
        """
        terminate = False
        if (self.curr_pos[0] < self.threshold["x"].min
                or self.curr_pos[0] > self.threshold["x"].max
                or self.curr_pos[1] < self.threshold["y"].min
                or self.curr_pos[1] > self.threshold["y"].max
                or self.curr_pos[2] < self.threshold["z"].min
                or self.curr_pos[2] > self.threshold["z"].max
                or self.horizon_length <= self.horizon_counter):
            print(self.horizon_counter, "out of workspace bounds")
            terminate = True
        if (self.horizon_length <= self.horizon_counter):
            terminate = True
            template = '########## Evaluation Over: Steps {:d}\t, Coverage: {:.2f} %\t, Object:' + self.object_name + '\n'
            print(template.format(self.horizon_counter, self.recon.coverage))

            if self.rl_cfg.RL.mode=='test':
                path = self.rl_cfg.RL.settings + "_" + self.object_name + "_CV_" + '{0:.2f}'.format(
                    self.recon.coverage)
                save_path = os.path.join('outputs', path)
                np.save(save_path, self.metric_history)

                path = self.rl_cfg.RL.settings + "_" + self.object_name + "_" + '{0:.2f}'.format(
                    self.recon.coverage)
                save_path = os.path.join('outputs', path)
                np.save(save_path, self.recon.pcd)

        self.metric_history.append(np.array([self.horizon_counter, self.recon.coverage]))

        if (self.recon.coverage >= 90.0 and self.print_once):
            self.reward += 100 if self.rl_cfg.reward.type == "AMN" else 50000 / (self.horizon_counter)
            template = '########## Steps {:d}\t, Coverage: {:.2f} %\t, Reward: {:.2f}\t, Action: {:d}\n'
            print(template.format(self.horizon_counter, self.recon.coverage, self.reward, self.action))

            if self.rl_cfg.RL.mode=='test':
                path = self.rl_cfg.RL.settings + "_" + self.object_name + "_CV_" + '{0:.2f}'.format(
                    self.recon.coverage)
                save_path = os.path.join('outputs', path)
                np.save(save_path, self.metric_history)
            self.print_once = False

        if self.rl_cfg.RL.mode=='test' and terminate:
            if self.render:
                self.recon.visualize_result()
        template = '########## Evaluation Over: Steps {:d}\t, Coverage: {:.2f} %\t, Object:' + self.object_name + '\n'
        print(template.format(self.horizon_counter, self.recon.coverage))
        return terminate

    def _action_in_local_frame(self, relative_to_id, action):
        w_trans, w_rot = p.getBasePositionAndOrientation(relative_to_id)  # world coordinates
        delta = self.action_to_direction[action]
        curr_l2w = R.from_quat(w_rot)  # from object local frame to world frame
        new_trans = curr_l2w.apply(delta[:3]) + np.array(w_trans)  # back to world coordinates
        new_rot = (curr_l2w * R.from_rotvec(delta[3:])).as_quat()
        return new_trans, new_rot

    def _touch_indicator(self):
        if self.rl_cfg.state.input_type == 'point' or self.rl_cfg.state.input_type == 'grid':
            if len(self.pointcloud) > 10:
                return True

            else:
                return False

        else:
            if len(self.depth) == 0:
                return False

            return self._compute_area() > 0.05

    def _compute_area(self):
        if self.rl_cfg.reward.area_regularizer == "gaussian":
            depth = np.clip(self.depth / self.digits.zrange, 0, 1) * 255
            centered_depth = np.multiply(depth, self.gaussian_kernel)
            gauss = np.copy(centered_depth)
            gauss[gauss - 1e-4 <= 0] = 0
            cv2.imshow("centered", gauss / np.max(gauss) * 255)
            cv2.waitKey(1)
        else:
            centered_depth = self.depth
        return np.count_nonzero(centered_depth[centered_depth - 1e-4 > 0]) / centered_depth.size

    def update_camera_pose(self):
        # Set the camera position and orientation
        camera_distance = self.rl_cfg.pybullet_camera.cameraDistance
        camera_yaw = self.rl_cfg.pybullet_camera.cameraYaw
        camera_pitch = self.rl_cfg.pybullet_camera.cameraPitch
        camera_roll = 0.0
        camera_target = self.rl_cfg.pybullet_camera.cameraTargetPosition

        # Loop to update camera position and orientation
        while True:
            # Get the robot position and orientation
            robot_pos, robot_ori = p.getBasePositionAndOrientation(self.digit_body.id)

            # Compute the camera position and orientation based on the robot's position and orientation
            # camera_pos, camera_ori = p.computeViewMatrixFromYawPitchRoll(
            #     camera_target, camera_distance, camera_yaw + robot_ori[2],
            #                                     camera_pitch + robot_ori[1], camera_roll)

            # Set the camera position and orientation
            camera_target = robot_pos
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw + robot_ori[1] + 180,
                                         camera_pitch + robot_ori[2], camera_target)

            # Sleep for a short time to avoid excessive CPU usage
            time.sleep(0.01)

    def rotate_camera(self):
        # Set the camera position and orientation
        camera_distance = self.rl_cfg.pybullet_camera.cameraDistance
        camera_yaw = self.rl_cfg.pybullet_camera.cameraYaw
        camera_pitch = self.rl_cfg.pybullet_camera.cameraPitch
        camera_roll = 0.0
        camera_target = self.rl_cfg.pybullet_camera.cameraTargetPosition

        # Loop to update camera position and orientation
        # Get the robot position and orientation
        while True:
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw + self.camera_offset,
                                         camera_pitch, camera_target)
            self.camera_offset += 0.05
            if self.camera_offset == 360:
                self.camera_offset = 0
            # Sleep for a short time to avoid excessive CPU usage
            time.sleep(0.01)
