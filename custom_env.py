import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional

class RobotDogEnv(gym.Env):
    def __init__(self, x_ref, y_ref, z_ref=0.0):
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.z_ref = z_ref
        
        self.roll_th = 10
        self.pitch_th = 10
        self.max_steps = 10000
        self.curr_step = 0
        
        self.q_homing = np.array([0, 1.04, -1.8, 
                                  0, 1.04, -1.8, 
                                  0, 1.04, -1.8, 
                                  0, 1.04, -1.8])
        
        xml_path = "boston_dynamics_spot/scene.xml"
        dirname = os.path.dirname(__file__)
        abs_path = os.path.join(dirname, xml_path)

        model_name = 'model.txt'
        model_path = os.path.join(dirname, model_name)

        # Load model
        self.model = mujoco.MjModel.from_xml_path(abs_path)
        mujoco.mj_printModel(self.model, model_path)

        # Create data
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        self.body_id = self.model.body(name="body").id
        
        action_space_low = np.array([
            -0.785398, -1.938845, -0.9929,
            -0.785398, -1.938845, -0.9929,
            -0.785398, -1.938845, -0.9929,
            -0.785398, -1.938845, -0.9929
        ])
        
        action_space_high = np.array([
            0.785398, 1.25511, 1.545598,
            0.785398, 1.20363, 1.544352,
            0.785398, 1.25511, 1.552933,
            0.785398, 1.25511, 1.551718
        ])
        
        observation_space_low = np.array([
            -np.inf, -np.inf, -np.inf,
            -np.pi, -np.pi, -np.pi,
            -0.785398, -0.898845, -2.7929,
            -0.785398, -0.898845, -2.7929,
            -0.785398, -0.898845, -2.7929,
            -0.785398, -0.898845, -2.7929,
            -np.inf, -np.inf, -np.inf,
            -np.inf, -np.inf, -np.inf,
            -np.inf, -np.inf, -np.inf,
            -np.inf, -np.inf, -np.inf,
            -0.785398, -1.938845, -0.9929,
            -0.785398, -1.938845, -0.9929,
            -0.785398, -1.938845, -0.9929,
            -0.785398, -1.938845, -0.9929,
            -np.inf, -np.inf, -np.inf
        ])

        observation_space_high = np.array([
            np.inf, np.inf, np.inf,
            np.pi, np.pi, np.pi,
            0.785398, 2.29511, -0.254402,
            0.785398, 2.24363, -0.255648,
            0.785398, 2.29511, -0.247067,
            0.785398, 2.29511, -0.248282,
            np.inf, np.inf, np.inf,
            np.inf, np.inf, np.inf,
            np.inf, np.inf, np.inf,
            np.inf, np.inf, np.inf,
            0.785398, 1.25511, 1.545598,
            0.785398, 1.20363, 1.544352,
            0.785398, 1.25511, 1.552933,
            0.785398, 1.25511, 1.551718,
            np.inf, np.inf, np.inf
        ])

        self.action_space = gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=observation_space_low, high=observation_space_high, dtype=np.float64)
            
        self.viewer = None
        
        self.prev_action = np.zeros(12, dtype=np.float32)

    def get_obs(self):
        # x, y, z
        robot_pos = self.data.xpos[self.body_id]
        x = robot_pos[0]
        y = robot_pos[1]
        z = robot_pos[2]
        xyz = np.array([x, y, z])
        
        # roll, pitch, yaw
        quat = self.data.xquat[self.body_id]
        
        # Check if quaternion is valid
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            # print("Warning: Zero quaternion detected. Replacing with identity quaternion.")
            quat = np.array([1, 0, 0, 0])  # MuJoCo uses wxyz order
            
        rpy = R.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=True)
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]
        rpy = np.array([roll, pitch, yaw])
        
        # q1 ~ q12
        joint_pos = self.data.qpos[7:]
        
        # q1' ~ q12'
        joint_vel = self.data.qvel[6:]
        
        # previous action a1 ~ a12
        prev_action = self.prev_action
        
        # x_ref, y_ref, x_ref
        x_ref = self.x_ref
        y_ref = self.y_ref
        z_ref = self.z_ref
        xyz_ref = np.array([x_ref, y_ref, z_ref])
        
        return np.concatenate((xyz, rpy, joint_pos, joint_vel, prev_action, xyz_ref), axis=0)
    
    def get_reward(self, prev_obs, action):
        # Position Tracking Reward
        R_xyz = np.exp(-np.linalg.norm(prev_obs[0:2] - np.array([self.x_ref, self.y_ref]), ord=2))
        
        # Pose Similarity Penalty
        R_pose = -np.linalg.norm(prev_obs[6:18] - self.q_homing, ord=2)
        
        # Action Rate Penalty
        R_action = -np.linalg.norm(prev_obs[30:42] - action, ord=2)
        
        # Stabilization Penalty
        R_stable = -(prev_obs[3]**2 + prev_obs[4]**2)
        
        # Facing Target Reward
        yaw = np.deg2rad(prev_obs[5])
        facing_vector = np.array([np.cos(yaw), np.sin(yaw)])
        goal_vector = np.array([self.x_ref - prev_obs[0], self.y_ref - prev_obs[1]])
        goal_vector = goal_vector / (np.linalg.norm(goal_vector, ord=2) + 1e-8)
        R_facing = np.dot(facing_vector, goal_vector)
        
        # print("R_xyz", R_xyz)
        # print("R_pose", R_pose)
        # print("R_action", R_action)
        # print("R_stable", R_stable)
        # print("R_facing", R_facing)
        
        return R_xyz + R_pose + R_action + R_stable + R_facing
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            
        self.prev_action = np.zeros(12, dtype=np.float32)
        
        observation = self.get_obs()
        info = {}

        return observation, info

    def step(self, action):
        prev_obs = self.get_obs()
        
        # excute action
        self.data.ctrl[:len(action)] = self.q_homing + action
        mujoco.mj_step(self.model, self.data)
        self.prev_action = action
        
        x = prev_obs[0]
        y = prev_obs[1]
        roll = prev_obs[4]
        pitch = prev_obs[5]
        if (self.x_ref - 1 <= x <= self.x_ref + 1) and (self.y_ref - 1 <= y <= self.y_ref + 1):
            terminated = True
        elif (abs(roll) > self.roll_th or abs(pitch) > self.pitch_th or self.curr_step > self.max_steps):
            terminated = True
        else:
            terminated = False
            
        truncated = False
        reward = self.get_reward(prev_obs, action)
        observation = self.get_obs()
        info = {}
        self.curr_step += 1
        
        self.render()

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Initialize the viewer if it hasn't been initialized
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Sync and render the viewer
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()