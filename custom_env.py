import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional

class GridWorldEnv(gym.Env):
    def __init__(self, x_ref, y_ref, z_ref):
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.z_ref = z_ref
        
        self.q_homing = np.array([0.002915, -0.025384, -0.271586, -0.002868, -0.025559, -0.272974, 0.007593, -0.048279, -0.283963, -0.007583, -0.048371, -0.285347])
        
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
        
        self.body_id = self.model.body(name="body").id
        
        action_space_low = [-0.788313, -0.873461, -2.521314, -0.782530, -0.873286, -2.519926, -0.777805, -0.850566, -2.508937, -0.777815, -0.850474, -2.507553]
        action_space_high = [0.782483, 2.320494, 0.017184, 0.788266, 2.269189, 0.017326, 0.777805, 2.343389, 0.036896, 0.777815, 2.343481, 0.037065]
        observation_space_low = [-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -0.785398, -0.898845, -2.7929, -0.785398, -0.898845, -2.7929, -0.785398, -0.898845, -2.7929, -0.785398, -0.898845, -2.7929, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -0.788313, -0.873461, -2.521314, -0.782530, -0.873286, -2.519926, -0.777805, -0.850566, -2.508937, -0.777815, -0.850474, -2.507553, -np.inf, -np.inf, -np.inf]
        observation_space_high = [np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, 0.785398, 2.29511, -0.254402, 0.785398, 2.24363, -0.255648, 0.785398, 2.29511, -0.247067, 0.785398, 2.29511, -0.248282, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.782483, 2.320494, 0.017184, 0.788266, 2.269189, 0.017326, 0.777805, 2.343389, 0.036896, 0.777815, 2.343481, 0.037065, np.inf, np.inf, np.inf]
        
        self.action_space = gym.spaces.Box(low=np.array(action_space_low), high=np.array(action_space_high), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array(observation_space_low), high=np.array(observation_space_high), dtype=np.float32)

        # let the simulation run for 1000 steps for it to become stable
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
            
        self.viewer = None

    def get_obs(self):
        # x, y, z
        robot_pos = self.data.xpos[self.body_id]
        x = robot_pos[0]
        y = robot_pos[1]
        z = robot_pos[2]
        
        # roll, pitch, yaw
        quat = self.data.xquat[self.body_id]
        rpy = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=False)
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]
        
        # q1 ~ q12
        joint_pos = self.data.qpos[7:]
        
        # q1' ~ q12'
        joint_vel = self.data.qvel[7:]
        
        # previous action a1 ~ a12
        
        
        # x_ref, y_ref, x_ref
        x_ref = self.x_ref
        y_ref = self.y_ref
        z_ref = self.z_ref
    
    def get_reward(self):
        pass
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self.get_obs()
        info = None

        return observation, info

    def step(self, action):
        # excute action
        self.data.ctrl[:len(action)] = self.q_homing + action
        mujoco.mj_step(self.model, self.data)
        
        x = self.data.xpos[self.body_id][0]
        y = self.data.xpos[self.body_id][0]
        if (self.x_ref - 1 <= x <= self.x_ref + 1) and (self.y_ref - 1 <= y <= self.y_ref + 1):
            terminated = True
        else:
            terminated = False
            
        truncated = False
        
        reward = self.get_reward()
        observation = self.get_obs()
        info = None

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