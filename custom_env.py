import copy
import math

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

        self.maxDistance = 0

        self.x_ref = x_ref
        self.y_ref = y_ref
        self.z_ref = z_ref

        self.roll_th = 5 ## original 10
        self.pitch_th = 5 ## original 10
        self.yaw_th = 5
        self.max_steps = 10000
        self.curr_step = 0


        ## This is the natural Gate for the Robot
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
        self.body_pos = self.data.xpos[self.body_id]

        self.prevBodyPos = [0,0,0]

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
            -np.inf, -np.inf, -np.inf,
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
            np.inf, np.inf, np.inf,
            np.inf, np.inf, np.inf
        ])  ##Added extra line (three elements) for previous and current body pos

        self.action_space = gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=observation_space_low, high=observation_space_high,
                                                dtype=np.float64)

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


        xPrev = self.prevBodyPos[0]
        yPrev = self.prevBodyPos[1]
        zPrev = self.prevBodyPos[2]
        xyzPrev = np.array([xPrev, yPrev, zPrev])

        return np.concatenate((xyz, rpy, joint_pos, joint_vel, prev_action, xyz_ref,xyzPrev), axis=0)

    def get_reward(self, prev_obs, action, NewRecordBool):

        ## NOTE that in this part prev_obs is refering to the most recent observation space.

        RightDirectionReward = 0

############################################################################
        RightDirectionBool = False
        if prev_obs[42] > 0 and (prev_obs[0] - prev_obs[45]) > 0 and (not(
                -.0002 < (prev_obs[0] - prev_obs[45]) < .0002)): ## If X_ref is positive and the difference from the current body position along x axis and the prevoius body position along x-axis is going in the correct direction, and the distance it moves is greater than .0002 then a reward is given.

            RightDirectionReward += (self.maxDistance + 10) - (self.maxDistance - abs(prev_obs[0])) ## Accidently had this as self.MaxRewardDistance
            RightDirectionBool = True
        elif prev_obs[42] < 0 and (prev_obs[0] - prev_obs[45]) < 0:
            RightDirectionReward += (self.maxDistance + 10) - (self.maxDistance - abs(prev_obs[0]))
            RightDirectionBool = True

        if prev_obs[43] > 0 and (prev_obs[1] - prev_obs[46]) > 0 and (not(
                -.0002 < (prev_obs[1] - prev_obs[46]) < .0002)):

            RightDirectionReward += (self.maxDistance + 10) - (self.maxDistance - abs(prev_obs[1]))
            RightDirectionBool = True
        elif prev_obs[43] < 0 and (prev_obs[1] - prev_obs[46]) < 0 and (not(
                -.0002 < (prev_obs[1] - prev_obs[46]) < .0002)):

            RightDirectionReward += (self.maxDistance + 10) - (self.maxDistance - abs(prev_obs[1]))
            RightDirectionBool = True

        OutOfLinePenalty = 0

        if prev_obs[42] == 0:
            OutOfLinePenalty += -(abs(prev_obs[0] - prev_obs[45]) * RightDirectionReward)

        if prev_obs[43] == 0:
            OutOfLinePenalty += -(abs(prev_obs[1] - prev_obs[46]) * RightDirectionReward)
##############################################################################################
        newRecord = 0
        if NewRecordBool and RightDirectionBool:
            newRecord = 10*RightDirectionReward ##Needs to be changed for omni directional

#############################################################################################

        rollPenalty = 0
        pitchPenalty = 0
        yawPenalty = 0

        if abs(prev_obs[4]) > (.5 * self.roll_th):
            rollPenalty = -(RightDirectionReward * .7)

        if abs(prev_obs[5]) > (.5 * self.pitch_th):
            pitchPenalty = -(RightDirectionReward * .7)

        if abs(prev_obs[6]) > (.5 * self.yaw_th):
            yawPenalty = -(RightDirectionReward * .7)

        return RightDirectionReward + newRecord + yawPenalty + rollPenalty + pitchPenalty


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.prev_action = np.zeros(12, dtype=np.float32)
        self.body_id = self.model.body(name="body").id
        self.body_pos = self.data.xpos[self.body_id]

        self.prevBodyPos = [0,0,0]


        self.curr_step = 0

        observation = self.get_obs()
        info = {}

        return observation, info

    def step(self, action):
        self.prevBodyPos = np.array(self.body_pos)

        # excute action
        self.data.ctrl[:len(action)] = self.q_homing + action ## Wouldn't it be previous pos + action
        mujoco.mj_step(self.model, self.data)
        self.prev_action = action

        self.body_id = self.model.body(name="body").id
        self.body_pos = self.data.xpos[self.body_id]

        observation = self.get_obs() ## There could be an issue, maybe use prev_obs


        x = observation[0]
        y = observation[1]
        roll = observation[4]
        pitch = observation[5]

        terminated = False
        if (abs(roll) > self.roll_th or abs(pitch) > self.pitch_th or self.curr_step > self.max_steps):
            terminated = True

        truncated = False

        NewRecordBool = False

        if abs(observation[0]) > self.maxDistance:
            self.maxDistance = abs(observation[0])
            NewRecordBool = True

        reward = self.get_reward(observation, action, NewRecordBool)



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