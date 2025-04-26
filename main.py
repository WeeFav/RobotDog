import mujoco
import mujoco.viewer
import time
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def trot_gait(t, step_size=0.5, step_freq=2.0):
    """
    Generate a trotting gait for 12 leg actuators (3 per leg)
    Assume order: fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, hl_hx, ...
    """
    phase = step_freq * t

    # Front left and hind right in phase
    fl = [np.sin(phase), np.cos(phase), -abs(np.sin(phase))]
    hr = [np.sin(phase), np.cos(phase), -abs(np.sin(phase))]

    # Front right and hind left out of phase
    fr = [np.sin(phase + np.pi), np.cos(phase + np.pi), -abs(np.sin(phase + np.pi))]
    hl = [np.sin(phase + np.pi), np.cos(phase + np.pi), -abs(np.sin(phase + np.pi))]

    # Combine all 12 actuator values
    ctrl = fl + fr + hl + hr
    return step_size * np.array(ctrl)

def main():
    xml_path = "boston_dynamics_spot/scene.xml"
    dirname = os.path.dirname(__file__)
    abs_path = os.path.join(dirname, xml_path)

    model_name = 'model.txt'
    model_path = os.path.join(dirname, model_name)

    # Load model
    model = mujoco.MjModel.from_xml_path(abs_path)
    mujoco.mj_printModel(model, model_path)

    # Create data
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    body_id = model.body(name="body").id
    quat = data.xquat[body_id]
    
    # Launch passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Close the window to stop.")
        rpy = R.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=True)

        start = time.time()
        
        pos_list = []
        linear_list = []
        angular_list = []
        rpy_list = []
                        
        try:
            while viewer.is_running():
                t = time.time() - start
                
                # ctrl = trot_gait(t)
                # data.ctrl[:len(ctrl)] = ctrl

                body_id = model.body(name="body").id
                quat = data.xquat[body_id]
                rpy = R.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=True)
                mujoco.mj_step(model, data)
                viewer.sync()  # Update the visualization
                
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
                
                robot_pos = data.xpos[body_id]
                linear_vel = vel[3:]
                angular_vel = vel[:3]
                
                # w x y z
                print(quat)
                print(rpy)
                
                # print(data.qpos[7:])
                # print(data.qvel[7:])
                
                # pos_list.append(robot_pos.copy())
                # linear_list.append(linear_vel.copy())
                # angular_list.append(angular_vel.copy())
                # rpy_list.append(rpy.copy())

                time.sleep(0.01)  # Sleep to slow down simulation (optional)
        
        except KeyboardInterrupt:
            pass
        
        
        # pos_list = np.array(pos_list)
        # plt.plot(pos_list[:, 0], label="x")
        # plt.plot(pos_list[:, 1], label="y")
        # plt.plot(pos_list[:, 2], label="z")
        # plt.legend()
        # # plt.show()
        # plt.clf()
        
        # linear_list = np.array(linear_list)
        # plt.plot(linear_list[:, 0], label="x")
        # plt.plot(linear_list[:, 1], label="y")
        # plt.plot(linear_list[:, 2], label="z")
        # plt.legend()
        # # plt.show()
        # plt.clf()
        
        # angular_list = np.array(angular_list)
        # plt.plot(angular_list[:, 0], label="x")
        # plt.plot(angular_list[:, 1], label="y")
        # plt.plot(angular_list[:, 2], label="z")
        # plt.legend()
        # # plt.show()
        # plt.clf()
        
        # rpy_list = np.array(rpy_list)
        # plt.plot(rpy_list[:, 0], label="x")
        # plt.plot(rpy_list[:, 1], label="y")
        # plt.plot(rpy_list[:, 2], label="z")
        # plt.legend()
        # plt.show()
        # plt.clf()
        

if __name__ == "__main__":
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
    main()
