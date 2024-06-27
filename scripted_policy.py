import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        # update box info
        box_info = np.array(ts.observation['env_state'])
        #print(f"Updated box_info: {box_info}")

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
        policy = PickAndTransferPolicy(inject_noise)
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    elif 'sim_stack_cube' in task_name:
        env = make_ee_sim_env('sim_stack_cube')
        policy = PickAndStackAndTransferPolicy(inject_noise)
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


class PickAndStackAndTransferPolicy(BasePolicy):
     
     def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        box_info = np.array(ts_first.observation['env_state'])
        print(f"init_mocap_pose_right: {init_mocap_pose_right}")
        print(f"init_mocap_pose_left: {init_mocap_pose_left}")
        #print(f"{box_info}")
        #box_xyz = box_info[:3]
        #box_quat = box_info[3:]
        #print(f"[Policy] Generate trajectory for {box_xyz=}")

        # split box_info
        red_box_xyz = box_info[:3]
        red_box_quat = box_info[3:7]
        green_box_xyz = box_info[7:10]
        green_box_quat = box_info[10:14]

        # print the positions of red_box and green_box
        print(f"Red Box Position: {red_box_xyz}, Quaternion: {red_box_quat}")
        print(f"Green Box Position: {green_box_xyz}, Quaternion: {green_box_quat}")
        
        #AttributeError: 'PickAndStackAndTransferPolicy' object has no attribute 'curr_left_waypoint'
    
        # generate trajectory
        right_gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        right_gripper_pick_quat = right_gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        right_meet_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # initial position
            
            {"t": 60, "xyz": red_box_xyz + np.array([0, 0, -0.015]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # go down to red box
            {"t": 100, "xyz": red_box_xyz + np.array([0, 0, -0.015]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # close gripper
            {"t": 130, "xyz": green_box_xyz + np.array([0, 0, 0.1]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # move above green box
            {"t": 150, "xyz": green_box_xyz + np.array([0, 0, 0.02]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # place red box on green box
            {"t": 180, "xyz": green_box_xyz + np.array([0, 0, 0.02]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # open gripper
            {"t": 200, "xyz": green_box_xyz + np.array([0, 0.00, 0.02]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # move away from green box
            {"t": 250, "xyz": green_box_xyz + np.array([0.3, 0, 0.3]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # move away from green box
            {"t": 300, "xyz": green_box_xyz + np.array([0.3, 0, 0.3]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # move away from green box
            {"t": 400, "xyz": green_box_xyz + np.array([0.3, 0, 0.3]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # move away from green box
        ]

        # left arm trajectory: keep sleep
        left_gripper_pick_quat = Quaternion(init_mocap_pose_left[3:])
        left_gripper_pick_quat = left_gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        left_meet_quat = Quaternion(axis=[0.0, 1.0, 0.0], degrees=10)
        
        #{"t": 400, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # insertion
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 260, "xyz": green_box_xyz + np.array([0, 0, 0.015]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # approach green box
            {"t": 290, "xyz": green_box_xyz + np.array([0, 0, 0.015]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # approach red box
            {"t": 320, "xyz": green_box_xyz + np.array([0, 0, 0.015]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # 
            {"t": 350, "xyz": green_box_xyz + np.array([0, 0, 0.015]), "quat": left_gripper_pick_quat.elements, "gripper": 0},  # close gripper
            {"t": 400, "xyz": green_box_xyz + np.array([0, 0, 0.2]), "quat": left_gripper_pick_quat.elements, "gripper": 0},  # close gripper
            
        ]


class OpenBoxCoverPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        box_info = np.array(ts_first.observation['env_state'])
        print(f"init_mocap_pose_right: {init_mocap_pose_right}")
        print(f"init_mocap_pose_left: {init_mocap_pose_left}")
        # split box_info
        box_xyz = box_info[:3]
        box_quat = box_info[3:7]
        cover_xyz = box_info[7:10]
        cover_quat = box_info[10:14]


        # print the positions of box and cover
        print(f"Box Position: {box_xyz}, Quaternion: {box_quat}")
        print(f"Cover Position: {cover_xyz}, Quaternion: {cover_quat}")


        # generate trajectory
        right_gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        right_gripper_pick_quat = right_gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        right_meet_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        #
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # initial position
            {"t": 50, "xyz": cover_xyz + np.array([0.05, 0, 0.03]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # go down to cover
            {"t": 100, "xyz": cover_xyz + np.array([0, 0, -0.01]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # gripper close
            {"t": 150, "xyz": cover_xyz + np.array([0, 0, -0.01]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # gripper close
            {"t": 200, "xyz": cover_xyz + np.array([0.01, 0, 0.15]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # move cover up
            {"t": 250, "xyz": box_xyz + np.array([0, 0, 0.15]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # go down to box
            {"t": 300, "xyz": box_xyz + np.array([0, 0, 0.1]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # go down to box
            {"t": 350, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # go down to box
            {"t": 400, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # go down to box
        ]

        # left arm trajectory: keep sleep
        left_gripper_pick_quat = Quaternion(init_mocap_pose_left[3:])
        left_gripper_pick_quat = left_gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        left_meet_quat = Quaternion(axis=[0.0, 1.0, 0.0], degrees=10)
        
        #{"t": 400, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 400, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # do nothing
        ]

        print(f"Right Trajectory: {self.right_trajectory}")
        print(f"Left Trajectory: {self.left_trajectory}")


class TubeBoxCoverPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        box_info = np.array(ts_first.observation['env_state'])
        #print(f"init_mocap_pose_right: {init_mocap_pose_right}")
        #print(f"init_mocap_pose_left: {init_mocap_pose_left}")
        # split box_info
        box_xyz = box_info[:3]
        box_quat = box_info[3:7]
        cover_xyz = box_info[7:10]
        cover_quat = box_info[10:14]
        tube_xyz = box_info[14:17]
        tube_quat = box_info[17:21]

        # print the positions of box and cover
        #print(f"Box Position: {box_xyz}, Quaternion: {box_quat}")
        #print(f"Cover Position: {cover_xyz}, Quaternion: {cover_quat}")
        #print(f"Tube Position:{tube_xyz}, Quaternion:{tube_quat}")

        # generate trajectory
        right_gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        right_gripper_pick_quat = right_gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        right_meet_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        #
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # initial position
            {"t": 50, "xyz": cover_xyz + np.array([0.05, 0, 0.03]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # go down to cover
            {"t": 100, "xyz": cover_xyz + np.array([0, 0, -0.01]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # go to the cover
            {"t": 120, "xyz": cover_xyz + np.array([0, 0, -0.01]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # gripper close
            {"t": 150, "xyz": cover_xyz + np.array([0.01, 0, 0.15]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # move cover up
            {"t": 180, "xyz": box_xyz + np.array([0.07, 0, 0.15]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # go to the box
            {"t": 200, "xyz": box_xyz + np.array([0.07, 0, 0.15]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # go to the box
            {"t": 240, "xyz": box_xyz + np.array([0.07, 0, 0.15]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # wait
            {"t": 330, "xyz": box_xyz + np.array([-0.004, 0, 0.13]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # wait
            {"t": 340, "xyz": box_xyz + np.array([-0.004, 0, 0.11]), "quat": right_gripper_pick_quat.elements, "gripper": 0},  # open the gripper
            {"t": 345, "xyz": box_xyz + np.array([-0.004, 0, 0.11]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # open the gripper
            {"t": 400, "xyz": box_xyz + np.array([0.2, 0, 0.11]), "quat": right_gripper_pick_quat.elements, "gripper": 1},  # move away
        ]

        # left arm trajectory: keep sleep
        left_gripper_pick_quat = Quaternion(init_mocap_pose_left[3:])
        left_gripper_pick_quat = left_gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        left_meet_quat = Quaternion(axis=[0.0, 1.0, 0.0], degrees=10)
        
        #{"t": 400, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # initial position
            {"t": 50, "xyz": tube_xyz + np.array([0.00, 0, 0.05]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # go to tube
            {"t": 80, "xyz": tube_xyz + np.array([0.00, 0, 0.02]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # go to tube
            {"t": 100, "xyz": tube_xyz + np.array([0.00, 0, -0.01]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # go to tube
            {"t": 120, "xyz": tube_xyz + np.array([0.00, 0, -0.01]), "quat": left_gripper_pick_quat.elements, "gripper": 0},  # close the gripper
            {"t": 150, "xyz": tube_xyz + np.array([0.00, 0, 0.1]), "quat": left_gripper_pick_quat.elements, "gripper": 0},  # lift the tube
            {"t": 180, "xyz": box_xyz + np.array([0.00, 0, 0.15]), "quat": left_gripper_pick_quat.elements, "gripper": 0},  # lift the tube to box
            {"t": 210, "xyz": box_xyz + np.array([0.00, 0, 0.11]), "quat": left_gripper_pick_quat.elements, "gripper": 0},  # lift the tube to box
            {"t": 220, "xyz": box_xyz + np.array([0.00, 0, 0.11]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # open the gripper
            {"t": 240, "xyz": box_xyz + np.array([-0.05, 0, 0.11]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # move away 
            {"t": 270, "xyz": box_xyz + np.array([-0.08, 0, 0.11]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # move away
            {"t": 300, "xyz": box_xyz + np.array([-0.1, 0, 0.11]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # move away
            {"t": 400, "xyz": box_xyz + np.array([-0.15, 0, 0.11]), "quat": left_gripper_pick_quat.elements, "gripper": 1},  # move away
        ]

        #print(f"Right Trajectory: {self.right_trajectory}")
        #print(f"Left Trajectory: {self.left_trajectory}")


if __name__ == '__main__':
    #test_task_name = 'sim_transfer_cube_scripted'
    test_task_name = 'sim_stack_cube_scripted'
    test_policy(test_task_name)

