import numpy as np

'''
toolbox for new tasks
'''

def sample_boxes_pose(min_distance=0.05):
    def generate_random_position(ranges):
        return np.random.uniform(ranges[:, 0], ranges[:, 1])

    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])

    cube1_position = generate_random_position(ranges)
    cube1_quat = np.array([1, 0, 0, 0])
    cube1_pose = np.concatenate([cube1_position, cube1_quat])

    while True:
        cube2_position = generate_random_position(ranges)
        distance = np.linalg.norm(cube1_position[:2] - cube2_position[:2])
        if distance >= min_distance:
            break

    cube2_quat = np.array([1, 0, 0, 0])
    cube2_pose = np.concatenate([cube2_position, cube2_quat])

    return cube1_pose, cube2_pose


def sample_tube_box_pose(min_distance=0.05):
    def generate_random_position(ranges):
        return np.random.uniform(ranges[:, 0], ranges[:, 1])

    x_range = [0.0, 0.2]
    x_range_left = [-0.2, 0.0]
    x_range_right = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    ranges_left = np.vstack([x_range_left, y_range, z_range])
    ranges_right = np.vstack([x_range_right, y_range, z_range])

    cube1_position = generate_random_position(ranges_left)
    cube1_quat = np.array([1, 0, 0, 0])
    cube1_pose = np.concatenate([cube1_position, cube1_quat])

    while True:
        cube2_position = generate_random_position(ranges_right)
        distance = np.linalg.norm(cube1_position[:2] - cube2_position[:2])
        if distance >= min_distance:
            break

    cube2_quat = np.array([1, 0, 0, 0])
    cube2_pose = np.concatenate([cube2_position, cube2_quat])

    return cube1_pose, cube2_pose

def sample_box_tube_cover_pose():
    # 注意 tube 應該是靠近左側， cover 靠近右側
    tube_pose, cover_pose = sample_tube_box_pose()
    box_pose = np.array([0.0, .8, 0.01, 1, 0, 0, 0])
    # 這裡很奇怪，應該是 tube 然後是 cover，但是這樣傳到 imitate_episodes.py 時這兩個位置會對調
    return box_pose, tube_pose, cover_pose



if __name__ == '__main__':
    cube1_pose, cube2_pose = sample_boxes_pose()
    print(f"Red Box Pose: {cube1_pose}")
    print(f"Green Box Pose: {cube2_pose}")
