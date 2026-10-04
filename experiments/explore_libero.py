import os

os.environ["MUJOCO_GL"] = "osmesa"

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
import numpy as np
from PIL import Image

benchmark_dict = benchmark.get_benchmark_dict()

task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)

print("Task:", task.name)
print("Language:", task.language)

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=128,
    camera_widths=128,
)

obs = env.reset()

image = obs["agentview_image"]
Image.fromarray(image[::-1]).save("agentview.png")

print("joint pos:", obs["robot0_joint_pos"])
print("eef pos:", obs["robot0_eef_pos"])
print("eef quat:", obs["robot0_eef_quat"])
print("gripper:", obs["robot0_gripper_qpos"])

print("agentview image shape:", obs["agentview_image"].shape)
print("eye-in-hand image shape:", obs["robot0_eye_in_hand_image"].shape)

print("\nObservation keys:")
for key in obs.keys():
    print(key)

print("Initial end-effector position:")
print(obs["robot0_eef_pos"])

action = np.array([
    0.2,   # +x
    0.0,   # y
    0.0,   # z
    0.0,   # rotation x
    0.0,   # rotation y
    0.0,   # rotation z
    0.0    # gripper
])

for i in range(20):
    obs, reward, done, info = env.step(action)

print("Final end-effector position:")
print(obs["robot0_eef_pos"])

next_obs, reward, done, info = env.step(action)

print("\nReward:", reward)
print("Done:", done)

env.close()