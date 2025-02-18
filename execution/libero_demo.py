
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import glob
import draccus
import numpy as np
import tqdm
import wandb
from PIL import Image
# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")

from collections import Counter
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
import argparse


import re
from libero.libero.utils.bddl_generation_utils import *

from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
import json
import imageio
# absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
# 
import debugpy
# debugpy.listen(5678)
debugpy.listen(('0.0.0.0', 5681))
print('Waiting for debugger attach')
debugpy.wait_for_client()

from typing import Optional, Union
from dataclasses import dataclass


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, mp4_path = None):
    """Saves an MP4 replay of an episode."""
    # rollout_dir = f"/data/workspace/LIBERO/rollouts/{DATE}"
    # # os.makedirs(rollout_dir, exist_ok=True)
    # processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    # mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    # mp4_path = f"{rollout_dir}/{result}"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


@dataclass
class GenerateConfig:
    # fmt: off
                 # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                   # Number of rollouts per task
    max_steps: int = 650
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on









@register_mu(scene_type="kitchen")
class KitchenDemoScene(InitialSceneTemplates):
    def __init__(self, json_data = None):
        self.json_data = json_data
        self.objs = [obj.replace(' ', '_') for obj in self.json_data['selected_obj_names']]
        self.number_obj = len(self.objs)
        self.init_obj_pos = self.json_data['init_obj_pos']
        self.quat_dict =  dict()
        self.goal_object = self.json_data['target_obj_name']
        self.xml_dict = [os.path.dirname(select) for select in self.json_data['selected_urdfs']]
            
        fixture_num_info = {
            "your_task": 1,
        }

        
        objects_dict = Counter(self.objs)
        object_num_info = {
            **objects_dict,
            # "libero_mug_yellow": 1,
        }

        
        super().__init__(
            workspace_name="your_task",  # define the scene base
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )
        
    def define_regions(self):
        objects_dict = dict.fromkeys(self.objs, 1)
        for id in range(self.number_obj):
            self.regions.update(
                self.get_region_dict(
                region_centroid_xy = self.init_obj_pos[id],
                region_name=self.objs[id].replace(' ', '_')+'_'+ str(objects_dict[self.objs[id]])+'_init_region',
                target_name=self.workspace_name,
                region_half_len=0.02,
                yaw_rotation = tuple(self.init_obj_pos[id][3:7]),
                goal_quat = self.quat_dict.get(self.objs[id], [0, 0, 0, 1]),
                xml = self.xml_dict[id],
                init_pos = self.init_obj_pos[id][:3],
                init_quat = self.init_obj_pos[id][3:7],
            )
            )
            objects_dict[self.objs[id]] +=1
            # print(self.quat_dict[self.objs[id]] if self.quat_dict[self.objs[id]] else [1,0,0,0])
            
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = []
        objects_dict = dict.fromkeys(self.objs, 1)
        for id in range(self.number_obj):
            states.append(
                ("On", self.objs[id]+'_'+ str(objects_dict[self.objs[id]]), "your_task_"+self.objs[id]+'_'+ str(objects_dict[self.objs[id]])+'_init_region'
            )
            )
            objects_dict[self.objs[id]] +=1

        return states
    


import re

def extract_object_from_instruction(orientation, instruction, selected_obj, goal_objs):
    # 加入对冠词的忽略（如 "the", "a", "an"）
    found_objects = []
    if orientation == "between":
    # 匹配 between 后的两个物体名称
        pattern = rf"between (?:the|a|an)?\s*([a-zA-Z_\s]+?)\s*and (?:the|a|an)?\s*([a-zA-Z\s]+?)(?:\s|$)"
        match = re.search(pattern, instruction)
        if match:
            obj1 = match.group(1).strip()
            obj2 = match.group(2).strip()
            # 在 selected_obj_names 中查找两个物体
            
            for obj in selected_obj:
                if obj in obj1:
                    found_objects.append(obj)
                if obj in obj2:
                    found_objects.append(obj)
            return found_objects if found_objects else None
    elif orientation in ["behind"]:
        pattern = rf"{orientation} (?:the|a|an)?\s*([a-zA-Z_\s]+?)(?:\s|$)"
    elif orientation in ["center"]:
        return [obj for obj in selected_obj if obj != goal_objs]
    else:
        pattern = rf"{orientation} of (?:the|a|an)?\s*([a-zA-Z_\s]+?)(?:\s|$)"
    match = re.search(pattern, instruction)
    if match:
        following_text = match.group(1).strip()
        for obj in selected_obj:
            if following_text in obj:
                
                found_objects.append(obj)
                return found_objects



def create_task_dict(bddl_path, task_json_path, video_path, final_positions):
    task_dict = {
        "bddl_path": bddl_path,
        "task_json_path": task_json_path,
        "video_path": video_path,
        "final_positions": final_positions,  # List of positions for 10 trials
    }
    return task_dict

def eval_libero(save_path, root, json_data, json_file, cfg: GenerateConfig, waypoint_path):
    
    scene_name = "kitchen_demo_scene"
    bddl_name = os.path.basename(root)
    goal_objs = json_data['target_obj_name']  

    orientation = json_data['position_tag']
    instruction = json_data['instruction']
    selected_obj = json_data['selected_obj_names']
    matching_word = extract_object_from_instruction(orientation,instruction, selected_obj, goal_objs)

    # if the orientation is the center, we have to consider all objects
    print('json_file', json_file)
    register_task_info(
            bddl_name, # register the task 
            scene_name=scene_name,
            objects_of_interest=[],
            goal_states=[
                (orientation.title(), matching_word[i]+'_1', goal_objs+'_1') for i in range(len(matching_word))
            ],
            json_data = json_data,
        )
    bddl_file_names, failures = generate_bddl_from_task_info(folder=save_path, json_data=json_data) # generate the bddl_file
    mp4_path = bddl_file_names[0].replace('bddl','mp4')

    local_log_filepath = os.path.join(save_path, "result.txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Initialize LIBERO environment and task description
    env_args = {
    "bddl_file_name": bddl_file_names[0],
    # "bddl_file_name": bddl_file_namesssss,
    "camera_heights": 256,
    "camera_widths": 256,
    "camera_depths": True,
    "camera_names": ["frontview","agentview"],

    "controller": "OSC_POSE",
    "controller_config_file": "",
    "ignore_done": True
}
    task_description = instruction
    
    env = OffScreenRenderEnv(**env_args)
    
    # from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
    # intrinstic_matrix = get_camera_intrinsic_matrix(env.sim,'agentview',256, 256)
    # extrinsic_matrix = get_camera_extrinsic_matrix(env.sim, 'agentview')
    # import pdb; pdb.set_trace()


    obj_pos_dict = dict()
    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        assert cfg.num_trials_per_task==1
        
        print(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")

        # Reset environment
        obs = env.reset()
        # Setup
        t = 0
        replay_images = []

        # waypoint_path = "/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_your_task_ours_test4_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_your_task_ours_test4_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_your_task_ours_test4_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_rot_test6.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_rot_test4.npy" #"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_ours_pre0.3_n20_filter5_osc_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_ours_osc_formatted.npy" #"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_ours_osc_formatted.npy" #"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_ours1_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints.npy"
        actions = np.load(waypoint_path, allow_pickle=True)
        print("task_description:", task_description)
        print(f"Starting episode {task_episodes+1}...")
        log_file.write(f"Starting episode {task_episodes+1}...\n")
        # for t in range(10):
        #     obs, reward, done, info = env.step([-0.2, 0, 0, 0, 0, 0, 0])

        while t < cfg.max_steps + cfg.num_steps_wait:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step([0, 0, 2, 0, 0, 0, 0])
             
                t+=1



            img = obs["frontview_image"][::-1]
            depth = obs['frontview_depth'][::-1]
            # img1 = obs["agentview_image"][::-1]
            # depth1 = obs['agent1view_depth'][::-1]

            i = (t-cfg.num_steps_wait) // 4
            action = actions[i]
            if action[6] != 0:
                action[6] = 8
     
       

            # Save preprocessed image for replay video
            replay_images.append(img)
       
            obs, reward, done, info = env.step(np.array(action))

            from PIL import Image
   

            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1
            
        # log_file.write(f"Inputs: stored\n")         
        log_file.write(f"Position: {done}\n")           
        task_episodes += 1
        total_episodes += 1
        for obj in selected_obj:
            obj = obj.lower().replace(' ', '_')
            obj_pos_dict.update({obj: []}) if obj not in obj_pos_dict else None
            obj_pos_dict[obj].append(np.hstack((obs[obj+"_1_pos"], obs[obj+"_1_quat"])).reshape(1, 7).tolist())
        #     # log_file.write(obj: obs[obj+"_1"] \n")
        # # Save a replay video of the episode
        # # save_rollout_video(
        # #     replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, mp4_path=mp4_path
        # # )
        
        # # Log current results
        # print(f"Success: {done}")
        # print(f"# episodes completed so far: {total_episodes}")
        # print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        # log_file.write(f"Success: {done}\n")
        log_file.write(f"# episodes completed so far: {total_episodes}\n")
        log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
        log_file.flush()
    save_rollout_video(
            replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, mp4_path=mp4_path
        )
    for obj in selected_obj:
        obj = obj.lower().replace(' ', '_')
        obj_pos_dict.update({obj: []}) if obj not in obj_pos_dict else None
        obj_pos_dict[obj].append(np.hstack((obs[obj+"_1_pos"], obs[obj+"_1_quat"])).reshape(1, 7).tolist())
    print("save_rollout_video success", mp4_path)    
    log_file.write(f"save_rollout_video success: {mp4_path}\n")
    task_info = create_task_dict(bddl_file_names[0], json_file, mp4_path,  obj_pos_dict)
    # # Log final results
    print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
    print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
    log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
    log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
    log_file.flush()

    # # Save local log file
    log_file.close()

    return task_info

if __name__ == "__main__":
    # add args
    args = argparse.ArgumentParser()
    args.add_argument("--category", type=str, default="behind") # model path
    
    args = args.parse_args()
    category = args.category
    cfg = GenerateConfig   
    # Load model

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    root_dir = ""## 
    
    grasp_root = ""

    grasp_track_name = "task"
    output_root = ""

    output_file = ""

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            task_dict = json.load(f)
    else:
        os.makedirs(os.path.join(output_root,grasp_track_name), exist_ok=True)
        task_dict = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            
            if file.endswith('new5.json'):
                json_file = os.path.join(root, file)
                task_name = os.path.basename(root)

                if task_name in task_dict:
                    print(f"Skipping {task_name} as it is already processed.")
                    continue
                with open(json_file, 'r') as f:
                    file_data = json.load(f)

                    parts = json_file.split("/")

                    save_path = os.path.join(output_root, f"{grasp_track_name}","/".join(parts[6:9]))
                    os.makedirs(save_path, exist_ok=True)
                    waypoint_paths = glob.glob(os.path.join(grasp_root, f"**/**/{grasp_track_name}","/".join(parts[6:9]), "grasp_waypoints_formatted_90_90_0_large_grip=8.npy"))
                   
                    if len(waypoint_paths) == 0:
                        # import pdb; pdb.set_trace()
                        continue
                    waypoint_path = waypoint_paths[0]
                    # if category not in waypoint_path:
                    #     continue
                    # import pdb; pdb.set_trace()
                    task_dict[os.path.basename(root)] = eval_libero(save_path, root, file_data, json_file, cfg, waypoint_path=waypoint_path)#, model)
                    # 将 task_dict 保存到 JSON 文件
                    with open(output_file, 'w') as f: 
                        json.dump(task_dict, f, indent=4)