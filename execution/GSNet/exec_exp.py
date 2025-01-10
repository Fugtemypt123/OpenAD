import json
from gsnet import grasp_inference
import numpy as np
import os
import glob

if __name__ == '__main__':
    root_dir = ""
    
    result_files = glob.glob(os.path.join(root_dir, "**/**/**/output/result.json"))




    save_dir0 = "/mnt/afs/spatial/experiments/GSNet/output/exp_ours_ft/0"
    save_dir1 = "/mnt/afs/spatial/experiments/GSNet/output/exp_ours_ft/1"
    save_dir2 = "/mnt/afs/spatial/experiments/GSNet/output/exp_ours_ft/2"
    if not os.path.exists(save_dir0):
        os.makedirs(save_dir0)
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    extrinsics = np.array([[-5.55111512e-17,  2.58174524e-01, -9.66098295e-01,
        1.60000000e+00],
    [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00],
    [ 0.00000000e+00, -9.66098295e-01, -2.58174524e-01,
        1.30000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00]])


    for result_file in result_files:
        task_dir = result_file.split("output_ft/result.json")[0]
        pc_path = os.path.join(task_dir, "pc_front_cam.ply")
        obj_pc_path = os.path.join(task_dir, "output/picked_obj.npy")
        # task_dir = "/mnt/afs/spatial/experiments/inputs/6dof_task_refine_rot/behind/Place_the_USB_behind_the_bottle_on_the_table.__plug_right/20240824-224716_no_interaction/output"
        save_path1 = os.path.join(save_dir0, "/".join(task_dir.split("/")[-6:-1]), "grasp_waypoints.npy")
        save_path2 = os.path.join(save_dir1, "/".join(task_dir.split("/")[-6:-1]), "grasp_waypoints.npy")
        save_path3 = os.path.join(save_dir2, "/".join(task_dir.split("/")[-6:-1]), "grasp_waypoints.npy")
        if os.path.exists(save_path1) or os.path.exists(save_path2) or os.path.exists(save_path3):
            continue
        if not os.path.exists(os.path.join(save_dir0, "/".join(task_dir.split("/")[-6:-1]))):
            os.makedirs(os.path.join(save_dir0, "/".join(task_dir.split("/")[-6:-1])))
        if not os.path.exists(os.path.join(save_dir1, "/".join(task_dir.split("/")[-6:-1]))):
            os.makedirs(os.path.join(save_dir1, "/".join(task_dir.split("/")[-6:-1])))
        if not os.path.exists(os.path.join(save_dir2, "/".join(task_dir.split("/")[-6:-1]))):
            os.makedirs(os.path.join(save_dir2, "/".join(task_dir.split("/")[-6:-1])))

        mask_path = os.path.join(task_dir, "picked_obj_mask.npy")
        obj_path = os.path.join(task_dir, "picked_obj.npy")    

        # pc_path = os.path.join(task_dir, "scene_pcd.npy")
        json_path = os.path.join(task_dir, "output_ft/result.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        translation_global = np.array(data['translation'])
        rotation_global = np.array(data['rotation'])
        obj_bb_x = data['x_thresh']
        obj_bb_y = data['y_thresh']
        obj_bb_z = data['z_thresh']
        # transfer this pc from glob frame to camera frame using extrinsics
        obj_pc_glob = np.load(obj_pc_path)
        R = extrinsics[:3, :3]  # Rotation matrix
        t = extrinsics[:3, 3]   # Translation vector
        positions_glob = obj_pc_glob[:, :3]  # Shape: (N, 3)
        attributes = obj_pc_glob[:, 3:]

        # Compute the inverse rotation and translation
        R_inv = R.T             # Transpose of R
        t_inv = -R_inv @ t
        # Translate and then rotate
        positions_cam = (positions_glob - t) @ R#.T
       
        obj_pc_cam = np.hstack((positions_cam, attributes))
        for i in range(3):
            delta = 0.1 * i
            x_thresh = [data['x_thresh'][0] - delta, data['x_thresh'][1] + delta] #list: [a, b]
            y_thresh = [data['y_thresh'][0] - delta, data['y_thresh'][1] + delta] #list: [a, b]
            z_thresh = [data['z_thresh'][0], data['z_thresh'][1] + delta+0.1] #list: [a, b]
        # Extract rotation and translation components



            if i < 0:
                all_waypoints = grasp_inference(pc_path, extrinsics, rotation_global, translation_global, save_path1, save_path2, save_path3, mask=None, x_thresh=None, y_thresh=y_thresh, z_thresh=z_thresh, obj_pc_cam=obj_pc_cam) 

            else:
                all_waypoints = grasp_inference(pc_path, extrinsics, rotation_global, translation_global, save_path1, save_path2, save_path3, mask=None, x_thresh=x_thresh, y_thresh=y_thresh, z_thresh=z_thresh)
            if all_waypoints is not None and len(all_waypoints[0]) > 0:
                break

        if all_waypoints is None or len(all_waypoints[0]) == 0:
            all_waypoints = grasp_inference(pc_path, extrinsics, rotation_global, translation_global, save_path1, save_path2, save_path3, mask=None, x_thresh=x_thresh, y_thresh=y_thresh, z_thresh=z_thresh)

 

    

