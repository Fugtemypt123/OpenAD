import os
import sys
import numpy as np
import time
import torch
import open3d as o3d
from copy import deepcopy
# from graspnetAPI.graspnet_eval import GraspGroup
from graspnetAPI import GraspGroup, Grasp
# os.environ["OPEN3D_RENDERING_BACKEND"] = "egl"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector, FrankaCollisionDetector
from scipy.spatial.transform import Rotation as R, Slerp
import plotly.graph_objects as go

# from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask

class GSNet():
    def __init__(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        class Config():
            pass
        self.cfgs = Config()
        self.cfgs.dataset_root = f'{dir}/data/datasets/graspnet'
        self.cfgs.checkpoint_path = f'{dir}/assets/minkuresunet_realsense_tune_epoch20.tar'
        self.cfgs.dump_dir = 'logs'
        self.cfgs.seed_feat_dim = 512
        self.cfgs.camera = 'realsense'
        self.cfgs.num_point = 100000
        self.cfgs.batch_size = 1
        self.cfgs.voxel_size = 0.005
        self.cfgs.collision_thresh = 0.01#
        self.cfgs.voxel_size_cd = 0.01
        self.cfgs.infer = False
        self.cfgs.vis = False
        self.cfgs.scene = '0188'
        self.cfgs.index = '0000'
        
    def inference(self, cloud_masked, max_grasps=500):
        """Inference grasp from point cloud

        Args:
            cloud_masked (np.ndarray): masked point cloud
            max_grasps (int, optional): max number of grasps to return. Defaults to 200.

        Returns:
            GraspGroup: GraspGroup object
        """
        # sample points random
        if len(cloud_masked) >= self.cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
            # print("sampled point cloud idxs:", idxs.shape)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        data_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                     'coors': cloud_sampled.astype(np.float32) / self.cfgs.voxel_size,
                     'feats': np.ones_like(cloud_sampled).astype(np.float32)}
        
        batch_data = minkowski_collate_fn([data_dict])
        net = GraspNet(seed_feat_dim=self.cfgs.seed_feat_dim, is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        # print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

        net.eval()
        tic = time.time()

        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
                
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            if end_points is None:
                return None 
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)
        
        # collision detection
        if self.cfgs.collision_thresh > 0:

            cloud = data_dict['point_clouds']

            # Model-free collision detector
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size_cd)
            collision_mask_mfc = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            gg = gg[~collision_mask_mfc]

            # # Franka collision detector
            # fcdetector = FrankaCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size_cd)
            # collision_mask_fc, global_iou_fc = fcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            # gg = gg[~collision_mask_fc]
        
        gg = gg.nms()
        gg = gg.sort_by_score()
        
        if gg.__len__() > max_grasps:
            gg = gg[:max_grasps]

        return gg
    
    def visualize(self, cloud, gg: GraspGroup = None, g: Grasp = None, display = True, save_image: str = "output/grasp.png", save_pc: str = "output/grasp.ply"):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cloud)
        if not display:
            os.environ.pop("DISPLAY", None)
        pcd = cloud
    
        geoms = []
        if gg is not None:
         
            grippers = gg.to_open3d_geometry_list()   
            # o3d.visualization.draw_geometries([pcd, *grippers])
            geoms.extend(grippers)
        elif g is not None:
           
            gripper = g.to_open3d_geometry()
            # o3d.visualization.draw_geometries([pcd, gripper])
            geoms.append(gripper)
        else:
            pass
            # o3d.visualization.draw_geometries([pcd])
        # combined_pcd = o3d.geometry.PointCloud()
        # for geom in geoms:
        
        # # Save the combined point cloud to .ply
        # o3d.io.write_point_cloud(save_pc, combined_pcd)
        # print(f"Saved point cloud to {save_pc}")
        combined_mesh = o3d.geometry.TriangleMesh()
        for geom in geoms:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                combined_mesh += geom

        # Save the combined mesh as a .ply file
        if len(combined_mesh.vertices) > 0:
            o3d.io.write_triangle_mesh(save_pc, combined_mesh)
            print(f"Saved combined mesh to {save_pc}")
        else:
            print("No mesh data to save.")
  
        if display:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=display)
            vis.add_geometry(combined_mesh)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_image)
            print(f"Saved visualization to {save_image}")
            vis.destroy_window()

def visualize_plotly_single(pc, rgb,g: Grasp, extrinsics, max_points=100000, gg_glob=False,rotation=None):
    g_visual = deepcopy(g)
    gg = GraspGroup()
    gg.add(g_visual)
    if gg_glob == False:
        gg.transform(extrinsics)
    gg.rotation_matrices = rotation
    gripper = gg.to_open3d_geometry_list()
    gripper = gripper[0]
    vertices = np.asarray(gripper.vertices)
    triangles = np.asarray(gripper.triangles)
    color = np.asarray(gripper.vertex_colors)[0]
    color = (color * 255).astype(np.uint8)
    color = f'rgb({color[0]}, {color[1]}, {color[2]})'
    grasps_plotly = [go.Mesh3d(
        x=vertices[:, 0], 
        y=vertices[:, 1], 
        z=vertices[:, 2], 
        i=triangles[:, 0], 
        j=triangles[:, 1], 
        k=triangles[:, 2], 
        color=color, 
        opacity=1,
    )]
    pc = pc[:, :3]
    stride = max(1, pc.shape[0] // max_points)
    pc = pc[::stride]
    pc = pc @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    pc_plotly = [go.Scatter3d(
        x=pc[:, 0], 
        y=pc[:, 1], 
        z=pc[:, 2], 
        mode='markers', 
        marker=dict(size=2, color=rgb, opacity=0.8)
        # marker=dict(size=3, color='lightgreen')
    )]
    
    fig = go.Figure(data=pc_plotly + grasps_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

def visualize_plotly(pc, rgb, gg: GraspGroup, extrinsics, max_points=100000, gg_glob=False):
    """
    Args:
    - pc: np.ndarray[N, 3], point cloud in camera frame
    - gg: GraspGroup, grasps in camera frame
    - extrinsics: np.ndarray[4, 4], extrinsics from camera to table
    - max_points: int, maximum number of points to visualize
    """
    if gg is not None:
        gg_visual = deepcopy(gg)
        if gg_glob == False:
            gg_visual.transform(extrinsics)
        grippers = gg_visual.to_open3d_geometry_list()   

        # visualize grasps
        grasps_plotly = []
        for grasp in grippers:
            vertices = np.asarray(grasp.vertices)
            triangles = np.asarray(grasp.triangles)
            color = np.asarray(grasp.vertex_colors)[0]
            color = (color * 255).astype(np.uint8)
            color = f'rgb({color[0]}, {color[1]}, {color[2]})'
            grasps_plotly.append(go.Mesh3d(
                x=vertices[:, 0], 
                y=vertices[:, 1], 
                z=vertices[:, 2], 
                i=triangles[:, 0], 
                j=triangles[:, 1], 
                k=triangles[:, 2], 
                color=color, 
                opacity=1,
            ))
    else:
        grasps_plotly = []
    
    # visualize pc
    pc = pc[:, :3]
    stride = max(1, pc.shape[0] // max_points)
    pc = pc[::stride]
    pc = pc @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    pc_plotly = [go.Scatter3d(
        x=pc[:, 0], 
        y=pc[:, 1], 
        z=pc[:, 2], 
        mode='markers', 
        marker=dict(size=2, color=rgb, opacity=0.8)
        # marker=dict(size=3, color='lightgreen')
    )]
    fig = go.Figure(data=pc_plotly + grasps_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()


def rot_matrices2quat(rotations):
    # convert rotation matrices to quaternions
    quats = []
    for rot in rotations:
        quat = R.from_matrix(rot).as_quat()
        quats.append(quat)
    return quats
def quat2rot_matrice(quat):
    
    return R.from_quat(quat).as_matrix()
class PlacementPlanner():
    def __init__(
        self, 
        pregrasp_distance: float,
        lift_height: float, 
        retract_distance: float, 
        extrinsics: np.ndarray,
        table_height=None,
    ):
        """
        Args:
        - pregrasp_distance: float, distance to move towards the object
        - lift_height: float, height to lift the object
        - retract_distance: float, distance to move away from the object
        """
        self.pregrasp_distance = pregrasp_distance
        self.lift_height = lift_height
        self.retract_distance = retract_distance
        self.extrinsics = extrinsics
        self.table_height = table_height
    
    @staticmethod
    def transform_gg(
        gg: GraspGroup, 
        T: np.ndarray,
    ):
        gg = deepcopy(gg)
        if len(T.shape) == 2:
            return gg.transform(T)
        else:
            rotation = T[:, :3, :3]
            translation = T[:, :3, 3]
            gg.translations = (rotation @ gg.translations.reshape(-1, 3, 1)).reshape(-1, 3) + translation
            gg.rotation_matrices = np.matmul(rotation, gg.rotation_matrices).reshape((-1, 3, 3)) # (-1, 9)
            return gg

    def plan(
        self, 
        pc: np.ndarray, 
        gg: GraspGroup, 
        extrinsics: np.ndarray, 
        relative_translation_table: np.ndarray,
        relative_rotation_table: np.ndarray,
    ) -> list:
        """
        Args:
        - pc: np.ndarray[N, 3], point cloud in camera frame
        - gg: GraspGroup, grasps in camera frame
        - extrinsics: np.ndarray[4, 4], extrinsics from camera to table
        - relative_translation_table: np.ndarray[3], relative translation of object in table frame
        - relative_rotation_table: np.ndarray[3, 3], relative rotation of object in table frame
        
        Returns:
        - waypoints: list[Grasp], waypoints in camera frame
        """
        
        num_grasps = len(gg)
        
        # sort grasps by score
        gg = gg.sort_by_score()
        
        # get transformation in camera frame
        relative_rotation_camera = extrinsics[:3, :3].T @ relative_rotation_table @ extrinsics[:3, :3]
        relative_translation_table = relative_translation_table @ extrinsics[:3, :3]
        
        # get goal grasps
        gg_goal = deepcopy(gg)
        gg_goal.rotation_matrices = relative_rotation_camera @ gg_goal.rotation_matrices
        gg_goal.translations = gg_goal.translations + relative_translation_table  # note that this is not an affine transformation
        
        # waypoint 1: pregrasp pose
        gg_xneg_camera = -gg.rotation_matrices[:, :, 0]
        pregrasp_side_trans_camera = gg_xneg_camera
        pregrasp_camera = np.eye(4).reshape(1, 4, 4).repeat(num_grasps, axis=0)
        pregrasp_camera[:, :3, 3] = pregrasp_side_trans_camera * self.pregrasp_distance
        gg_pregrasp = self.transform_gg(gg, pregrasp_camera)
        
        # waypoint 2: grasp pose (gg)
        
        # waypoint 3: squeeze pose
        gg_squeeze = deepcopy(gg)
        gg_squeeze.widths[:] = 0
        
        # waypoint 4: lift pose
        lift_up_trans_table = np.array([[0, 0, 1]]).repeat(num_grasps, axis=0)
        lift_up_trans_camera = lift_up_trans_table @ extrinsics[:3, :3]
        lift_camera = np.eye(4).reshape(1, 4, 4).repeat(num_grasps, axis=0)
        lift_camera[:, :3, 3] = lift_up_trans_camera * self.lift_height
        gg_lift = self.transform_gg(gg_squeeze, lift_camera)
        
        # waypoint 5: move pose
        gg_move = self.transform_gg(gg_goal, lift_camera)
        gg_move.widths[:] = 0
        
        # waypoint 6: place pose
        gg_place = deepcopy(gg_goal)
        gg_place.widths[:] = 0
        
        # waypoint 7: goal pose (gg_goal)
        
        # waypoint 8: retract pose
        gg_goal_xneg_camera = -gg_goal.rotation_matrices[:, :, 0]
        retract_side_trans_camera = gg_goal_xneg_camera
        retract_camera = np.eye(4).reshape(1, 4, 4).repeat(num_grasps, axis=0)
        retract_camera[:, :3, 3] = retract_side_trans_camera * self.retract_distance
        gg_retract = self.transform_gg(gg_goal, retract_camera)
        
        waypoints = [
            gg_pregrasp, 
            gg, 
            gg_squeeze, 
            gg_lift, 
            gg_move, 
            gg_place, 
            gg_goal, 
            gg_retract, 
        ]
        
        # filter 1: waypoints should not collide with the object
        mfcdetector = ModelFreeCollisionDetector(pc, voxel_size=0.01)
        mask_collision = np.ones(num_grasps, dtype=bool)
        for waypoint in [gg_pregrasp, gg, gg_goal, gg_retract]:
            mask_collision &= ~mfcdetector.detect(waypoint, approach_dist=0.05, collision_thresh=0.02)
        
        # filter 2: -x axis of goal grasps should point up
        goal_gg_xneg_camera = -gg_goal.rotation_matrices[:, :, 0]
        goal_gg_xneg_table = goal_gg_xneg_camera @ extrinsics[:3, :3].T
        mask_orientation = goal_gg_xneg_table[:, 2] > 0
        
        # filter 3: -x axis of grasps should point up
        gg_xneg_camera = -gg.rotation_matrices[:, :, 0]
        gg_xneg_table = gg_xneg_camera @ extrinsics[:3, :3].T
        mask_orientation &= gg_xneg_table[:, 2] > 0.6

        # filter 4: 
        gg_z_camera = gg.rotation_matrices[:, :, 2]  # z-axis of each grasp in camera frame
        gg_z_global = gg_z_camera @ extrinsics[:3, :3].T  # convert to global frame
        mask_vertical = np.abs(gg_z_global[:, 2]) < 0.8
        mask_orientation &= mask_vertical

        # filter 4: -x axis of grasps and goal grasps should point towards the agent
        # mask_orientation &= gg_xneg_table[:, 0] < 0.8
        # mask_orientation &= goal_gg_xneg_table[:, 0] < 0.8
        gg_reserved = gg[(mask_collision & mask_orientation)]
        # import pdb; pdb.set_trace()
        # visualize_plotly(pc, 100, gg_reserved, extrinsics)

        # filter 5: translation z of grasps in table coord should be above table_height:
        #transform gg to table frame via extrinsics
        # import pdb; pdb.set_trace()
        if self.table_height is not None:
            gg_glob = deepcopy(gg)
            gg_glob.transform(extrinsics)

            mask_orientation &= (gg_glob.translations[:, 2] >= self.table_height) 
            mask_orientation &= (gg_glob.translations[:, 0] >= 0)
            # visualize the filtered grasps and reserved ones with different color:
            gg_filtered = gg[~(mask_collision & mask_orientation)]
            gg_reserved = gg[(mask_collision & mask_orientation)]
            # plotly visualize filtered with blue and reserved with red
            # visualize_plotly(pc, 200, gg_filtered, extrinsics)
            # visualize_plotly(pc, 100, gg_reserved, extrinsics)


        # filter
        mask = mask_collision & mask_orientation
        # print("=========")
        # print(gg_xneg_table[mask][0])
        # print(goal_gg_xneg_table[mask][0])
        
        for i, waypoint in enumerate(waypoints):
            waypoints[i] = waypoint[mask]
            

        
        # return best
        # best_waypoints = []
        # for waypoint in waypoints:
        #     best_waypoints.append(waypoint[0])
        
        return waypoints



class Interpolator:
    def __init__(self, waypoints_gg, N):
        """
        Initialize the Interpolator with waypoints_gg and number of interpolation points.
        
        Parameters:
        - waypoints_gg: GraspGroup object containing the original grasps
        - N: Number of points to interpolate between each pair of grasps
        """
        self.waypoints_gg = waypoints_gg
        self.N = N
        self.gg_rots = waypoints_gg.rotation_matrices  # Extract rotation matrices
        self.gg_trans = waypoints_gg.translations  # Extract translations
        self.initial_grasps = deepcopy(waypoints_gg)  # Deep copy of the original GraspGroup
        self.interpolated_grasps = GraspGroup()  # Initialize the interpolated GraspGroup

    def interpolate(self):
        """
        Perform the interpolation between grasps in the GraspGroup.
        """
  

        # Iterate through each pair of sequential grasps
        for i in range(len(self.gg_trans) - 1):
            translation_start = np.array(self.gg_trans[i])
            translation_end = np.array(self.gg_trans[i + 1])
            rotation_start = R.from_matrix(self.gg_rots[i])
            rotation_end = R.from_matrix(self.gg_rots[i + 1])

            # Set up the Slerp interpolation for rotations
            key_times = np.array([0, 1])
            key_rotations = R.from_quat([rotation_start.as_quat(), rotation_end.as_quat()])
            slerp = Slerp(key_times, key_rotations)

            # Add the starting grasp
            g = deepcopy(self.initial_grasps[i])
            self.interpolated_grasps.add(g)

            # Generate N interpolated translations and rotations
            for t in np.linspace(0, 1, self.N + 2)[1:-1]:  # Exclude the endpoints
                interpolated_translation = (1 - t) * translation_start + t * translation_end
                interpolated_rotation = slerp(t).as_matrix()

                # Create a deep copy of the grasp and update its translation and rotation
                g = deepcopy(self.initial_grasps[i])
                g.translation = interpolated_translation
                g.rotation_matrix = interpolated_rotation  
                self.interpolated_grasps.add(g)

        # Add the final grasp
        g = deepcopy(self.initial_grasps[-1])
        self.interpolated_grasps.add(g)

    def get_interpolated_grasps(self):
        """
        Return the interpolated GraspGroup.
        """
        return self.interpolated_grasps


def grasp_inference(pointcloud_path, extrinsics, relative_rotation_table, relative_translation_table, save_path1, save_path2, save_path3,x_thresh=None, y_thresh=None, z_thresh=None,mask=None, obj_pc_cam=None):
    if False:#obj_pc_cam is not None:
        # pc = np.load(obj_pc_path, allow_pickle=True)
        pc = obj_pc_cam
        points = pc[:, :3]
        colors = pc[:, 3:]
    # pc = np.load(pointcloud_path, allow_pickle=True)
    else:
        pc = o3d.io.read_point_cloud(pointcloud_path)
        points = np.asarray(pc.points)
        colors = np.asarray(pc.colors)
    # points = pc[:, :3]
    # colors = pc[:, 3:]
    max_points = 100000
    stride = max(1, points.shape[0] // max_points)
    points = points[::stride]  
    colors = colors[::stride]  
    scene_pc = o3d.io.read_point_cloud(pointcloud_path)
    scene_points = np.asarray(scene_pc.points)
    scene_colors = np.asarray(scene_pc.colors)
    # x_thresh = [0.14,0.8]
    # y_thresh = [-0.3, 0.3]#[-0.5,0.5]
    # z_thresh = [0.8,1.2]
    # mask = (points[:, 0] >= x_thresh[0]) & (points[:, 0] <= x_thresh[1]) & (points[:, 1] >= y_thresh[0]) & (points[:, 1] <= y_thresh[1]) & (points[:, 2] >= z_thresh[0]) & (points[:, 2] <= z_thresh[1])
    # mask in camera frame

    # points_homog = np.hstack((points, np.ones((points.shape[0], 1))))
    # points_global = (extrinsics @ points_homog.T).T[:, :3]  # Transform and remove the homogeneous coordinate

    if isinstance(mask, np.ndarray):
        masked_points = points[mask]
        masked_colors = colors[mask]
    else:
    # Step 3: Apply mask to original points in camera frame
        if isinstance(x_thresh, list):
            mask = generate_mask(points, extrinsics, x_thresh, y_thresh, z_thresh)
            masked_points = points[mask]
            masked_colors = colors[mask] 
            if len(masked_points) < 10:  
                return None
        else:
            masked_points = points
            masked_colors = colors
    # visualize_plotly(masked_points, masked_colors, gg=None, extrinsics=extrinsics)

    # import pdb; pdb.set_trace()
    gsnet = GSNet()
    gg = gsnet.inference(masked_points)
    if gg is None:
        return None
    planner = PlacementPlanner(pregrasp_distance=0.4, lift_height=0.3, retract_distance=0.3, extrinsics=extrinsics, table_height=0.8)
    all_waypoints = planner.plan(masked_points, gg, extrinsics, relative_translation_table, relative_rotation_table)


    if len(all_waypoints[0]) == 0:
        print("No waypoints found.")
        return None

    for i in range(min([len(all_waypoints[0]), 3])): #len(all_waypoints[0])
        print(f"trying on the {i}th ranking:")
        current_best_waypoints = []
        for waypoint_id in all_waypoints:
            current_best_waypoints.append(waypoint_id[i])
        waypoints = current_best_waypoints
        waypoints_gg = GraspGroup()

        for waypoint in waypoints:
            waypoints_gg.add(waypoint)
        # import pdb; pdb.set_trace()
        # visualize_plotly(scene_points, scene_colors,waypoints_gg, extrinsics)

        waypoints_gg_glob = deepcopy(waypoints_gg)
        waypoints_gg_glob.transform(extrinsics)
        
        interpolator = Interpolator(waypoints_gg_glob, N=30)
        interpolator.interpolate()
        interpolated_grasp_group = interpolator.get_interpolated_grasps() # in global coordinate

        # rotation mats:
        gg_rots = interpolated_grasp_group.rotation_matrices
        # convert to quaternions:
        gg_rots = [R.from_matrix(rot).as_quat() for rot in gg_rots]
        # translations:
        gg_trans = interpolated_grasp_group.translations
        print(gg_trans)
        
        # store the interpolated grasps in a dict
        interpolated_waypoints =  [{'translation': trans, 'rotation': rot} for trans, rot in zip(gg_trans, gg_rots)]
        save_path = ""
        if i == 0:
            save_path = save_path1
        elif i == 1:
            save_path = save_path2
        else:
            save_path = save_path3
        np.save(save_path, interpolated_waypoints)
        # print(f"Saved interpolated waypoints to {save_path}")

        # visualize_plotly(points, colors, interpolated_grasp_group, extrinsics, gg_glob=True)

        i += 1
    
    return all_waypoints

def generate_mask(points, extrinsics, x_thresh=[0.14,0.8], y_thresh = [-0.3, 0.3], z_thresh = [0.8,1.2]):
    # max_points = 100000
    # stride = max(1, points.shape[0] // max_points)
    pc = points[:, :3]
    # pc = pc[::stride]
    points_global = pc @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    # points_homog = np.hstack((points, np.ones((points.shape[0], 1))))
    # points_global = (extrinsics @ points_homog.T).T[:, :3]  # Transform and remove the homogeneous coordinate

    # Step 2: Apply the bounding box filter in global frame
    mask_global = (
        (points_global[:, 0] >= x_thresh[0]) & (points_global[:, 0] <= x_thresh[1]) &
        (points_global[:, 1] >= y_thresh[0]) & (points_global[:, 1] <= y_thresh[1]) &
        (points_global[:, 2] >= z_thresh[0]) & (points_global[:, 2] <= z_thresh[1])
    )
    return mask_global
        
if __name__ == '__main__':
    
    # quat = [0,0,0,1]#[0,0.70710678   ,      0.     ,    0.70710678]#[0,0,0,1]
    # quat1 = [0,0,1,0]#[0,-0.70710678, 0.        ,    0.70710678]#[0,0,1,0]
    # quat2 = [0,1,0,0]
    # quat3 = [1,0,0,0]
    # rots = []
    # rots.append(quat2rot_matrice(quat))
    # rots.append(quat2rot_matrice(quat1))
    # rots.append(quat2rot_matrice(quat2))
    # rots.append(quat2rot_matrice(quat3))

    # rot = np.array([quat2rot_matrice(quat)])
    # rot1 = np.array([quat2rot_matrice(quat1)])
    # rot2 = np.array([quat2rot_matrice(quat2)])
    # rot3 = np.array([quat2rot_matrice(quat3)])


    # os.environ.pop("DISPLAY", None)
    # extrinsics = np.load('assets/extrinsics.npy')
    # extrinsics = np.array([[-7.26818450e-08,  6.28266450e-01, -7.77998244e-01,
    #      6.58613175e-01],
    #    [ 1.00000000e+00, -7.26818450e-08, -1.52115266e-07,
    #      0.00000000e+00],
    #    [-1.52115266e-07, -7.77998244e-01, -6.28266450e-01,
    #      1.61035002e+00],
    #    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #      1.00000000e+00]])
    # extrinsics_libero = np.array([[ 0.        , -0.57357643,  0.81915205, -0.65      ],
    #    [-1.        ,  0.        ,  0.        ,  0.        ],
    #    [ 0.        , -0.81915205, -0.57357643,  1.16      ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])#ours
    extrinsics = np.array([[-5.55111512e-17,  2.58174524e-01, -9.66098295e-01,
         1.60000000e+00],
       [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00, -9.66098295e-01, -2.58174524e-01,
         1.30000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]) #open6dor
  
#     extrinsics = np.array([[ 0.76604444,  0.,         -0.64278761, -0.5       ],
#  [ 0.64278761,  0. ,         0.76604444,  0      ],
#  [ 0. ,        -1.  ,        0.     ,    1.26      ],
#  [ 0. ,         0.  ,        0.     ,    1.        ]])
    # pc = np.load('assets/pc.npy')
    relative_rotation_table = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ])
    relative_translation_table = np.array([0, -0.25, 0])
    pc_path = "/data/workspace/LIBERO/experiments/pc/depth_open6dor_ours_agent_test6_cam.ply"
    save_path = "output/traj/interpolated_waypoints_open6dor_ours_test6.npy"
    grasp_inference(pc_path, extrinsics, relative_rotation_table, relative_translation_table, save_path1="output/waypoints1.npy", save_path2="output/waypoints2.npy", save_path3="output/waypoints3.npy")

    exit()
    # import pdb; pdb.set_trace()
    # # pc1 = o3d.io.read_point_cloud("/data/workspace/LIBERO/experiments/pc/depth_norm_color_ours_evalyf_final_cam.ply")#("/data/workspace/LIBERO/experiments/pc/depth_norm_color_ours_final_cam.ply")#("/data/workspace/LIBERO/experiments/pc/depth_norm_color.ply")
    # pc1 = o3d.io.read_point_cloud("/data/workspace/LIBERO/experiments/pc/depth_open6dor_ours_agent_test6_cam.ply")

    # # camera frame pointcloud
    # points = np.asarray(pc1.points)
    # colors = np.asarray(pc1.colors)

    # # open6dor mask in global frame
    # x_thresh = [0.14,0.8]
    # y_thresh = [-0.3, 0.3]#[-0.5,0.5]
    # z_thresh = [0.8,1.2]
    # # mask = (points[:, 0] >= x_thresh[0]) & (points[:, 0] <= x_thresh[1]) & (points[:, 1] >= y_thresh[0]) & (points[:, 1] <= y_thresh[1]) & (points[:, 2] >= z_thresh[0]) & (points[:, 2] <= z_thresh[1])
    # # mask in camera frame



    # # Step 1: Transform points from camera frame to global frame
    # # Convert `points` to homogeneous coordinates by adding a column of ones
    # max_points = 100000
    # stride = max(1, points.shape[0] // max_points)
    # pc = points[:, :3]
    # pc = pc[::stride]
    # points_global = pc @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    # # points_homog = np.hstack((points, np.ones((points.shape[0], 1))))
    # # points_global = (extrinsics @ points_homog.T).T[:, :3]  # Transform and remove the homogeneous coordinate

    # # Step 2: Apply the bounding box filter in global frame
    # mask_global = (
    #     (points_global[:, 0] >= x_thresh[0]) & (points_global[:, 0] <= x_thresh[1]) &
    #     (points_global[:, 1] >= y_thresh[0]) & (points_global[:, 1] <= y_thresh[1]) &
    #     (points_global[:, 2] >= z_thresh[0]) & (points_global[:, 2] <= z_thresh[1])
    # )
    

    # # Step 3: Apply mask to original points in camera frame
    # masked_points = points[mask_global]

    # masked_colors = colors[mask_global] 
    # # import pdb; pdb.set_trace()
    # # masked_points = points[mask]
    # # mask = (points[:, 0] > 0.2) & (points[:, 2] < 0.5) & (points[:, 0] <= 1.0)
    # # discard the points with x > 0
    # # mask = points[:, 0] < 0.0#(points[:, 0] > 0.0) & (points[:, 1] > 0.1) 

    # # masked_points = points[mask]
    # # masked_points = poi nts
    # # masked_points = points
    # # cloud = o3d.io.read_point_cloud(f"{ROOT_DIR}/assets/pointcloud_raft.ply")
    # gsnet = GSNet()
    # gg = gsnet.inference(masked_points)
    # visualize_plotly(masked_points, masked_colors, gg, extrinsics, gg_glob=False)

    # # visualize_plotly(pc, gg, extrinsics)
    
    # # g_test = deepcopy(gg[0])
    # # g_test1 = deepcopy(gg[1])
    

    # # visualize_plotly_single(masked_points, colors[mask], g_test, extrinsics,rotation=np.array([rots[0]]))
    # # visualize_plotly_single(masked_points, colors[mask], g_test1, extrinsics,rotation=np.array([rots[1]]))
    # # visualize_plotly_single(masked_points, colors[mask], g_test, extrinsics,rotation=np.array([rots[2]]))
    # # visualize_plotly_single(masked_points, colors[mask], g_test1, extrinsics,rotation=np.array([rots[3]]))
    # # import pdb; pdb.set_trace()
    # # gg = gsnet.inference(np.array(pc))
    # # visualize_plotly(pc, gg, extrinsics)
    
    # # plan
    

    # planner = PlacementPlanner(pregrasp_distance=0.4, lift_height=0.5, retract_distance=0.3, extrinsics=extrinsics, table_height=0.2)


    # # import pdb; pdb.set_trace()
   
    
    # all_waypoints = planner.plan(masked_points, gg, extrinsics, relative_translation_table, relative_rotation_table)
    # # import pdb; pdb.set_trace()
    # if len(all_waypoints[0]) == 0: 
    #     print("No waypoints found.")
    #     exit()
    # i = 0
    # for i in range(len(all_waypoints[0])):
    #     print(f"trying on the {i}th ranking:")
    #     current_best_waypoints = []
    #     for waypoint_id in all_waypoints:
    #         current_best_waypoints.append(waypoint_id[i])
    #     waypoints = current_best_waypoints
    #     waypoints_gg = GraspGroup()

    #     for waypoint in waypoints:
    #         waypoints_gg.add(waypoint)
    #     import pdb; pdb.set_trace()
    #     visualize_plotly(points, colors,waypoints_gg, extrinsics)


    #     # identical matrix
    
    #     # visualize_plotly(points, colors, waypoints_gg, extrinsics)  
    #     # gsnet.visualize(cloud, gg)        

    #     # transfer gg_rots and gg_trans to dict
    #     # waypoints_gg = [{'translation': trans, 'rotation': rot} for trans, rot in zip(gg_trans, gg_rots)]

    #     waypoints_gg_glob = deepcopy(waypoints_gg)
    #     waypoints_gg_glob.transform(extrinsics)
        
    #     interpolator = Interpolator(waypoints_gg_glob, N=20)
    #     interpolator.interpolate()
    #     interpolated_grasp_group = interpolator.get_interpolated_grasps() # in global coordinate
    #     if False:
    #         continue
    #     # rotation mats:
    #     gg_rots = interpolated_grasp_group.rotation_matrices
    #     # convert to quaternions:
    #     gg_rots = [R.from_matrix(rot).as_quat() for rot in gg_rots]
    #     # translations:
    #     gg_trans = interpolated_grasp_group.translations
    #     print(gg_trans)
        
    #     # store the interpolated grasps in a dict
    #     interpolated_waypoints =  [{'translation': trans, 'rotation': rot} for trans, rot in zip(gg_trans, gg_rots)]
    #     save_path = "output/traj/interpolated_waypoints_open6dor_ours_test4.npy"
    #     np.save(save_path, interpolated_waypoints)
    #     print(f"Saved interpolated waypoints to {save_path}")


    #     visualize_plotly(points, colors, interpolated_grasp_group, extrinsics, gg_glob=True)
    #     i += 1
    #     if True:
    #         break
        