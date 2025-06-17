
import argparse
import numpy as np
import pickle
import os
import open3d as o3d

import pdb
from dwm.utils.lidar_visualizer_utils import *


map_tables = [
    {'token': '7e25a2c8ea1f41c5b0da1e69ecfa71a2', 'logfile': 'n015-2018-07-24-11-22-45+0800', 'vehicle': 'n015', 'date_captured': '2018-07-24', 'location': 'singapore-onenorth'}, 
    {'token': '53cf9c55dd8644bea67b9f009fc1ee38', 'logfile': 'n008-2018-08-01-15-16-36-0400', 'vehicle': 'n008', 'date_captured': '2018-08-01', 'location': 'boston-seaport'}, 
    {'token': '6f7fe59adf984e55a82571ab4f17e4e2', 'logfile': 'n008-2018-08-27-11-48-51-0400', 'vehicle': 'n008', 'date_captured': '2018-08-27', 'location': 'boston-seaport'}, 
    {'token': '881dd2f8aaec49b681eb54be5bf3b3e2', 'logfile': 'n008-2018-08-28-16-43-51-0400', 'vehicle': 'n008', 'date_captured': '2018-08-28', 'location': 'boston-seaport'}, 
    {'token': '3a43824b84534c98bda1d07548db5817', 'logfile': 'n008-2018-08-30-15-16-55-0400', 'vehicle': 'n008', 'date_captured': '2018-08-30', 'location': 'boston-seaport'}, 
    {'token': '8ff48ad1df8e4966a2151730c92b7f3c', 'logfile': 'n015-2018-10-02-10-50-40+0800', 'vehicle': 'n015', 'date_captured': '2018-10-02', 'location': 'singapore-queenstown'}, 
    {'token': '5bd40b613ac740cd9dbacdfbc3d68201', 'logfile': 'n015-2018-10-08-15-36-50+0800', 'vehicle': 'n015', 'date_captured': '2018-10-08', 'location': 'singapore-queenstown'}, 
    {'token': '8fefc430cbfa4c2191978c0df302eb98', 'logfile': 'n015-2018-11-21-19-38-26+0800', 'vehicle': 'n015', 'date_captured': '2018-11-21', 'location': 'singapore-hollandvillage'}]

hdmap_bev_settings ={
    'bev_from_ego_transform': [[6.4, 0, 0, 320], [0, 6.4, 0, 320], [0, 0, 6.4, 0], [0, 0, 0, 1]], 
    'fill_map': False, 'pen_width': 4}

color_table = {
    'drivable_area': (0, 0, 255), 
    'lane': (0, 255, 0), 
    'ped_crossing': (255, 0, 0)
}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_type', type=str, choices=['nuscenes', 'kitti360'], default='nuscenes', help='specify the data type')
    parser.add_argument('--lidar_root', type=str, default='demo_data',
                        help='The path to the generated lidar data.')
    parser.add_argument('--data_root', type=str, default='data/nuscenes', help='The data root that contains the json information of the nuscenes or the annotation file of kitti360.')
    parser.add_argument('--data_info_path', type=str, default='test_nuscenes_infos.pkl', help='Path to save data info. If the path exists, the data info will be loaded from the path. Otherwise, the data info will be generated and saved to the path.')
    parser.add_argument('--output_path', type=str, default='lidar_ns_visual_output', help='Output path.')
    parser.add_argument('--fov', type=float, default=60.0, help='specify the fov')
    parser.add_argument('--center', nargs='+', default=[0, 0, 0], help='specify the center')
    parser.add_argument('--eye', nargs='+', default=[22, 44, 44], help='specify the eye')
    args = parser.parse_args()
    return args
def check_box_in_range(box, x_min, x_max, y_min, y_max):
    return box[0] >= x_min and box[0] <= x_max and box[1] >= y_min and box[1] <= y_max


def line_sets_to_pts(line_sets, num_points_per_meter=3):
    """
    Save a list of Open3D line sets as a single PLY file, converting lines to point clouds.
    
    Args:
        line_sets (list): List of Open3D LineSet objects.
        filename (str): Output PLY filename.
        num_points_per_line (int): Number of points to generate per line segment.
    """
    combined_points = []
    combined_colors = []
    
    for line_set in line_sets:
        points = np.asarray(line_set.points)
        colors = np.asarray(line_set.colors)
        lines = np.asarray(line_set.lines)
        
        for line, color in zip(lines, colors):
            start = points[line[0]]
            end = points[line[1]]
            num_points_per_line = int(np.linalg.norm(start - end) * num_points_per_meter)
            # Generate points along the line
            t = np.linspace(0, 1, num_points_per_line)
            line_points = start[None, :] * (1 - t)[:, None] + end[None, :] * t[:, None]
            
            combined_points.extend(line_points)
            combined_colors.extend([color] * num_points_per_line)
    
    # Create a point cloud from the combined data
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(combined_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(combined_colors))
    
    return pcd


def draw_polygon_to_bev_3d(
        polygon: dict, nodes: list, transform: np.array,
        pen_color: tuple, num_points_per_meter=5
    ):
    polygon_nodes = np.array([
            [nodes[i]["x"], nodes[i]["y"], 0, 1]
            for i in polygon["exterior_node_tokens"]
    ]).transpose()
    p = transform @ polygon_nodes
    line_set = o3d.geometry.LineSet()
    p = p[:3, :]
    p = - p[[1, 0, 2], :]
    p[1, :] = - p[1, :]
    # p[2, :] = - p[2, :]
    p = p - np.array([[0, 0, 2.5]]).transpose()
    # p[0, :] = - p[0, :]
    # p[1, :] = - p[1, :]

    line_set.points = o3d.utility.Vector3dVector(p.transpose())
    lines = np.array([[i,i+1] for i in range(p.shape[1]-1)])
    lines = np.concatenate([lines, np.array([[p.shape[1]-1, 0]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.array([pen_color] * p.shape[1]) / 255.)
    pcd = line_sets_to_pts([line_set], num_points_per_meter=num_points_per_meter)
    return pcd

def filter_points_in_range(points, x_min, x_max, y_min, y_max):
    filtered_points = []
    for point in points:
        np_pts = np.array(point.points)
        np_pts = np_pts[:, :3]
        indices = np.logical_and(np_pts[:, 0] >= x_min , 
                                 np.logical_and(np_pts[:, 0] <= x_max, 
                                 np.logical_and(np_pts[:, 1] >= y_min , 
                                                np_pts[:, 1] <= y_max)))
        np_pts = np_pts[indices, :]
        np_colors = np.array(point.colors)
        np_colors = np_colors[indices, :]
        filtered_points.append([np_pts, np_colors])
    return filtered_points
        
def draw_hdmap_bev(info, map_expansion, map_expansion_dict, location):
    map = map_expansion[location]
    map_dict = map_expansion_dict[location]
    nodes = map_dict["node"]
    polygons = map_dict["polygon"]
    bev_from_world = info["car_from_global"]
    lidar_points = []
    
    if "drivable_area" in color_table and "drivable_area" in map:
        pen_color = tuple(color_table["drivable_area"])
        for i in map["drivable_area"]:
            for polygon_token in i["polygon_tokens"]:
                lidar_points.append(draw_polygon_to_bev_3d(
                    polygons[polygon_token], nodes, bev_from_world,
                    (0, 0, 255)))

    if "ped_crossing" in color_table and "ped_crossing" in map:
        pen_color = tuple(color_table["ped_crossing"])
        for i in map["ped_crossing"]:
            lidar_points.append(draw_polygon_to_bev_3d(
                polygons[i["polygon_token"]], nodes, bev_from_world,
                (255, 0, 0)))

    if "lane" in color_table and "lane" in map:
        pen_color = tuple(color_table["lane"])
        for i in map["lane"]:
            lidar_points.append(draw_polygon_to_bev_3d(
                polygons[i["polygon_token"]], nodes, bev_from_world, pen_color))
    return lidar_points

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    # remove last two lines which represent the car head
    lines = lines[:-2]
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_point_box(gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    output_pts = []
    for i in range(gt_boxes.shape[0]):
        line_set, _ = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])
        output_pts.append(line_set)
    return output_pts


def visualize_lidar_and_boxes():
    args = parse_config()
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    # create data info
    if args.data_info_path is not None and os.path.exists(args.data_info_path):
        with open(args.data_info_path, 'rb') as f:
            data_info_all = pickle.load(f)
        data_info = []
        for d in data_info_all:
            if d is not None:
                lidar_path = d['lidar_path']
                if args.data_type == 'nuscenes':
                    target_lidar_path = os.path.join(args.lidar_root, os.path.basename(lidar_path))
                elif args.data_type == 'kitti360':
                    target_lidar_path = os.path.join(args.lidar_root, d['scene_name'], os.path.basename(lidar_path))
                if os.path.exists(target_lidar_path):
                    d['lidar_path'] = target_lidar_path
                    data_info.append(d)
    else:
        if args.data_type == 'nuscenes':
            nusc = NuScenesDataset(version="v1.0-trainval", dataroot=args.data_root, verbose=True)
            available_scenes = get_available_scenes(nusc, lidar_root = args.lidar_root)
            available_scene_names = [s['name'] for s in available_scenes]
            from dwm.datasets.nuscenes_common import val
            val_scenes = val
            val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
            val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])
            data_info = fill_infos_nuscenes(
                data_path=args.lidar_root, nusc=nusc, val_scenes=val_scenes
            )
        elif args.data_type == 'kitti360':
            kitti360 = Kitti360Dataset(dataroot=args.data_root)
            data_info = fill_infos_kitti360(
                data_path=args.lidar_root, kitti360=kitti360
            )
        with open(args.data_info_path, 'wb') as f:
            pickle.dump(data_info, f)

    # load map expansion
    map_expansion, map_expansion_dict = load_map(os.path.join(args.data_root, "expansion"), map_tables)

    os.makedirs(args.output_path, exist_ok=True)
    for idx, data_dict in enumerate(data_info):
        if data_dict is None:
            continue
        # check if map expansion is available
        if len(map_expansion) > 0 and len(map_expansion_dict) > 0:
            lidar_base_name = os.path.basename(data_dict['lidar_path'])
            location = None
            for table in map_tables:
                if table['logfile'] in lidar_base_name:
                    location = table['location']
                    break
            if location is None:
                continue
        
        # Create a new renderer for each iteration to avoid memory issues
        render = o3d.visualization.rendering.OffscreenRenderer(1600, 900)
        # Set up materials
        pc_material = o3d.visualization.rendering.MaterialRecord()
        pc_material.base_color = [168 / 255., 168 / 255., 87 / 255., 1.0]
        pc_material.shader = "defaultLit"
        pcd = o3d.geometry.PointCloud()
        if data_dict['lidar_path'].endswith('.bin'):
            points = np.fromfile(data_dict['lidar_path'], dtype=np.float32).reshape(-1, 5)
            points = points[:, :3]
        elif data_dict['lidar_path'].endswith('.npy'):
            points = np.load(data_dict['lidar_path'])
        pcd.points = o3d.utility.Vector3dVector(points)
        render.scene.add_geometry("pointcloud", pcd, pc_material)
        pcd = o3d.geometry.PointCloud()
        
        # draw gt boxes
        gt_boxes = None
        gt_labels = []
        gt_boxes = []
        for i in range(len(data_dict['gt_names'])):
            if data_dict['gt_names'][i] in class_names:
                box = data_dict['gt_boxes'][i]
                if check_box_in_range(box, -50, 50, -50, 50):
                    gt_labels.append(class_names.index(data_dict['gt_names'][i]))
                    gt_boxes.append(data_dict['gt_boxes'][i])
        if len(gt_boxes) == 0:
            continue
        gt_labels = np.array(gt_labels)
        gt_boxes = np.array(gt_boxes)
        for index, (label, box) in enumerate(zip(gt_labels, gt_boxes)):
            color_box = draw_point_box(box[None,...], color=box_colormap[label])
            color_box = line_sets_to_pts(color_box)
            pcd += color_box
            box_material = o3d.visualization.rendering.MaterialRecord()
            box_material.base_color = box_colormap[label].tolist() + [1.0]
            box_material.shader = "defaultLit"
            render.scene.add_geometry(f"boxes_{index}", color_box, box_material)
        if len(map_expansion) > 0 and len(map_expansion_dict) > 0:
            hdmap_points = draw_hdmap_bev(data_dict, map_expansion, map_expansion_dict,                                       location)
            hdmap_points = filter_points_in_range(hdmap_points, -50, 50, -50, 50)
            hdmap_pcd = o3d.geometry.PointCloud()
            for index, (pc, color) in enumerate(hdmap_points):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)
                pcd.colors = o3d.utility.Vector3dVector(color)
                hdmap_pcd += pcd
                render.scene.add_geometry(f"hdmap_{index}", pcd, pc_material)
        render.scene.set_background([255., 255., 255., 1.])
        # Set up camera view        
        fov = args.fov
        center = args.center
        eye = args.eye
        render.setup_camera(fov, center, eye, [0, 0, 1])

        # Render and save image
        img = render.render_to_image()
        if args.data_type == 'nuscenes':
            output_basename = os.path.basename(data_dict['lidar_path'])[:-8]
            image_path = f"{args.output_path}/{output_basename}_topfront.png"
        elif args.data_type == 'kitti360':
            output_basename = os.path.basename(data_dict['lidar_path'])[:-4]
            image_path = os.path.join(args.output_path, data_dict['scene_name'], f"{output_basename}_topfront.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
        o3d.io.write_image(image_path, img)
        print(f"Saved visualization using offscreen renderer to {image_path}")
        
        # Explicitly clean up resources
        render.scene.clear_geometry()
        del render

    print('Demo done.')


if __name__ == '__main__':
    visualize_lidar_and_boxes()
