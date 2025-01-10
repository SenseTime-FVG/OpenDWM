import av
import bisect
import json
from PIL import Image
import torch
import os
import aoss_client.client
import time

# The walkaround for video rotation meta (waiting for side data support on pyav
# streams)
# Note: Linux should apt install the mediainfo package for the shared library
# files.
import pymediainfo
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
import io
import pickle
from pypcd4 import PointCloud


class MotionDataset(torch.utils.data.Dataset):

    CAMS = ['center_camera_fov30', 'center_camera_fov120', 'left_front_camera',
            'left_rear_camera', 'rear_camera', 'right_front_camera', 'right_rear_camera']
    LIDARS = ['top_center_lidar']

    def parse_valid_file(self):
        rt = []
        cur_time = time.time()
        for rid, root in enumerate(self.roots):
            batches, prefix = self.strategies[rid]
            for bid in range(*batches):
                # only 0-4 have raw_cloud
                if prefix == "gac_baidu_segments_batch":
                    batch_dir = os.path.join(root, f'{prefix}_{bid}')
                    seqs = list(self.client.list(batch_dir))
                else:
                    cur = f'annotations/valid_sequence/batch_{bid}_valid_sequence.txt'
                    seqs = self.client.get(os.path.join(root, cur))

                    # bytes decode
                    seqs = seqs.decode(
                        encoding="utf-8", errors="strict").split('\n')

                for seq in seqs:
                    batch_dir = os.path.join(root, f'{prefix}_{bid}', seq)
                    num_modalities = len(list(self.client.list(batch_dir)))
                    vid_dir = os.path.join(
                        batch_dir, 'image_undistort', self.cams[0])
                    full_imgs = list(self.client.list(vid_dir))
                    full_imgs = sorted(full_imgs)

                    full_len = len(full_imgs)
                    for st in range(0, full_len - self.sequence_length * self.fps_stride_tuples[0], self.fps_stride_tuples[1]):
                        cpaths = list(map(lambda x: os.path.join(
                            vid_dir, x),  full_imgs[st: st+self.sequence_length*self.fps_stride_tuples[0]: self.fps_stride_tuples[0]]))
                        rt.append(dict(pth=cpaths, type=rid,
                                  num_modalities=num_modalities))
        print(time.time() - cur_time)

        return rt

    def project_cloud_to_image(self, pc, image, intrinsic, lidar_extrinsic, out_path):
        lidar_in_cam = lidar_extrinsic @ np.vstack(
            (pc[0:3][:], np.ones_like(pc[1])))
        lidar_in_cam = lidar_in_cam[:, lidar_in_cam[2] > 0]  # 只保留相机前面的物体/点云
        pc_xyz_ = lidar_in_cam[:3, :]
        pc_xyz = pc_xyz_ / pc_xyz_[2, :]
        image_point = intrinsic @ pc_xyz
        image_point = image_point[:2, :]
        image_limit_w = np.logical_and(
            image_point[0, :] >= 0, image_point[0, :] <= image.shape[1])
        image_limit_h = np.logical_and(
            image_point[1, :] >= 0, image_point[1, :] <= image.shape[0])
        image_limit = np.logical_and(image_limit_h, image_limit_w)
        image_point = image_point[:, image_limit]
        # distance
        cam_z = pc_xyz_[2, image_limit]
        cam_z = cam_z[np.newaxis, :]
        # concanate depth sparse info
        dsparse = np.concatenate((image_point, cam_z), axis=0)
        im_color = cv2.applyColorMap(
            (cam_z/200 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        im_color = im_color.reshape(
            (im_color.shape[1], im_color.shape[0], im_color.shape[2]))
        image = image.copy()
        for idx in range(image_point.shape[1]):
            try:
                color = (int(im_color[idx][0][0]), int(
                    im_color[idx][0][1]), int(im_color[idx][0][2]))
                cv2.circle(image, (int(round(image_point[0][idx])), int(
                    round(image_point[1][idx]))), 2, color, -1)
            except:
                import pdb
                pdb.set_trace()
        if not out_path is None:
            cv2.imwrite(out_path, image)
        # return dsparse info for future use
        return image, dsparse

    def __init__(
        self, client_config_path, roots, strategies, sequence_length: int, fps_stride_tuples: list,
        cams=None, lidars=None, cache_dir=None,
        visualization_projection_dir=None
    ):
        self.client = aoss_client.client.Client(client_config_path)
        self.roots = roots
        self.strategies = strategies
        self.fps_stride_tuples = fps_stride_tuples
        self.visualization_projection_dir = visualization_projection_dir
        if self.fps_stride_tuples[0] > 10:
            raise ValueError
        self.fps = self.fps_stride_tuples[0]
        self.fps_stride_tuples[0] = int(10 / self.fps_stride_tuples[0])

        self.sequence_length = sequence_length

        self.cams = cams if cams is not None else self.CAMS
        self.lidars = lidars if lidars is not None else self.LIDARS
        self.poses = dict()

        skip_load = False
        if cache_dir is not None:
            cache_name = '_'.join(list(map(lambda x: x.split('/')[-1], roots)))
            cache_pth = f'{cache_dir}/{cache_name}_{sequence_length}_{fps_stride_tuples[0]}_{fps_stride_tuples[1]}.pkl'
            if os.path.exists(cache_pth):
                self.items = pickle.load(open(cache_pth, 'rb'))
                skip_load = True
                print("Skip parse dataset items")

        if not skip_load:
            self.items = self.parse_valid_file()
        if cache_dir is not None and (not skip_load):
            with open(cache_pth, "wb") as f:
                pickle.dump(self.items, f)

        self.items = list(
            filter(lambda x: x['num_modalities'] > 1, self.items))
        # self.get_gop_calib()

    def _parse_pose(self, pose: str):
        for tinfo in pose.split('\n'):
            if tinfo == '':
                continue
            tinfos = tinfo.split(' ')
            img_key, arr = tinfos[0], np.loadtxt(
                io.StringIO(' '.join(tinfos[1:])))
            arr = arr.reshape(-1, 4)
            self.poses[img_key] = arr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        metas = self.items[index]
        cam0_paths, data_type = metas['pth'], metas['type']
        from_gop = True
        result = {
            "fps": self.fps
        }
        images, lidars, ego_transforms = [], [], []

        # top_center_lidar-lo-pose-abs
        top_center_lidar_to_pose_pth = os.path.join(
            os.path.dirname(os.path.dirname(
                cam0_paths[0].replace('image_undistort', 'pose'))),
            'top_center_lidar-lo-pose-abs.txt'
        )
        ctfm_bytes = self.client.get(
            top_center_lidar_to_pose_pth).decode(encoding="utf-8", errors="strict")
        self._parse_pose(ctfm_bytes)

        # Load top_center_lidar-to-car_center-extrinsic
        top_center_lidar_to_car_center_pth = os.path.join(
            os.path.dirname(cam0_paths[0].replace(
                'image_undistort', 'calib').replace(self.cams[0], 'top_center_lidar')),
            'top_center_lidar-to-car_center-extrinsic.json'
        )
        tmp = json.loads(
            self.client.get(top_center_lidar_to_car_center_pth).decode(encoding="utf-8", errors="strict"))
        top_center_lidar_to_car_center_inv = np.linalg.inv(np.array(tmp[
            'top_center_lidar-to-car_center-extrinsic']['param']['sensor_calib']['data']))

        for tid, pth in enumerate(cam0_paths):
            img_key = os.path.basename(pth).split('.')[0]

            # load cams
            mv_images, mv_calibs = [], []
            for cid, cam in enumerate(self.cams):
                cpth = pth.replace(self.cams[0], cam)
                cimg_bytes = self.client.get(cpth)

                assert (cimg_bytes is not None)
                img = Image.open(io.BytesIO(cimg_bytes))
                mv_images.append(img)         # [v.shape for v in rts]

                # load calib
                # intrinsic
                cpth_calib = cpth.replace('image_undistort', 'calib')
                dirname = os.path.dirname(cpth_calib)
                isc_pth = f'{dirname}/{cam}-intrinsic.json'
                cjs_data = json.loads(self.client.get(isc_pth).decode(
                    encoding="utf-8", errors="strict"))
                if from_gop:
                    isc = np.array(cjs_data['value0']
                                   ['param']['cam_K']['data'])
                else:
                    ref_isc = self.mv_isc[cid]
                    isc = np.array(
                        cjs_data[f'{cam}-intrinsic']['param']['cam_K']['data'])
                    isc[0, 0] = ref_isc[0, 0] / ref_isc[0, 2] * isc[0, 2]
                    isc[1, 1] = ref_isc[1, 1] / ref_isc[1, 2] * isc[1, 2]
                mv_calibs.append(isc)

            images.append(mv_images)

            lid = self.lidars[0]
            cpth = pth.replace(self.cams[0], lid).replace(
                'image_undistort', 'raw_cloud').split('.')[0]
            cpth = cpth + '.pcd'
            cpcd_bytes = self.client.get(cpth)

            pcd_array = PointCloud.from_fileobj(
                io.BytesIO(cpcd_bytes)).numpy()[:, :3]

            # Debug code: Load calib & visual
            # for cid, cam in enumerate(self.cams):
            #     cpth_calib = pth.replace('image_undistort', 'calib').replace(self.cams[0], lid)
            #     dirname = os.path.dirname(cpth_calib)
            #     esc_pth = f'{dirname}/{lid}-to-{self.cams[cid]}-extrinsic.json'
            #     cjs_data = json.loads(self.client.get(esc_pth).decode(encoding="utf-8", errors="strict"))
            #     if from_gop:
            #         esc = np.array(cjs_data[f'{lid}-to-{self.cams[cid]}-extrinsic']['param']['sensor_calib']['data'])
            #     else:
            #         esc = np.array(cjs_data[f'{lid}-to-{self.cams[cid]}-extrinsic']['param']['sensor_calib']['data'])
            #     if self.visualization_projection_dir is not None:
            #         self.project_cloud_to_image(pcd_array.T, np.array(mv_images[cid])[..., ::-1], mv_calibs[cid], esc,
            #             f'{self.visualization_projection_dir}/{index}_{tid}_{self.cams[cid]}.jpg')
            cur_pose = np.concatenate(
                [self.poses[img_key], [[0, 0, 0, 1.0]]], axis=0)
            ego_transform = cur_pose @ top_center_lidar_to_car_center_inv
            lidars.append(pcd_array)
            ego_transforms.append(ego_transform)

        result['images'] = images
        result['lidar_points'] = lidars
        result['ego_transforms'] = ego_transforms

        return result


if __name__ == '__main__':
    cur = MotionDataset(
        "/mnt/afs/user/wuzehuan/aoss.conf",
        ["s3://gac2"],
        [[(0, 1), "gac_baidu_segments_batch"]],
        sequence_length=6,
        fps_stride_tuples=[10, 1],
        visualization_projection_dir='/mnt/afs/user/nijingcheng/workspace/codes/sup_codes/DWM/work_dirs/shell_logs/visual_gac',
        cache_dir='/mnt/afs/user/nijingcheng/workspace/codes/sup_codes/DWM/work_dirs/shell_logs/visual_gac/cache'
    )
    for i in range(1, 1000, 100):
        x = cur[i]