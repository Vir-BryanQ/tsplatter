#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
from PIL import Image
from pathlib import Path
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.colmap_loader import Image as ColmapImage
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import random
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from nerfstudio.cameras import camera_utils
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: np.array
    cy: np.array
    image: np.array
    depth: np.array     # not used
    sam_mask: np.array  # modify -----
    mask_feat: np.array # modify -----
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        # 计算所有相机位置的平均中心和这些相机到该中心的最大距离（即对角线的长度）
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    # 半径：对角线长度的 1.1 倍，确保场景的范围能够完全包围住所有的相机
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = focal_length_x
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if not os.path.exists(image_path):
            # modify -----
            base, ext = os.path.splitext(image_path)
            if ext.lower() == ".jpg":
                image_path = base + ".png"
            elif ext.lower() == ".png":
                image_path = base + ".jpg"
            if not os.path.exists(image_path):
                continue
            # modify ----

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # NOTE: load SAM mask and CLIP feat. [OpenGaussian]
        mask_seg_path = os.path.join(images_folder[:-6], "language_features/" + extr.name.split('/')[-1][:-4] + "_s.npy")
        mask_feat_path = os.path.join(images_folder[:-6], "language_features/" + extr.name.split('/')[-1][:-4] + "_f.npy")
        if os.path.exists(mask_seg_path):
            sam_mask = np.load(mask_seg_path)    # [level=4, H, W]
        else:
            sam_mask = None
        if mask_feat_path is not None and os.path.exists(mask_feat_path):
            mask_feat = np.load(mask_feat_path)    # [level=4, H, W]
        else:
            mask_feat = None
        # modify -----

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=width/2, cy=height/2, image=image, 
                              depth=None, sam_mask=sam_mask, mask_feat=mask_feat,
                              image_path=image_path, image_name=image_name, width=width, height=height, fx=focal_length_x, fy=focal_length_y)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    if {'red', 'green', 'blue'}.issubset(vertices.data.dtype.names):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.random.rand(positions.shape[0], 3)
    if {'nx', 'ny', 'nz'}.issubset(vertices.data.dtype.names):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.random.rand(positions.shape[0], 3)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, 
                        orientation_method="up", center_method="poses", auto_scale_poses=True, scale_factor=1.0, assume_colmap_world_coordinate_convention=True, 
                        train_list_file=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "images.bin")
        cameras_intrinsic_file = os.path.join(path, "cameras.bin")

        # cam_extrinsics和cam_intrinsics都是字典
        # images[image_id] = Image(
        #         id=image_id, qvec=qvec, tvec=tvec,
        #         camera_id=camera_id, name=image_name,
        #         xys=xys, point3D_ids=point3D_ids)
        # OpenCV相机坐标系
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

        # cameras[camera_id] = Camera(id=camera_id,
        #                                 model=model_name,
        #                                 width=width,
        #                                 height=height,
        #                                 params=np.array(params))
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "images.txt")
        cameras_intrinsic_file = os.path.join(path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    poses = []
    im_ids = sorted(cam_extrinsics.keys())
    for im_id in im_ids:
        im_data = cam_extrinsics[im_id]
        rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)    
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)    # 转换为齐次矩阵
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1     # OpenCV -> OpenGL
        if assume_colmap_world_coordinate_convention:
            # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
            # 在 COLMAP 中，世界坐标系的重力猜测是 -y 方向，而在 nerfstudio 中，世界坐标系的重力猜测是 +z 方向
            # 向nerfstudio convention 的转换本质上等价于乘以矩阵：
            # 1 0 0
            # 0 0 1
            # 0 -1 0
            # 这个矩阵本质上是一个旋转矩阵，对应于绕 x 轴顺时针旋转 90 度
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1
        poses.append(c2w)
    poses = torch.from_numpy(np.array(poses).astype(np.float32))

    # 使用OpenGL相机坐标系约定
    poses, _ = camera_utils.auto_orient_and_center_poses(   
        poses,
        method=orientation_method,
        center_method=center_method
    )

    final_scale_factor = 1.0
    if auto_scale_poses:    # True
        # 取所有相机的位置并对每个坐标取绝对值，从所有位置的每个维度中，找出最大数值，得到的就是场景中离原点最远的坐标值
        # 将最远的相机位置缩放到 1.0 以内，使得整个场景被包裹在一个单位体积 [-1,1]^3 内
        final_scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    final_scale_factor *= scale_factor    # 1.0
    poses[:, :3, 3] *= final_scale_factor     # 将所有相机位置缩放到 1.0 以内，完成场景的“尺寸归一化”

    poses[:, 0:3, 1:3] *= -1    # OpenGL -> OpenCV

    im_ids = list(sorted(cam_extrinsics.keys()))
    for i, pose in enumerate(poses):
        # 构造 4x4 齐次矩阵
        c2w = torch.eye(4, dtype=pose.dtype, device=pose.device)
        c2w[:3, :4] = pose

        # world2camera = inverse(camera2world)
        w2c = torch.linalg.inv(c2w)

        # 提取 R, t (先转成 numpy)
        R = w2c[:3, :3].cpu().numpy()
        t = w2c[:3, 3].cpu().numpy()

        # 转四元数 (rotmat2qvec 是 numpy 版本)
        qvec = colmap_utils.rotmat2qvec(R)
        tvec = t

        im_id = im_ids[i]
        img = cam_extrinsics[im_id]
        cam_extrinsics[im_id] = ColmapImage(
            id=img.id, qvec=qvec, tvec=tvec,
            camera_id=img.camera_id, name=img.name,
            xys=img.xys, point3D_ids=img.point3D_ids)

    reading_dir = "images_rgb" if images == None else images
    # cam_infos_unsorted是一个list
    # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=width/2, cy=height/2, image=image, 
    #                           depth=None, sam_mask=sam_mask, mask_feat=mask_feat,
    #                           image_path=image_path, image_name=image_name, width=width, height=height)
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # 按图像名排序
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if train_list_file is not None:
        with (Path(path) / "train_lists" / f"{train_list_file}").open("r", encoding="utf8") as f:
            filenames = [line.strip() for line in f.read().splitlines() if line.strip()]
    else:
        try:
            with (Path(path) / "train_list.txt").open("r", encoding="utf8") as f:
                filenames = [line.strip() for line in f.read().splitlines() if line.strip()]
        except FileNotFoundError as e:
            print(f'Trivial FileNotFoundError: {e}')
            filenames = []
    image_filenames = [image.name for image in cam_extrinsics.values()]

    # 检测 split_filenames 中的文件名是否在 image_filenames 中存在
    unmatched_filenames = set(filenames).difference(image_filenames)   # 找出在 split_filenames 中 存在，但在 image_filenames 中 不存在 的文件名，即集合差值A - B
    if unmatched_filenames:
        raise RuntimeError(
            f"Some filenames for split {split} were not found: {set(map(str, unmatched_filenames))}."
        )

    filenames = [name.split('.')[0] for name in filenames]

    # self.eval = False
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        # 用于训练温度场的相机视角
        test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.image_name in filenames]


    # 根据多个相机的位置计算一个包围整个场景的球形范围，返回这个范围的中心（translate）和半径（radius）
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 我们不关心点云文件
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    ply_path = ''
    pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        # ----- modify -----
        if "camera_angle_x" not in contents.keys():
            fovx = None
        else:
            fovx = contents["camera_angle_x"] 
        # ----- modify -----

        # modify -----
        cx, cy = -1, -1
        if "cx" in contents.keys():
            cx = contents["cx"]
            cy = contents["cy"]
        elif "h" in contents.keys():
            cx = contents["w"] / 2
            cy = contents["h"] / 2
        # modify -----

        frames = contents["frames"]
        # for idx, frame in enumerate(frames):
        for idx, frame in tqdm(enumerate(frames), total=len(frames), desc="load images"):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1    # TODO

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            if not os.path.exists(image_path):
                # modify -----
                base, ext = os.path.splitext(image_path)
                if ext.lower() == ".jpg":
                    image_path = base + ".png"
                elif ext.lower() == ".png":
                    image_path = base + ".jpg"
                if not os.path.exists(image_path):
                    continue
                # modify ----

            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            # NOTE: load SAM mask and CLIP feat. [OpenGaussian]
            mask_seg_path = os.path.join(path, "language_features/" + frame["file_path"].split('/')[-1] + "_s.npy")
            mask_feat_path = os.path.join(path, "language_features/" + frame["file_path"].split('/')[-1] + "_f.npy")
            if os.path.exists(mask_seg_path):
                sam_mask = np.load(mask_seg_path)    # [level=4, H, W]
            else:
                sam_mask = None
            if os.path.exists(mask_feat_path):
                mask_feat = np.load(mask_feat_path)  # [num_mask, dim=512]
            else:
                mask_feat = None
            # modify -----

            # ----- modify -----
            if "K" in frame.keys():
                cx = frame["K"][0][2]
                cy = frame["K"][1][2]
            if cx == -1:
                cx = image.size[0] / 2
                cy = image.size[1] / 2
            # ----- modify -----

            # ----- modify -----
            if fovx == None:
                if "K" in frame.keys():
                    focal_length = frame["K"][0][0]
                if "fl_x" in contents.keys():
                    focal_length = contents["fl_x"]
                if "fl_x" in frame.keys():
                    focal_length = frame["fl_x"]
                FovY = focal2fov(focal_length, image.size[1])
                FovX = focal2fov(focal_length, image.size[0])
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovx 
                FovX = fovy
            # ----- modify -----

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, image=image, 
                            depth=None, sam_mask=sam_mask, mask_feat=mask_feat,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    if os.path.exists(os.path.join(path, "transforms_test.json")):
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    else:
        test_cam_infos = train_cam_infos
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}