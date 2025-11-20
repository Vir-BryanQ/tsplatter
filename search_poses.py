import torch
import numpy as np
import os
import sys
from torch.nn.functional import normalize
from collections import defaultdict
from scene.colmap_loader import read_extrinsics_binary
from nerfstudio.cameras import camera_utils
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils

def compute_cosine_similarity(q1, q2):
    return torch.dot(q1 / q1.norm(dim=-1), q2 / q2.norm(dim=-1))

# 计算平移向量的欧几里得距离
def euclidean_distance(t1, t2):
    return torch.norm(t1 - t2)

# 读入图像文件列表
def read_image_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# 主要的处理函数
def process_camera_poses(path, k, orientation_method="up", center_method="poses", auto_scale_poses=True, scale_factor=1.0, assume_colmap_world_coordinate_convention=True):
    # 读取训练和测试文件
    train_list = read_image_list(os.path.join(path, 'full_train_list.txt'))
    test_list = read_image_list(os.path.join(path, 'test_list.txt'))

    cameras_extrinsic_file = os.path.join(path, "images.bin")
    # cam_extrinsics和cam_intrinsics都是字典
    # images[image_id] = Image(
    #         id=image_id, qvec=qvec, tvec=tvec,
    #         camera_id=camera_id, name=image_name,
    #         xys=xys, point3D_ids=point3D_ids)
    # OpenCV相机坐标系
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

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

    pose_dict = {}
    im_ids = list(sorted(cam_extrinsics.keys()))
    for i, pose in enumerate(poses):
        im_id = im_ids[i]
        img = cam_extrinsics[im_id]
        pose_dict[img.name] = pose

    # 划分训练集和测试集
    train_dict = {key: value for key, value in pose_dict.items() if key in train_list}
    test_dict = {key: value for key, value in pose_dict.items() if key in test_list}

    # 存储所有位姿对的差异和对应的图像名
    differences = []

    # 遍历训练集和测试集
    for train_img, train_pose in train_dict.items():
        train_rotation = torch.tensor(colmap_utils.rotmat2qvec(train_pose[:, :3].cpu().numpy()))  # 假设旋转矩阵为3x3
        train_translation = train_pose[:, 3]  # 假设平移向量为3x1

        for test_img, test_pose in test_dict.items():
            test_rotation = torch.tensor(colmap_utils.rotmat2qvec(test_pose[:, :3].cpu().numpy()))
            test_translation = test_pose[:, 3]

            # 计算旋转差异（四元数的余弦值）
            rot_diff = 1 - compute_cosine_similarity(train_rotation, test_rotation)

            # 计算平移差异（欧几里得距离）
            # translation_diff = euclidean_distance(train_translation, test_translation)
            translation_diff = 1 - compute_cosine_similarity(train_translation, test_translation)

            # 总差异为旋转差异和平移差异的加权和
            # total_diff = rot_diff + 0.5 * translation_diff
            total_diff = rot_diff + translation_diff

            # 记录差异
            differences.append((total_diff.item(), train_img, test_img, rot_diff.item(), translation_diff.item()))

    # 按照总差异排序
    differences.sort(key=lambda x: x[0])

    # 输出前k个最小差异的训练图像名（确保训练图像名互不相同）
    output_train_images = []
    result = []

    for diff, train_img, test_img, rot_diff, translation_diff in differences:
        # if (train_img == 'FLIR3188.jpg' and test_img == 'FLIR3187.jpg'):
        #     print(diff, rot_diff, translation_diff)
        #     sys.exit(0)

        if len(output_train_images) >= k:
            break
        if train_img not in output_train_images:
            output_train_images.append(train_img)
            result.append((diff, train_img, test_img, rot_diff, translation_diff))

    # 输出前k个结果
    for img in result:
        print(img)
    for img in result:
        print(img[1])

# 从命令行获取路径和k值
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_files> <k>")
        sys.exit(1)

    path = sys.argv[1]
    k = int(sys.argv[2])
    torch.set_grad_enabled(False)

    process_camera_poses(path, k)