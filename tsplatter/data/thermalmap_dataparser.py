from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import open3d as o3d
import torch
from natsort import natsorted
from rich.console import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
    ColmapDataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.process_data.colmap_utils import colmap_to_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600

@dataclass
class ThermalMapDataParserConfig(ColmapDataParserConfig):
    _target: Type = field(default_factory=lambda: ThermalMapDataParser)
    rgb_ckpt_path: Path = Path("RGB_Scene.ckpt")

class ThermalMapDataParser(ColmapDataParser):
    config: ThermalMapDataParserConfig

    def __init__(self, config: ThermalMapDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None

    def _load_rgb_scene(self, ckpt_path: str) -> dict:
            """
            从指定的 checkpoint 文件中加载 RGB 场景（高斯参数）。

            参数：
                ckpt_path (str): checkpoint 文件路径（.ckpt）

            返回：
                dict: 仅包含以 "_model.gauss_params." 开头的参数（去掉前缀后的 key）
            """
            checkpoint = torch.load(Path(ckpt_path), map_location="cpu")
            pipeline_state_dict = checkpoint.get("pipeline", {})
            prefix = "_model.gauss_params."
            gauss_params = {
                key[len(prefix):]: value
                for key, value in pipeline_state_dict.items()
                if key.startswith(prefix)
            }
            return gauss_params
    
    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        meta = self._get_all_images_and_cameras(self.config.data)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]    # "OPENCV": CameraType.PERSPECTIVE

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))

            # 0.0
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            image_filenames.append(Path(frame["file_path"]))    # 这里获取的是文件路径
            # 相机pose是C2W
            poses.append(frame["transform_matrix"])

            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        poses = torch.from_numpy(np.array(poses).astype(np.float32))    # 将一个 NumPy 格式的相机位姿列表 poses，转换为 PyTorch 的 FloatTensor 类型，方便用于神经网络中计算
        # 自动调整所有相机的位姿（poses）方向与位置，使得场景模型在空间中对齐方向（如 Z 轴朝上）并居中（如模型中心在原点）
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(   
            poses,
            method=self.config.orientation_method,  # "up"
            center_method=self.config.center_method, # "poses"
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:    # True
            # 取所有相机的位置并对每个坐标取绝对值，从所有位置的每个维度中，找出最大数值，得到的就是场景中离原点最远的坐标值
            # 将最远的相机位置缩放到 1.0 以内，使得整个场景被包裹在一个单位体积 [-1,1]^3 内
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor    # 1.0
        poses[:, :3, 3] *= scale_factor     # 将所有相机位置缩放到 1.0 以内，完成场景的“尺寸归一化”

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        # 根据 split 获取图像索引
        indices = self._get_image_indices(image_filenames, split)
        image_filenames, mask_filenames, depth_filenames, downscale_factor = self._setup_downscale_factor(
            image_filenames, mask_filenames, depth_filenames
        )

        image_filenames = [image_filenames[i] for i in indices]     # 提取用于指定 split 的图像
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        # 根据索引列表 indices，选出对应的相机位姿 poses 子集
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        # 构造一个立方体范围的 AABB，用于定义场景边界或可渲染区域
        # [-aabb_scale, -aabb_scale, -aabb_scale]是 AABB 的最小点
        # [aabb_scale, aabb_scale, aabb_scale]是 AABB 的最大点
        # 这里的scene_box是[-1,1]^3
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        # 根据索引列表 idx_tensor 选取对应split的相机参数
        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]     # 0.0

        # 由于 fx fy表示的是归一化成像平面上单位长度的像素数，缩放变换不会影响归一化成像平面上像素的大小，即不会影响fx和fy的值
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],  # 每个C2W是 3x4矩阵
            camera_type=camera_type,    # CameraType.PERSPECTIVE    存在广播机制
        )

        # 调整相机的内参，使其适应缩放后GT的分辨率
        cameras.rescale_output_resolution(
            scaling_factor=1.0 / downscale_factor, scale_rounding_mode=self.config.downscale_rounding_mode  # floor
        )

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            # applied_transform + auto_orient_and_center_poses
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            # 无
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        metadata = {}
        metadata.update(self._load_rgb_scene(self.config.data / self.config.rgb_ckpt_path))

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,    # 图像路径
            cameras=cameras,             # 相机内外参
            scene_box=scene_box,        # 场景包围盒 [-1,1]^3
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,  # 包围盒缩放
            dataparser_transform=transform_matrix,  # applied_transform + auto_orient_and_center_poses
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,     # RGB场景
            },
        )
        return dataparser_outputs

ThermalMapDataParserSpecification = DataParserSpecification(
    config=ThermalMapDataParserConfig(),
    description="ThermalMap: modified thermal version of Colmap dataparser",
)