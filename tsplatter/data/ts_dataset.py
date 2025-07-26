from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.utils.rich_utils import CONSOLE

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

def get_viewmat1(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1

    # flip the z and y axes to align with gsplat conventions
    # gsplat的相机坐标系约定和OpenCV相同？
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)

    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)

    # 构建完整的 4×4 viewmat
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


class TSDataset(InputDataset):
    cameras: Cameras

    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0
    ):
        super().__init__(dataparser_outputs, scale_factor)

    def render_image(self, image_idx: int):
        with torch.no_grad():
            camera = self.cameras[image_idx : image_idx + 1]

            BLOCK_WIDTH = 16
            viewmat = get_viewmat1(camera.camera_to_worlds).cuda() 
            K = camera.get_intrinsics_matrices().cuda()
            W, H = int(camera.width.item()), int(camera.height.item())
            means = self.metadata['means'].cuda()
            quats = self.metadata['quats'].cuda()
            scales = self.metadata['scales'].cuda()
            opacities = self.metadata['opacities'].cuda()
            features_dc = self.metadata['features_dc'].cuda()
            features_rest = self.metadata['features_rest'].cuda()
            colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
            sh_degree_to_use = int((colors.shape[-2] ** 0.5) - 1)

            rgb, alpha, _ = rasterization(
                means=means,
                quats=quats / quats.norm(dim=-1, keepdim=True),   # 实际上不归一化也是可以的
                scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities).squeeze(-1),
                colors=colors,
                viewmats=viewmat, 
                Ks=K, 
                width=W,
                height=H,
                tile_size=BLOCK_WIDTH,  
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode="classic",  # 默认是 "classic" 模式
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )

        return rgb, alpha


    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")
        
        image_rgb, alpha = self.render_image(image_idx)

        data = {"image_idx": image_idx, "image": image, "image_rgb": image_rgb, "alpha": alpha}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data
