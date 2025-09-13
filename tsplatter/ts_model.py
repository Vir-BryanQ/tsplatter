import sys
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from concurrent.futures import ThreadPoolExecutor
from rich.progress import track
from pathlib import Path
import torchvision.utils as vutils
import numpy as np

import torch, gc
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import SSIM

try:
    from gsplat.rendering import rasterization
    from gsplat.rendering import rasterization_thermal
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat import rasterize_gaussians
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from gsplat.cuda_legacy._wrapper import num_sh_bases
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import (
    RGB2SH,
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
)
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import colormaps
from tsplatter.data.ts_dataset import TSDataset
from tsplatter.utils.normal_utils import normal_from_depth_image
from tsplatter.losses import NormalLoss, NormalLossType
from tsplatter.utils.color_utils import rgb_to_lab, lab_to_rgb

def unproject_depth_to_world(depth_im, Ks, viewmats):
    C, W, H, _ = depth_im.shape

    # 创建像素网格 [W, H]
    u = torch.linspace(0, W - 1, W, device=depth_im.device) # 生成一个从 0 到 W-1 等间距的长度为 W 的 1D 向量 [W]
    v = torch.linspace(0, H - 1, H, device=depth_im.device) # [H]
    u_grid, v_grid = torch.meshgrid(u, v, indexing="ij")  # [W, H]

    # 像素坐标转为齐次形式 [3, W, H]
    ones = torch.ones_like(u_grid)
    pixel_coords = torch.stack([u_grid, v_grid, ones], dim=0)  # [3, W, H]

    # 扩展维度以匹配 batch size
    pixel_coords = pixel_coords[None].repeat(C, 1, 1, 1)  # [C, 3, W, H]

    # 深度展开成 [C, 1, W, H]
    depth = depth_im.permute(0, 3, 1, 2)  # [C, 1, W, H]

    # 内参逆 [C, 3, 3]
    K_inv = torch.inverse(Ks)  # [C, 3, 3]

    # 相机坐标：X_cam = K^-1 @ [u, v, 1] * depth
    pixel_coords_flat = pixel_coords.reshape(C, 3, -1)             # [C, 3, WH]
    cam_points = torch.bmm(K_inv, pixel_coords_flat)            # [C, 3, WH]
    cam_points = cam_points * depth.reshape(C, 1, -1)              # 乘以深度：点乘，每个像素放大到实际距离

    # 变换到世界坐标（先构造相机到世界的变换）
    view_to_world = torch.inverse(viewmats)  # [C, 4, 4]

    # 添加齐次维度 1：-> [C, 4, WH]
    cam_points_homo = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], dim=1)  # [C, 4, WH]

    # 点变换：X_world = T @ X_cam
    world_points = torch.bmm(view_to_world, cam_points_homo)  # [C, 4, WH]

    # 只要前三维
    world_points = world_points[:, :3]  # [C, 3, WH]

    # 还原成图像结构 [C, W, H, 3]
    world_points = world_points.reshape(C, 3, W, H).permute(0, 2, 3, 1)  # [C, W, H, 3]

    return world_points

def assign_thermal_colors(means: torch.Tensor, 
                          world_points: torch.Tensor, 
                          thermal_images: torch.Tensor, 
                          k: int) -> torch.Tensor:
    """
    为 means 中的每个点赋予 thermal_colors，通过查找其在 world_points 中最近的 k 个点，
    然后将这 k 个点的颜色（来自 thermal_images）做平均。
    
    参数:
    - means: [N, 3] 的三维点张量
    - world_points: [M, 3] 的三维点张量
    - thermal_images: [M, 3] 的 RGB 颜色张量，对应于 world_points
    - k: 最近邻个数

    返回:
    - thermal_colors: [N, 3] 的 RGB 平均颜色张量
    """

    # 转为 numpy
    means_np = means.cpu().numpy()
    world_np = world_points.cpu().numpy()
    thermal_np = thermal_images.cpu().numpy()

    from sklearn.neighbors import NearestNeighbors

    # 建立最近邻模型
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nn.fit(world_np)

    # 查找每个 means 点的最近邻
    distances, indices = nn.kneighbors(means_np)  # indices: [N, k]

    # 取出这些邻居的颜色，求平均
    neighbor_colors = thermal_np[indices]        # shape: [N, k, 3]
    color_means = neighbor_colors.mean(axis=1)   # shape: [N, 3]

    # 转为 Tensor 返回
    thermal_colors = torch.from_numpy(color_means).to(dtype=means.dtype, device=means.device)
    return thermal_colors

def num_sh_bases1(degree: int):
    return degree + 1

def get_scale_loss(scales):
    """Scale loss"""
    # loss to minimise gaussian scale corresponding to normal direction
    scale_loss = torch.min(torch.exp(scales), dim=1, keepdim=True)[0].mean()
    return scale_loss


@dataclass
class TSplatterModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: TSplatterModel)
    # warmup_length: int = 500
    warmup_length: int = 10
    """period of steps where refinement is turned off"""
    # refine_every: int = 100
    refine_every: int = 10
    """period of steps where gaussians are culled and densified"""
    # resolution_schedule: int = 3000
    resolution_schedule: int = 150
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    # reset_alpha_every: int = 30
    reset_alpha_every: int = 2
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    # sh_degree_interval: int = 1000
    sh_degree_interval: int = 1
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    # stop_split_at: int = 15000
    stop_split_at: int = 150
    """stop splitting at this step"""
    sh_degree: int = 0
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    predict_normals: bool = False
    """Whether to extract and render normals or skip this"""
    use_normal_loss: bool = False
    """Enables normal loss('s)"""
    use_normal_tv_loss: bool = False
    """Use TV loss on predicted normals."""
    smooth_loss_lambda: float = 0.1
    """Regularizer for smooth loss"""
    normal_lambda: float = 0.1
    """Regularizer for normal loss"""
    use_scale_loss: bool = False
    use_rgb_loss: bool = True
    disable_refinement: bool = False
    use_vanilla_sh: bool = False
    use_merge_sparsification: bool = True # disable_refinement should be False
    stop_merge_at: int = 150

class TSplatterModel(SplatfactoModel):
    config: TSplatterModelConfig

    def __init__(
        self,
        *args,
        train_dataset: TSDataset, 
        **kwargs,
    ):
        self.train_dataset = train_dataset
        super().__init__(*args, **kwargs)

    # 在super().__init__()中调用了populate_modules()
    def populate_modules(self):
        dataset = self.train_dataset
        self.xys_grad_norm = None
        self.max_2Dsize = None

        BLOCK_WIDTH = 16
        viewmats = get_viewmat(dataset.cameras.camera_to_worlds).cuda() 
        Ks = dataset.cameras.get_intrinsics_matrices().cuda()
        W, H = int(dataset.cameras.width[0, 0].item()), int(dataset.cameras.height[0, 0].item())
        means = dataset.metadata['means'].cuda()
        quats = dataset.metadata['quats'].cuda()
        scales = dataset.metadata['scales'].cuda()
        opacities = dataset.metadata['opacities'].cuda()
        features_dc = dataset.metadata['features_dc'].cuda()
        features_rest = dataset.metadata['features_rest'].cuda()
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
        sh_degree_to_use = int((colors.shape[-2] ** 0.5) - 1)

        depth_im, alpha, _ = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),   # 实际上不归一化也是可以的
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities).squeeze(-1),
            colors=colors,
            viewmats=viewmats, 
            Ks=Ks, 
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,  
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="ED",
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",  # 默认是 "classic" 模式
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).permute(0, 2, 1, 3)    # [C,W,H,1]

        world_points = unproject_depth_to_world(depth_im=depth_im, Ks=Ks, viewmats=viewmats)    # [C,W,H,3]

        def _load_all_thermal(idx: int) -> torch.Tensor:
            data = dataset.get_data(idx, image_type="float32")  # 默认是 float32
            camera = dataset.cameras[idx].reshape(())
            # 如果相机的宽度和高度与图像的尺寸不匹配，则抛出异常
            assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
                f'The size of image ({data["image"].shape[1]}, {data["image"].shape[0]}) loaded '
                f'does not match the camera parameters ({camera.width.item(), camera.height.item()})'
            )
            return data["image"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            thermal_images = list(
                track(
                    executor.map(
                        _load_all_thermal,
                        range(len(dataset)),
                    ),
                    description=f"loading thermal images",   
                    transient=True,     
                    total=len(dataset),  
                )
            )
        thermal_images = torch.stack(thermal_images, dim=0).permute(0, 2, 1, 3).cuda() # [C,W,H,1]

        assert thermal_images.shape[:3] == world_points.shape[:3], (
            f'The shape of world_points {world_points.shape} does not match the loaded thermal_images {thermal_images.shape}'
            )

        # [C, W, H, 3] -> [CWH, 3]
        world_points = world_points.reshape(-1, 3)
        thermal_images = thermal_images.reshape(-1, 3)

        if self.config.use_vanilla_sh:
            dim_sh = num_sh_bases(self.config.sh_degree)
        else:
            dim_sh = num_sh_bases1(self.config.sh_degree)

        thermal_colors = assign_thermal_colors(means=means, thermal_images=thermal_images, world_points=world_points, k=3)

        shs = torch.zeros((means.shape[0], dim_sh, 3)).float().cuda()
        if self.config.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(thermal_colors)
        else:
            CONSOLE.log("use color only optimization with sigmoid activation")
            shs[:, 0, :3] = torch.logit(thermal_colors, eps=1e-10)

        thermal_features_dc = torch.nn.Parameter(shs[:, 0, :]) 
        thermal_features_rest = torch.nn.Parameter(shs[:, 1:, :])

        # 在 get_param_groups() 中获取
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "thermal_features_dc": thermal_features_dc,
                "thermal_features_rest": thermal_features_rest,
                "opacities": opacities,

                # RGB对应的球谐系数不参与优化
                "features_dc": features_dc,
                "features_rest": features_rest,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        if self.config.use_normal_loss:
            self.normal_loss = NormalLoss(NormalLossType.L1)
        if self.config.use_normal_tv_loss:
            self.normal_tv_loss = NormalLoss(NormalLossType.Smooth)

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            # 默认是随机背景色
            # self.background_color在非训练时使用
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]    # 深蓝灰色
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

    @property
    def thermal_features_dc(self):
        return self.gauss_params["thermal_features_dc"]

    @property
    def thermal_features_rest(self):
        return self.gauss_params["thermal_features_rest"]

    @property
    def thermal_colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.thermal_features_dc)
        else:
            return torch.sigmoid(self.thermal_features_dc)
    
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            # 和 PyTorch 优化器（如 torch.optim.Adam(params: Iterable[Tensor], ...)）兼容接口
            # 优化器期望传入参数是 可迭代的对象（如列表），即便只优化一个参数，也得放在 [param] 里
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "thermal_features_dc", "thermal_features_rest", "opacities"]
        }

    # 获取优化器的参数组
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps) # off
        return gps
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        # return super().get_outputs(camera=camera)

        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            # 某些数据源的相机位姿可能存在系统性误差或需要手动校准。因此，需要对原始相机的位姿做调整（correction）
            # apply_to_camera： 对相机的 camera-to-world 位姿矩阵应用校正（pose correction），比如旋转补偿、平移微调等，返回 修正后的相机位姿
            # 默认返回camera.camera_to_worlds
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            # 训练时不裁剪
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            thermal_features_dc_crop = self.thermal_features_dc[crop_ids]
            thermal_features_rest_crop = self.thermal_features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
        else:
            # 训练时不裁剪
            opacities_crop = self.opacities
            means_crop = self.means
            thermal_features_dc_crop = self.thermal_features_dc
            thermal_features_rest_crop = self.thermal_features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest

        colors_crop = torch.cat((thermal_features_dc_crop[:, None, :], thermal_features_rest_crop), dim=1)
        rgb_colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        # 将相机的输出分辨率缩放到1/2^d
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)  

        viewmat = get_viewmat(optimized_camera_to_world)    # 这里的 optimized_camera_to_world 是 3x4矩阵
        K = camera.get_intrinsics_matrices().cuda()     # 3x3内参矩阵
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            # 默认是 "classic" 模式
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            # output_depth_during_training: bool = False
            render_mode = "Thermal+ED"
        else:
            render_mode = "Thermal"
        
        render_normal_map = (self.config.predict_normals or not self.training)

        if self.config.sh_degree > 0:
            # sh_degree_interval: int = 1000
            # 每隔sh_degree_interval个step就使sh_degree加一
            # sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            sh_degree_to_use = self.config.sh_degree
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(-2)
            sh_degree_to_use = None

        render, alpha, info = rasterization_thermal(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),   # 实际上不归一化也是可以的
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,  
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,  # 默认是 "classic" 模式
            render_normal_map=render_normal_map,
            vanilla_sh=self.config.use_vanilla_sh
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]   # 一种“通用写法”，表达“保持第一个维度不变，其它维度全保留”，在大多数情况下什么也没改变，主要是为了代码兼容性或统一性

        background = self._get_background_color()
        # 把连续的若干个高斯的 αT 累加起来可以得到这一组高斯的 总alpha，可以继续用于后续的alpha blending中
        rgb = render[:, ..., :3] + (1 - alpha) * background
        # 在这里将rgb值限制到[0.0,1.0]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        # 这里的rgb指的是渲染出来的热红外伪彩色图像

        if render_mode == "Thermal+ED":
            depth_im = render[:, ..., 3:4]
            # 如果某个像素有物体（alpha > 0），就用该像素真实的深度值；
            # 否则拿当前渲染图像的最大深度值，作为“背景默认深度”
            # .detach() 是为了防止这个 max() 影响梯度传播（安全优化）
            # .squeeze(0) 去掉 batch 维，得到 [H, W, 1]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None
        
        normals_im = None
        if render_normal_map:
            normals_im = render[:, ..., 4:7]
            normals_im = normals_im.squeeze(0)
            # gsplat的相机坐标系约定(OpenCV) -> OpenGL的相机坐标系约定
            normals_im = normals_im * torch.tensor([[[1, -1, -1]]], device=normals_im.device, dtype=normals_im.dtype)
            normals_im = (normals_im + 1) / 2   # 线性映射
        

        surface_normal = None
        if self.config.use_normal_loss or not self.training:
            surface_normal = normal_from_depth_image(
                depths=depth_im.detach(),
                fx=camera.fx.item(),
                fy=camera.fy.item(),
                cx=camera.cx.item(),
                cy=camera.cy.item(),
                img_size=(W, H),
                c2w=torch.eye(4, dtype=torch.float, device=depth_im.device),
                device=self.device,
                smooth=False,
            )
            surface_normal = surface_normal @ torch.diag(
                torch.tensor([1, -1, -1], device=depth_im.device, dtype=depth_im.dtype)
            )
            surface_normal = (1 + surface_normal) / 2
        
        rgb0 = None
        if self.config.use_rgb_loss or not self.training:
            sh_degree_to_use = int(rgb_colors_crop.shape[-2] ** 0.5 - 1)
            render, alpha, _ = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),   # 实际上不归一化也是可以的
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=rgb_colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
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
            rasterize_mode=self.config.rasterize_mode,  # 默认是 "classic" 模式
        )
            alpha = alpha[:, ...]
            rgb0 = render[:, ..., :3] + (1 - alpha) * background
            rgb0 = torch.clamp(rgb0, 0.0, 1.0)
            rgb0 = rgb0.squeeze(0)

        
        # 获取相机内参等信息后，重新恢复原始的相机输出分辨率
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "rgb0": rgb0,
            "normal": normals_im,
            "surface_normal": surface_normal,
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # 这里没有用到 metrics_dict
        
        # 当图片没有alpha channel时，composite_with_background会返回原图
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        thermal_loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            # 默认 False
            scale_reg = torch.tensor(0.0).to(self.device)
        
        # 使高斯disc-like
        scale_loss = 0.0
        if self.config.use_scale_loss:
            scale_loss = get_scale_loss(self.scales)

        total_normal_loss = 0.0
        gt_normal = outputs["surface_normal"]
        pred_normal = outputs["normal"]
        if self.config.use_normal_loss:
            total_normal_loss += self.normal_loss(pred_normal, gt_normal) * self.config.normal_lambda
        if self.config.use_normal_tv_loss:
            total_normal_loss += self.normal_tv_loss(pred_normal) * self.config.smooth_loss_lambda

        rgb_loss = 0.0
        if self.config.use_rgb_loss:
            image_rgb = batch["image_rgb"]
            alpha = batch["alpha"]
            alpha = alpha[:, ...]   
            image_rgb = image_rgb + (1 - alpha) * outputs["background"]
            image_rgb = torch.clamp(image_rgb, 0.0, 1.0)
            image_rgb = image_rgb.squeeze(0)
            gt_img_rgb = self.get_gt_img(image_rgb)
            pred_img_rgb = outputs["rgb0"]

            Ll1_rgb = torch.abs(gt_img_rgb - pred_img_rgb).mean()
            simloss_rgb = 1 - self.ssim(gt_img_rgb.permute(2, 0, 1)[None, ...], pred_img_rgb.permute(2, 0, 1)[None, ...])
            rgb_loss = (1 - self.config.ssim_lambda) * Ll1_rgb + self.config.ssim_lambda * simloss_rgb

        loss_dict = {
            "main_loss": thermal_loss + scale_loss + total_normal_loss + rgb_loss,
            "scale_reg": scale_reg, # 默认是0
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        # torch.randn 生成的随机数是从标准高斯分布中采样的
        # 68%的随机数会落在 [-1, 1] 范围内（即在均值的正负一个标准差内）。
        # 95%的随机数会落在 [-2, 2] 范围内（即在均值的正负两个标准差内）。
        # 99.7%的随机数会落在 [-3, 3] 范围内（即在均值的正负三个标准差内）
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            # centered_samples 是为了给偏移向量增加随机微扰
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated

        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        # 将偏移向量旋转至指定方向
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()

        # means等属性都有 @property方法封装，所以可以直接当成属性访问
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # 球谐系数直接复制
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        new_thermal_features_dc = self.thermal_features_dc[split_mask].repeat(samps, 1)
        new_thermal_features_rest = self.thermal_features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        # 不透明度直接复制
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        # 缩放向量进行一定比例的缩小
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        # 旋转四元数直接复制
        new_quats = self.quats[split_mask].repeat(samps, 1)
        
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "thermal_features_dc": new_thermal_features_dc,
            "thermal_features_rest": new_thermal_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                # 其他参数直接复制
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            # 直接复制一组完全一样的高斯
            new_dups[name] = param[dup_mask]
        return new_dups

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        # cull_alpha_thresh: float = 0.1
        # .squeeze()会删除所有形状为1的维度
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        # 调用 .item() 会将 torch.sum(culls) 的单一数值提取为一个 Python 原生的 int 或 float 类型
        # 尝试对一个 非标量张量 使用 .item() 方法，PyTorch 会抛出一个错误
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask

        # if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            # cull_scale_thresh: float = 0.5
            # torch.exp(self.scales).max(dim=-1) 返回的是元组 (values, indices)：
            # 一个形状为 [N] 的张量，包含每一行的最大值（values）
            # 一个形状为 [N] 的张量，包含每一行最大值的列索引（indices）
            # toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            # 绝对大小和相对大小
            # if self.step < self.config.stop_screen_size_at:
                # stop_screen_size_at: int = 4000
                # cull big screen space
                # if self.max_2Dsize is not None:
                    # cull_screen_size: float = 0.15
                    # toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            # culls = culls | toobigs
            # toobigs_count = torch.sum(toobigs).item()
            
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at and not self.config.use_merge_sparsification:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()   # [M]
            grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # [M,2]  norm: [M] 默认是L2范数
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                # self.num_points是一个@property方法，返回高斯数目
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)  # [N]
                # 记录每个点被看见的次数，初始化为全1是为了避免除0错误
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)  # [N]
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads
            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            # 把当前可见点的 radius 投影到屏幕（归一化成 0~1）；
            # 与历史最大值做对比，保留更大的值；
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups() 
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            # 生成一个长度为 d-1 的元组，每个元素都是 1，并拼接成(n, 1, 1, ..., 1)
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            # 一阶动量
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            # 二阶动量
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups() # 字典，value是[param]
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def merge_in_optim(self, optimizer, new_params, n):
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            shape = (n,) + param_state["exp_avg"].shape[1:]
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros(shape, device=param_state["exp_avg"].device, dtype=param_state["exp_avg"].dtype),
                ],
                dim=0,
            )
            shape = (n,) + param_state["exp_avg_sq"].shape[1:]
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros(shape, device=param_state["exp_avg_sq"].device, dtype=param_state["exp_avg_sq"].dtype),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def merge_in_all_optim(self, optimizers, n):
        param_groups = self.get_gaussian_param_groups() 
        for group, param in param_groups.items():
            self.merge_in_optim(optimizers.optimizers[group], param, n)


    def quat_to_rotmat(self, q):
        q = F.normalize(q, dim=-1)
        qw, qx, qy, qz = q.unbind(-1)
        R = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2
        ], dim=-1).reshape(q.shape[:-1] + (3, 3))
        return R

    def rotmat_to_quat(self, R):
        m = R
        t = m[...,0,0] + m[...,1,1] + m[...,2,2]
        qw = torch.sqrt(torch.clamp(1 + t, min=0)) / 2
        qx = torch.sqrt(torch.clamp(1 + m[...,0,0] - m[...,1,1] - m[...,2,2], min=0)) / 2
        qy = torch.sqrt(torch.clamp(1 - m[...,0,0] + m[...,1,1] - m[...,2,2], min=0)) / 2
        qz = torch.sqrt(torch.clamp(1 - m[...,0,0] - m[...,1,1] + m[...,2,2], min=0)) / 2
        qx = torch.copysign(qx, m[...,2,1] - m[...,1,2])
        qy = torch.copysign(qy, m[...,0,2] - m[...,2,0])
        qz = torch.copysign(qz, m[...,1,0] - m[...,0,1])
        q = torch.stack([qw, qx, qy, qz], dim=-1)
        return F.normalize(q, dim=-1)

    # ---------------- 协方差处理 ----------------
    def get_covariance(self, quats, scales):
        R = self.quat_to_rotmat(quats)              # [N,3,3]
        S = torch.diag_embed(scales**2)             # [N,3,3]
        return R @ S @ R.transpose(-1, -2)          # [N,3,3]

    def decompose_covariance(self, Sigma):
        eigvals, eigvecs = torch.linalg.eigh(Sigma)  # [N,3], [N,3,3]
        scales = torch.sqrt(torch.clamp(eigvals, min=1e-8))
        # 行列式为1的正交矩阵才是旋转矩阵，因此需要将eigvecs中所有行列式为-1的正交矩阵修正为 1
        det = torch.det(eigvecs)  # [N]
        eigvecs[det < 0, :, -1] *= -1
        q = self.rotmat_to_quat(eigvecs)
        return q, scales

    def merge_params_blockwise(self, mu_i, mu_j, Sigma_i, Sigma_j, alpha_i, alpha_j, f_i, f_j, block_size=1000000):
        num_pairs = mu_i.shape[0]
        device = mu_i.device

        mu_new_list = []
        Sigma_new_list = []
        alpha_new_list = []
        f_new_list = []

        for start in range(0, num_pairs, block_size):
            end = min(start + block_size, num_pairs)

            mu_i_blk = mu_i[start:end]
            mu_j_blk = mu_j[start:end]
            Sigma_i_blk = Sigma_i[start:end]
            Sigma_j_blk = Sigma_j[start:end]
            alpha_i_blk = alpha_i[start:end]
            alpha_j_blk = alpha_j[start:end]
            f_i_blk = f_i[start:end]
            f_j_blk = f_j[start:end]

            alpha_sum_blk = alpha_i_blk + alpha_j_blk

            # 新均值
            mu_new_blk = (alpha_i_blk[:, None] * mu_i_blk + alpha_j_blk[:, None] * mu_j_blk) / alpha_sum_blk[:, None]

            # 外积（逐元素方式，避免广播成 NxN）
            mu_i_outer = mu_i_blk[:, :, None] * mu_i_blk[:, None, :]
            mu_j_outer = mu_j_blk[:, :, None] * mu_j_blk[:, None, :]

            # 新协方差
            Sigma_new_blk = (alpha_i_blk[:, None, None] * (Sigma_i_blk + mu_i_outer) +
                            alpha_j_blk[:, None, None] * (Sigma_j_blk + mu_j_outer))

            Sigma_new_blk = Sigma_new_blk / alpha_sum_blk[:, None, None] \
                        - mu_new_blk[:, :, None] * mu_new_blk[:, None, :]

            # 新不透明度
            alpha_new_blk = alpha_i_blk + alpha_j_blk - alpha_i_blk * alpha_j_blk

            # 新颜色
            f_new_blk = (alpha_i_blk[:, None] * f_i_blk + alpha_j_blk[:, None] * f_j_blk) / alpha_sum_blk[:, None]

            # 收集结果
            mu_new_list.append(mu_new_blk)
            Sigma_new_list.append(Sigma_new_blk)
            alpha_new_list.append(alpha_new_blk)
            f_new_list.append(f_new_blk)

            # 手动释放中间变量，减少显存压力
            del mu_i_blk, mu_j_blk, Sigma_i_blk, Sigma_j_blk
            del alpha_i_blk, alpha_j_blk, f_i_blk, f_j_blk
            del mu_i_outer, mu_j_outer, mu_new_blk, Sigma_new_blk, alpha_new_blk, f_new_blk
            gc.collect()

        # 拼接分块结果
        mu_new = torch.cat(mu_new_list, dim=0)
        Sigma_new = torch.cat(Sigma_new_list, dim=0)
        alpha_new = torch.cat(alpha_new_list, dim=0)
        f_new = torch.cat(f_new_list, dim=0)

        return mu_new, Sigma_new, alpha_new, f_new

    def merge_gaussians(self, merge_mask, k=5, color_thresh=2.5, dist_thresh=2.38, block_size=80000):
        N = self.means.shape[0]
        device = self.means.device

        means = self.means[merge_mask]                # [M,3]
        quats = self.quats[merge_mask]
        scales = torch.exp(self.scales[merge_mask])
        opacities = torch.sigmoid(self.opacities[merge_mask]).squeeze(-1)
        colors = rgb_to_lab(self.thermal_colors[merge_mask])

        covs = self.get_covariance(quats, scales)     # [M,3,3]
        # inv_covs = torch.linalg.pinv(covs)            # [M,3,3]
        inv_covs = torch.linalg.inv(covs)            # [M,3,3]
        M = means.shape[0]

        knn_idx_list = []
        for start in range(0, M, block_size):
            M1 = min(block_size, M - start)
            end = start + M1
            means_sampled = means[start:end]    # [M1, 3]

            # -------- KNN 搜索 (欧式距离) --------
            euclidean_dists = torch.cdist(means_sampled, means)   # [M1,M]
            # knn_idx = euclidean_dists.topk(k+1, largest=False).indices[:,1:]  存在数值精度问题
            knn_idx = euclidean_dists.topk(k+1, largest=False).indices  # [M1,k+1]
            knn_idx_list.append(knn_idx)

            del euclidean_dists
            gc.collect()

        knn_idx = torch.cat(knn_idx_list, dim=0)    # [M,k+1]

        # -------- Mahalanobis 距离 --------
        means_k = means[knn_idx]    # [M,k+1,3]
        diff = means[:,None,:] - means_k      # [M,k+1,3]

        # d_ij: 使用 j 的协方差
        inv_covs_k = inv_covs[knn_idx] # [M, k+1, 3, 3]
        temp = torch.matmul(diff.unsqueeze(-2), inv_covs_k) # [M, k+1, 1, 3]
        d_ij = torch.matmul(temp, diff.unsqueeze(-1)).squeeze()   # [M, k+1]

        # d_ji: 使用 i 的协方差
        temp = torch.matmul(diff.unsqueeze(-2), inv_covs.unsqueeze(1).repeat(1, k+1, 1, 1)) # [M, k+1, 1, 3]
        d_ji = torch.matmul(temp, diff.unsqueeze(-1)).squeeze() # [M, k+1]

        d_ij = torch.clamp(d_ij, min=1e-8)
        d_ji = torch.clamp(d_ji, min=1e-8)

        # 取对称化距离
        d_m = torch.max(d_ij, d_ji) # [M, k+1]

        # -------- 颜色距离 --------
        colors_k = colors[knn_idx]     # [M, k+1, 3]
        color_diff = torch.norm(colors[:,None,:] - colors_k, dim=-1)  # [M, k+1]

        valid_mask = (d_m < dist_thresh) & (color_diff < color_thresh)  # [M, k+1]

        i_idx, j_idx = torch.where(valid_mask)  # i in [M], j in [k+1]

        if i_idx.numel() == 0:
            return None, torch.zeros(N, dtype=torch.bool, device=device)

        j_idx = knn_idx[i_idx, j_idx]   # j in [M]

        i_idx = i_idx.cpu().numpy()
        j_idx = j_idx.cpu().numpy()
        i_list = []
        j_list = []
        used_idx = np.zeros(max(i_idx.max(), j_idx.max()) + 1, dtype=bool)
        for i in range(0, i_idx.shape[0]):
            if i_idx[i] == j_idx[i]:
                continue
            if used_idx[i_idx[i]] or used_idx[j_idx[i]]:
                continue
            used_idx[i_idx[i]] = used_idx[j_idx[i]] = True
            i_list.append(i_idx[i])
            j_list.append(j_idx[i])

        i_idx = torch.tensor(i_list).cuda()
        j_idx = torch.tensor(j_list).cuda()

        if i_idx.numel() == 0:
            return None, torch.zeros(N, dtype=torch.bool, device=device)

        mu_i, mu_j = means[i_idx], means[j_idx]
        Sigma_i, Sigma_j = covs[i_idx], covs[j_idx]
        alpha_i, alpha_j = opacities[i_idx], opacities[j_idx]
        f_i, f_j = colors[i_idx], colors[j_idx]

        mu_new, Sigma_new, alpha_new, f_new = self.merge_params_blockwise(
            mu_i, mu_j, Sigma_i, Sigma_j, alpha_i, alpha_j, f_i, f_j
        )

        q_new, s_new = self.decompose_covariance(Sigma_new)

        out = {
            "means": mu_new,
            "thermal_features_dc": torch.logit(lab_to_rgb(f_new), eps=1e-10),
            "opacities": torch.logit(alpha_new, eps=1e-10).unsqueeze(1),
            "scales": torch.log(s_new),
            "quats": q_new,
        }

        for name, param in self.gauss_params.items():
            if name not in out:
                # p_i, p_j = param[sampled_idx[i_idx]], param[j_idx]   # [num_pairs,...]
                # weight_i = alpha_i.view(-1, *([1] * (p_i.ndim - 1)))
                # weight_j = alpha_j.view(-1, *([1] * (p_j.ndim - 1)))
                # denom = alpha_sum.view(-1, *([1] * (p_i.ndim - 1)))
                # p_new = (weight_i * p_i + weight_j * p_j) / denom
                # out[name] = p_new
                shape = (mu_new.shape[0],) + param.shape[1:]
                out[name] = torch.zeros(shape, device=param.device, dtype=param.dtype)

        merged_mask = torch.zeros(N + mu_new.shape[0], dtype=torch.bool, device=device)
        idx_map = merge_mask.nonzero(as_tuple=True)[0]  # [M]
        merged_mask[idx_map[i_idx]] = True
        merged_mask[idx_map[j_idx]] = True

        return out, merged_mask

    def refinement_after(self, optimizers: Optimizers, step):
        if self.config.disable_refinement:
            return 

        if self.step <= self.config.warmup_length or self.step > self.config.stop_merge_at:
            # 训练开始阶段不进行refinement
            return

        if self.config.use_merge_sparsification:
            avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
            merges = (avg_grad_norm < self.config.densify_grad_thresh).squeeze()
            merge_params, merges_mask = self.merge_gaussians(merges)

            if merge_params:
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), merge_params[name]], dim=0)
                    )
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(merge_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                self.merge_in_all_optim(optimizers, merge_params["scales"].shape[0])

            deleted_mask = self.cull_gaussians(merges_mask)

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None
            return 


        assert step == self.step
        # self.config.warmup_length 500
        if self.step <= self.config.warmup_length:
            # 训练开始阶段不进行refinement
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset

            # self.config.refine_every 默认是100
            # self.config.reset_alpha_every 默认是30
            # reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                # self.config.stop_split_at 默认是15000
                self.step < self.config.stop_split_at
                # 等数据完整喂完一轮后再进行 densification
                # and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None

                #对于高位置梯度的所有高斯，高于densify_size_thresh的高斯进行split，小于等于densify_size_thresh的高斯进行duplicate
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])    # [N]
                # self.config.densify_grad_thresh默认是0.0008
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()    
                # self.config.densify_size_thresh默认是0.01
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                # self.config.stop_screen_size_at默认是4000
                if self.step < self.config.stop_screen_size_at:
                    # self.config.split_screen_size默认是0.05
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                # 对高位置梯度且size比较大的高斯进行split
                splits &= high_grads
                nsamps = self.config.n_split_samples    # 2
                split_params = self.split_gaussians(splits, nsamps)

                # self.config.densify_size_thresh 0.01
                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                # 对高位置梯度且size比较小的高斯进行duplicate
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)

                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                # self.config.continue_cull_post_densification默认是True
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            # if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
            #     # Reset value is set to be twice of the cull_alpha_thresh
            #     # cull_alpha_thresh: float = 0.1
            #     reset_value = self.config.cull_alpha_thresh * 2.0
            #     # self.opacities.data 是对 self.opacities 张量的 原始数据 的引用
            #     # 通过 data 属性直接访问或修改张量的原始数据，而不触发梯度计算
            #     # 也可使用 torch.no_grad()
            #     self.opacities.data = torch.clamp(
            #         self.opacities.data,
            #         # 先构建一个tensor是为了使用torch.logit,随后使用item变回普通标量
            #         max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
            #     )
            #     # reset the exp of optimizer
            #     optim = optimizers.optimizers["opacities"]
            #     param = optim.param_groups[0]["params"][0]
            #     param_state = optim.state[param]
            #     param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
            #     param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        predicted_rgb0 = outputs["rgb0"]
        predicted_normal = outputs["normal"]
        surface_normal = outputs["surface_normal"]
        predicted_depth = outputs["depth"]
        depth_color = colormaps.apply_depth_colormap(predicted_depth)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb, depth_color, surface_normal, predicted_normal, predicted_rgb0], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict