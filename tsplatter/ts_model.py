import sys
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from concurrent.futures import ThreadPoolExecutor
from rich.progress import track

import torch
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
from tsplatter.data.ts_dataset import TSDataset

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

@dataclass
class TSplatterModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: TSplatterModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
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
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
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
        sh_degree_to_use = self.config.sh_degree
        means = dataset.metadata['means'].cuda()
        quats = dataset.metadata['quats'].cuda()
        scales = dataset.metadata['scales'].cuda()
        opacities = dataset.metadata['opacities'].cuda()
        features_dc = dataset.metadata['features_dc'].cuda()
        features_rest = dataset.metadata['features_rest'].cuda()
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)

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

        dim_sh = num_sh_bases1(self.config.sh_degree)

        thermal_colors = assign_thermal_colors(means=means, thermal_images=thermal_images, world_points=world_points, k=3)

        shs = torch.zeros((means.shape[0], dim_sh, 3)).float().cuda()
        shs[:, 0, :3] = RGB2SH(thermal_colors)
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
        else:
            # 训练时不裁剪
            opacities_crop = self.opacities
            means_crop = self.means
            thermal_features_dc_crop = self.thermal_features_dc
            thermal_features_rest_crop = self.thermal_features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((thermal_features_dc_crop[:, None, :], thermal_features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        # 将相机的输出分辨率缩放到1/2^d
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)  

        viewmat = get_viewmat(optimized_camera_to_world)    # 这里的 optimized_camera_to_world 是 3x4矩阵
        K = camera.get_intrinsics_matrices().cuda()     # 3x3内参矩阵
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        # 获取相机内参矩阵等信息后，重新恢复原始的相机输出分辨率
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            # 默认是 "classic" 模式
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            # output_depth_during_training: bool = False
            render_mode = "Thermal+ED"
        else:
            render_mode = "Thermal"

        if self.config.sh_degree > 0:
            # sh_degree_interval: int = 1000
            # 每隔sh_degree_interval个step就使sh_degree加一
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop)
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

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
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

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
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