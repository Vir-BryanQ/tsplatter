import torch
from tsplatter import preallocate_vmem
# preallocate_vmem()

import sys
import os
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
import time
from torch_scatter import scatter_max
from random import randint
from gaussian_renderer import count_render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from gsplat.cuda._wrapper import (
    spherical_harmonics2,
)
from render_video import get_inv, get_intrinsics_matrix
from gsplat import rasterization
from tsplatter.ts_model import assign_thermal_colors, unproject_depth_to_world
from smoothing import move_checkpoint_file
from pytorch_msssim import SSIM
from nerfstudio.models.splatfacto import (
    RGB2SH
)

def estimate_lambda_l(M, K, lambda_base=1e-2, alpha=2.0):
    #------------------------------------------
    # Step 1: 根据视角数计算视角置信度
    #------------------------------------------
    # M=1 → confidence=0（噪声大 → 强正则）
    # M→∞ → confidence=1（数据充分 → 弱正则）
    confidence = M / (M + 20.0)   # 平滑函数，可调

    #------------------------------------------
    # Step 2: 定义 λ 的整体缩放因子
    #------------------------------------------
    # 视角越少，lambda_factor 越大
    lambda_factor = (1.0 - confidence) ** 2   # 二次衰减更平滑

    #------------------------------------------
    # Step 3: 为每个阶数生成 λ_l
    #------------------------------------------
    # 只取 m=0 → K = (L+1)
    reg = torch.zeros(K, dtype=torch.float32)
    
    # 阶数 l = 0 ~ K-1
    for l in range(K):
        # λ_l = base * (l**alpha) * lambda_factor
        reg[l] = lambda_base * (l ** alpha) * lambda_factor

    return reg

def build_per_gaussian_reg_mat(vis_masks, K, lambda_base=1e-2, alpha=2.0, min_lambda=1e-6):

    device = vis_masks.device
    M, N = vis_masks.shape

    # 统计每个高斯的可见视角数量 M_n
    M_n = vis_masks.sum(dim=0).float()      # [N]

    # 视角置信度  confidence = M_n / (M_n + 20)
    # 视角少 → confidence 小 → 正则强
    confidence = M_n / (M_n + 20.0)         # [N]

    # 正则缩放因子  lambda_factor = (1 - confidence)^2
    lambda_factor = (1.0 - confidence)**2   # [N]

    # 逐阶生成 λ_l(n)， shape = [N,K]
    l_idx = torch.arange(K, device=device).float()[None, :]   # [1,K]
    # λ_l = lambda_base * l^alpha * lambda_factor
    reg_vals = lambda_base * (l_idx**alpha) * lambda_factor[:, None]   # [N,K]

    # 增加一个极小数避免矩阵奇异
    reg_vals = reg_vals + min_lambda

    # 转成对角矩阵 [N,K,K]
    reg_mat = torch.diag_embed(reg_vals)

    return reg_mat

def dssim_loss(Ax, B, H, W, ssim):
    # reshape 到 [M, 3, H, W]
    Ax = Ax.transpose(1, 2).reshape(-1, 3, H, W)
    B  = B .transpose(1, 2).reshape(-1, 3, H, W)

    # d-SSIM = 1 - SSIM
    dssim = 1 - ssim(B, Ax)

    return dssim

class MatMulAC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, ids_flat, contrib_flat):
        # C: [M,G,3]
        # ids_flat: [M,N,K]
        # contrib_flat: [M,N,K]
        M, N, K = ids_flat.shape

        batch_idx = torch.arange(M, device=C.device)[:, None, None]
        C_sel = C[batch_idx, ids_flat]                      # [M,N,K,3]
        y = (C_sel * contrib_flat[..., None]).sum(dim=2)    # [M,N,3]

        # save tensors for backward
        ctx.save_for_backward(ids_flat, contrib_flat)
        ctx.M, ctx.G = C.shape[0], C.shape[1]
        return y

    @staticmethod
    def backward(ctx, grad_output):
        ids_flat, contrib_flat = ctx.saved_tensors
        M, G = ctx.M, ctx.G
        _, N, K = ids_flat.shape

        # grad_output: [M,N,3]
        weighted = contrib_flat[..., None] * grad_output[:, :, None, :]   # [M,N,K,3]

        grad_C = torch.zeros(M, G, 3, device=grad_output.device)

        index = ids_flat[..., None].expand(-1, -1, -1, 3)                 # [M,N,K,3]
        grad_C.scatter_add_(1, index.reshape(M, -1, 3), weighted.reshape(M, -1, 3))

        return grad_C, None, None


def solve_gaussian_colors_batch(contrib, ids, images, G, initial_colors = None, # [G,3]
                                iters=25, lr=0.1, dssim_lambda=0.2):

    M, H, W, K = contrib.shape
    N = H * W

    ids = ids.long()

    # safe dummy gauss index
    ids[ids == -1] = 0

    # flatten per view
    contrib_flat = contrib.reshape(M, N, K)   # [M, N, K]
    ids_flat     = ids.reshape(M, N, K)       # [M, N, K]
    B            = images.reshape(M, N, 3)    # [M, N, 3]

    if initial_colors is None:
        C = torch.zeros(M, G, 3, device=device, requires_grad=True)
    else:
        C = initial_colors.unsqueeze(0).expand(M, -1, -1).clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([C], lr=lr)

    ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    # for step in tqdm(range(iters)):
    for step in range(iters):
        Ax = MatMulAC.apply(C, ids_flat, contrib_flat)
        loss = ((Ax - B)**2).mean()
        # loss = (1 - dssim_lambda) * ((Ax - B).abs().mean()) + dssim_lambda * dssim_loss(Ax, B, H, W, ssim)

        print(f"step={step}, loss={loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            C.clamp_(0.0, 1.0)


    return C.detach()

def get_initial_colors(viewpoint_stack, gaussians):
    w2c_list = []
    imgs_list = []
    for viewpoint_cam in viewpoint_stack:
        w2c_list.append(viewpoint_cam.world_view_transform.transpose(0, 1))
        imgs_list.append(viewpoint_cam.original_image.permute(2,1,0))
    w2c = torch.stack(w2c_list, dim=0)   # [C,4,4]
    thermal_images = torch.stack(imgs_list, dim=0)    # [C,W,H,3]
    width, height = viewpoint_stack[0].image_width, viewpoint_stack[0].image_height
    fx, fy, cx, cy = viewpoint_stack[0].fx, viewpoint_stack[0].fy, width / 2, height / 2

    colors = gaussians.get_features
    Ks = get_intrinsics_matrix(fx, fy, cx, cy).cuda().repeat(w2c.shape[0], 1, 1)
    sh_degree_to_use = int((colors.shape[-2] ** 0.5) - 1)
    
    depth_im, alpha, _ = rasterization(
            means=gaussians._xyz,
            quats=gaussians._rotation,   # 实际上不归一化也是可以的
            scales=torch.exp(gaussians._scaling),
            opacities=torch.sigmoid(gaussians._opacity).squeeze(-1),
            colors=colors,
            viewmats=w2c, 
            Ks=Ks, 
            width=width,
            height=height,
            tile_size=16,  
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="ED",
            # render_mode="RGB+ED",
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",  # 默认是 "classic" 模式
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

    depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).permute(0, 2, 1, 3)    # [C,W,H,1]

    world_points = unproject_depth_to_world(depth_im=depth_im, Ks=Ks, viewmats=w2c)    # [C,W,H,3]

    # [C, W, H, 3] -> [CWH, 3]
    world_points = world_points.reshape(-1, 3)
    thermal_images = thermal_images.reshape(-1, 3)
    cwh = world_points.shape[0]
    indices = torch.randperm(cwh)[:cwh // 10]
    world_points = world_points[indices]
    thermal_images = thermal_images[indices]

    thermal_colors = assign_thermal_colors(means=gaussians._xyz, thermal_images=thermal_images, world_points=world_points, k=50)  # [N,3]

    return thermal_colors



def solve_shs1(dataset, opt, pipe, checkpoint, args):
    t0 = time.perf_counter()
    gaussians = GaussianModel(dataset.sh_degree)    
    scene = Scene(dataset, gaussians)
    
    ckpt = torch.load(checkpoint, map_location="cpu")
    pipeline_state_dict = ckpt.get("pipeline", {})
    prefix = "_model.gauss_params."
    gauss_params = {
        key[len(prefix):]: value
        for key, value in pipeline_state_dict.items()
        if key.startswith(prefix)
    }
    gaussians.restore(gauss_params, opt)
        
    # 设置背景色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") 

    N = len(gaussians.get_opacity)
    device = gaussians.get_opacity.device
    viewpoint_stack = scene.getTestCameras().copy()

    initial_colors = get_initial_colors(viewpoint_stack, gaussians)

    M = len(viewpoint_stack)
    k = 100
    block_size = 20
    per_gauss_color_list = []
    vis_mask_list = []
    w2c_list = []
    print(f"Use {M} viewpoints.")
    for start in range(0, M, block_size):
        end = min(start + block_size, M)
        ids_list = []
        contribution_list = []
        imgs_list = []
        for i in range(end - start):
            viewpoint_cam = viewpoint_stack[start + i]
            render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
            ids, contribution, vis_mask = (
                # [H,W,100]
                render_pkg['per_pixel_gaussian_ids'].detach(),
                render_pkg['per_pixel_gaussian_contributions'].detach(), 
                render_pkg['visibility_filter'].detach()
            )

            contribution, topk_indices = torch.topk(contribution, k, dim=-1)   # [H,W,k]
            ids = torch.gather(ids, dim=-1, index=topk_indices) # [H,W,k]

            ids_list.append(ids)
            contribution_list.append(contribution)
            imgs_list.append(viewpoint_cam.original_image.permute(1,2,0))
            vis_mask_list.append(vis_mask)
            w2c_list.append(viewpoint_cam.world_view_transform.transpose(0, 1))

        ids_blk = torch.stack(ids_list, dim=0)    # [C, H, W, L]
        contribution_blk = torch.stack(contribution_list, dim=0) # [C, H, W, L]
        imgs_blk = torch.stack(imgs_list, dim=0) # [C, H, W, 3]

        per_gauss_color_blk = solve_gaussian_colors_batch(contribution_blk.detach(), ids_blk.detach(), imgs_blk.detach(), N, 
                                                            initial_colors=initial_colors.detach())   # [C,N,3]
        per_gauss_color_list.append(per_gauss_color_blk)

    per_gauss_color = torch.cat(per_gauss_color_list, dim=0)  # [M,N,3]
    per_gauss_color = per_gauss_color.permute(1,0,2)    # [N,M,3]
    per_gauss_color -= 0.5  
    # per_gauss_color.clamp_(min=-0.5, max=0.5)
    vis_masks = torch.stack(vis_mask_list, dim=0)   # [M,N]
    vis_count = vis_masks.sum(dim=0)      # [N]
    gauss_mask = vis_count > 0

    means = gaussians._xyz[gauss_mask]   # [N,3]
    scales = gaussians._scaling[gauss_mask]  # [N,3]
    quats = gaussians._rotation[gauss_mask]  # [N,4]
    per_gauss_color = per_gauss_color[gauss_mask]
    vis_masks = vis_masks[:, gauss_mask]

    w2c = torch.stack(w2c_list, dim=0)   # [M,4,4]
    c2w = get_inv(w2c)

    viewdirs = c2w[:, None, :3, 3] - means[None, :, :]  # [M,N,3]
    viewdirs = F.normalize(viewdirs, dim=-1)

    quats = quats / quats.norm(dim=-1, keepdim=True)
    normals = F.one_hot(
        torch.argmin(scales, dim=-1), num_classes=3
    ).float()   # [N,3]
    rots = quat_to_rotmat(quats)    # [N,3,3]
    normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)  # [N,3]
    normals = F.normalize(normals, dim=-1)

    dots = (normals * viewdirs).sum(-1, keepdim=True) # [M,N,1]
    abs_dots = dots.abs().detach()  # [M,N,1]
    K = gaussians._thermal_features_rest.shape[1]
    thermal_sh_degree = K
    func_vals = spherical_harmonics2(
        thermal_sh_degree, abs_dots, vis_masks  
    )  # [M,N,K]
    
    func_vals = func_vals.permute(1,0,2)    # [N,M,K]
    # func_vals = func_vals[:, :, 1:]     # [N,M,K-1]
    # func_vals[:, :, 1:] *= 20

    N, M, K = func_vals.shape
    reg_mat = build_per_gaussian_reg_mat(vis_masks, K)    # [N, K, K]

    ftf = func_vals.transpose(1,2) @ func_vals               # [N,K,K]
    ftc = func_vals.transpose(1,2) @ per_gauss_color         # [N,K,3]

    lhs = ftf + reg_mat
    rhs = ftc

    try:
        thermal_features = torch.linalg.solve(lhs, rhs) # [N,K,3]
    except torch.linalg.LinAlgError:
        print('Encounter a singular matrix.')
        thermal_features = torch.linalg.lstsq(lhs, rhs).solution
    th_feats = torch.zeros_like(gaussians.get_thermal_features) 
    th_feats[:, 0, :] = RGB2SH(initial_colors)
    th_feats[gauss_mask] = thermal_features
    
    # thermal_features = torch.linalg.pinv(func_vals) @ per_gauss_color   # [N,K,3]
    # thermal_features_rest = torch.linalg.pinv(func_vals) @ per_gauss_color   # [N,K-1,3]

    # print(thermal_features[:, 0])
    # print(ckpt["pipeline"]["_model.gauss_params.thermal_features_dc"])
    # sys.exit(0)

    # thermal_features_dc = per_gauss_color.mean(dim=1)
    # thermal_features_dc -= 0.5
    # thermal_features_dc /= 0.2820947917738781

    ckpt["pipeline"]["_model.gauss_params.thermal_features_dc"] = th_feats[:, 0].cpu()
    ckpt["pipeline"]["_model.gauss_params.thermal_features_rest"] = th_feats[:, 1:].cpu()
    # ckpt["pipeline"]["_model.gauss_params.thermal_features_dc"] = thermal_features_dc.cpu()
    os.remove(checkpoint)
    torch.save(ckpt, checkpoint)

    t1 = time.perf_counter()
    print(f"Total: {(t1 - t0) * 1000:.3f} ms")


def solve_shs(dataset, opt, pipe, checkpoint, args):
    t0 = time.perf_counter()

    gaussians = GaussianModel(dataset.sh_degree)    # 简单地给所有属性赋空值
    scene = Scene(dataset, gaussians)
    
    ckpt = torch.load(checkpoint, map_location="cpu")
    pipeline_state_dict = ckpt.get("pipeline", {})
    prefix = "_model.gauss_params."
    gauss_params = {
        key[len(prefix):]: value
        for key, value in pipeline_state_dict.items()
        if key.startswith(prefix)
    }
    gaussians.restore(gauss_params, opt)
        
    # 设置背景色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") 

    N = len(gaussians.get_opacity)
    device = gaussians.get_opacity.device
    opacities = torch.sigmoid(gaussians._opacity)   # [N,1]
    
    viewpoint_stack = scene.getTestCameras().copy()
    M = len(viewpoint_stack)
    block_size = 5
    per_gauss_color_list = []
    vis_mask_list = []
    print(f"Use {M} viewpoints.")
    for start in range(0, M, block_size):
        end = min(start + block_size, M)
        ids_list = []
        contribution_list = []
        imgs_list = []
        for i in range(end - start):
            viewpoint_cam = viewpoint_stack[start + i]
            render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
            ids, contribution, vis_mask = (
                # [H,W,100]
                render_pkg['per_pixel_gaussian_ids'].detach(),
                render_pkg['per_pixel_gaussian_contributions'].detach(), 
                render_pkg['visibility_filter'].detach()
            )
            ids_list.append(ids)
            contribution_list.append(contribution)
            imgs_list.append(viewpoint_cam.original_image.permute(1,2,0))
            vis_mask_list.append(vis_mask)
        ids_blk = torch.stack(ids_list, dim=0)    # [C, H, W, L]
        contribution_blk = torch.stack(contribution_list, dim=0) # [C, H, W, L]
        imgs_blk = torch.stack(imgs_list, dim=0) # [C, H, W, 3]
        C, H, W, L = ids_blk.shape
        weighted_colors = contribution_blk[..., None] * imgs_blk[..., None, :]  # [C, H, W, L, 3]
        # weighted_colors = imgs_blk.unsqueeze(3).expand(C, H, W, L, 3)   # [C, H, W, L, 3]
        weighted_colors = weighted_colors.reshape(-1, 3) # [C*H*W*L, 3]
        ids_blk = ids_blk.reshape(-1) # [C*H*W*L]
        contribution_blk = contribution_blk.reshape(-1) # [C*H*W*L]
        valid_mask = (ids_blk != -1)    # [C*H*W*L]
        weighted_colors = weighted_colors[valid_mask]   # [B,3]
        weighted_colors = weighted_colors.reshape(-1) # [B*3]
        contribution_blk = contribution_blk[valid_mask] # [B]
        ids_blk = ids_blk[valid_mask]   # [B]
        ids_blk = ids_blk.unsqueeze(-1).repeat(1,3).reshape(-1) # [B*3]
        contribution_blk = contribution_blk.unsqueeze(-1).repeat(1,3).reshape(-1)   # [B*3]

        view_ids = torch.tensor(range(C), device=device, dtype=torch.int).unsqueeze(-1).repeat(1,H*W*L*3).reshape(C*H*W*L, 3)   # [C*H*W*L, 3]
        view_ids = view_ids[valid_mask] # [B,3]
        view_ids = view_ids.reshape(-1) # [B*3]

        color_ids = torch.tensor(range(3), device=device, dtype=torch.int).repeat(C*H*W*L).reshape(C*H*W*L, 3)   # [C*H*W*L, 3]
        color_ids = color_ids[valid_mask] # [B,3]
        color_ids = color_ids.reshape(-1) # [B*3]

        per_gauss_color_blk = torch.zeros((C, N, 3), device=device, dtype=torch.float32)
        per_gauss_color_blk.index_put_((view_ids, ids_blk, color_ids), weighted_colors, accumulate=True)

        weights_sum = torch.zeros((C, N), device=device, dtype=torch.float32)
        weights_sum.index_put_((view_ids, ids_blk), contribution_blk, accumulate=True)

        per_gauss_color_blk /= (weights_sum[..., None] + 1e-9)   # [C,N,3]
        # per_gauss_color_blk /= opacities[None, :, :]

        per_gauss_color_list.append(per_gauss_color_blk)

        del color_ids, weighted_colors, valid_mask, imgs_blk, ids_blk, contribution_blk, view_ids, weights_sum

    per_gauss_color = torch.cat(per_gauss_color_list, dim=0)  # [M,N,3]
    per_gauss_color = per_gauss_color.permute(1,0,2)    # [N,M,3]
    per_gauss_color -= 0.5  
    # per_gauss_color.clamp_(min=-0.5, max=0.5)
    vis_masks = torch.stack(vis_mask_list, dim=0)   # [M,N]
    means = gaussians._xyz   # [N,3]
    scales = gaussians._scaling  # [N,3]
    quats = gaussians._rotation  # [N,4]

    w2c = torch.stack([vp.world_view_transform.transpose(0, 1) for vp in viewpoint_stack])   # [M,4,4]
    c2w = get_inv(w2c)

    viewdirs = c2w[:, None, :3, 3] - means[None, :, :]  # [M,N,3]
    viewdirs = F.normalize(viewdirs, dim=-1)

    quats = quats / quats.norm(dim=-1, keepdim=True)
    normals = F.one_hot(
        torch.argmin(scales, dim=-1), num_classes=3
    ).float()   # [N,3]
    rots = quat_to_rotmat(quats)    # [N,3,3]
    normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)  # [N,3]
    normals = F.normalize(normals, dim=-1)

    dots = (normals * viewdirs).sum(-1, keepdim=True) # [M,N,1]
    abs_dots = dots.abs().detach()  # [M,N,1]
    K = gaussians._thermal_features_rest.shape[1]
    thermal_sh_degree = K
    func_vals = spherical_harmonics2(
        thermal_sh_degree, abs_dots, vis_masks  
    )  # [M, N, K]
    
    func_vals = func_vals.permute(1,0,2)    # [N,M,K]

    # thermal_features = torch.linalg.lstsq(func_vals, per_gauss_color).solution  # [N,K,3]
    thermal_features = torch.linalg.pinv(func_vals) @ per_gauss_color   # [N,K,3]

    ckpt["pipeline"]["_model.gauss_params.thermal_features_dc"] = thermal_features[:, 0].cpu()
    ckpt["pipeline"]["_model.gauss_params.thermal_features_rest"] = thermal_features[:, 1:].cpu()
    os.remove(checkpoint)
    torch.save(ckpt, checkpoint)

    t1 = time.perf_counter()
    print(f"Total: {(t1 - t0) * 1000:.3f} ms")
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) 
    pp = PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    # torch.set_grad_enabled(False)

    move_checkpoint_file(args, 'origin1')

    solve_shs1(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args)

    print("\nSolving complete.")

