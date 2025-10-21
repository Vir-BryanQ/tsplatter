import torch
import sys
import numpy as np
import cv2
import os
import random
import time
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gsplat.rendering import rasterization_thermal
import argparse

def slerp(start_rotation, end_rotation, t):
    rotations = R.from_matrix([start_rotation, end_rotation])
    slerp = Slerp([0, 1], rotations)  
    return slerp(t).as_matrix()

def cubic_spline_interpolation(positions, num_frames):
    t = np.linspace(0, 1, positions.shape[0])   # [N]
    spline = CubicSpline(t, positions, axis=0)
    new_t = np.linspace(0, 1, num_frames)   # [C]
    return spline(new_t)

def get_intrinsics_matrix(fx, fy, cx, cy):
    K = torch.zeros((3, 3), dtype=torch.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K

def get_inv(mat):
    R = mat[:, :3, :3]  # 3 x 3
    T = mat[:, :3, 3:4]  # 3 x 1
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    mat_inv = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    mat_inv[:, 3, 3] = 1.0  # homogenous
    mat_inv[:, :3, :3] = R_inv
    mat_inv[:, :3, 3:4] = T_inv
    return mat_inv

def get_fourcc_by_extension(file_extension):
    if file_extension.lower() == 'mp4':
        print('Use mp4v codec.')
        return cv2.VideoWriter_fourcc(*'mp4v')
    elif file_extension.lower() == 'avi':
        print('Use FFV1 codec.')
        return cv2.VideoWriter_fourcc(*'FFV1')
    elif file_extension.lower() == 'mov':
        print('Use jpeg codec.')
        return cv2.VideoWriter_fourcc(*'jpeg')
    elif file_extension.lower() == 'mkv':
        print('Use VP8 codec.')
        return cv2.VideoWriter_fourcc(*'VP80')
    else:
        print('Unsupported video format.')
        sys.exit(-1)

def rescale_output_resolution(
        fx, fy, cx, cy, width, height, new_width,
        scale_rounding_mode: str = "floor",
    ):
    scaling_factor = new_width / width
    new_fx = fx * scaling_factor
    new_fy = fy * scaling_factor
    new_cx = cx * scaling_factor
    new_cy = cy * scaling_factor
    if scale_rounding_mode == "floor":
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)
    elif scale_rounding_mode == "round":
        new_height = int(torch.floor(0.5 + (height * scaling_factor)))
        new_width = int(torch.floor(0.5 + (width * scaling_factor)))
    elif scale_rounding_mode == "ceil":
        new_height = int(torch.ceil(height * scaling_factor))
        new_width = int(torch.ceil(width * scaling_factor))
    else:
        raise ValueError("Scale rounding mode must be 'floor', 'round' or 'ceil'.")
    return (new_fx, new_fy, new_cx, new_cy, new_width, new_height)        

def main(args):
    torch.set_grad_enabled(False)
    dataset, opt, pipe, checkpoint = lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint

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

    viewpoint_stack = scene.getTrainCameras().copy()

    if args.shuffle:
        random.shuffle(viewpoint_stack)

    w2c = torch.stack([vp.world_view_transform.transpose(0, 1) for vp in viewpoint_stack])   # [N,4,4]
    c2w = get_inv(w2c)

    positions = c2w[:, :3, 3].cpu().numpy()   # [N,3]
    rotations = c2w[:, :3, :3].cpu().numpy()  # [N,3,3]

    N = positions.shape[0]
    num_steps = N - 1
    num_frames = ((args.duration * args.fps - 1 + num_steps - 1) // num_steps * num_steps) + 1
    C = num_frames

    new_positions = torch.tensor(cubic_spline_interpolation(positions, num_frames), device='cuda')  # [C,3]

    print('Spherical linear interpolation:')
    new_rotations = [torch.tensor(rotations[0], device='cuda')]
    t_list = np.linspace(0, 1, N)  # [N]
    delta_t = 1 / (num_frames - 1)
    assert (num_frames - 1) % num_steps == 0, f"{num_frames}, {num_steps}"
    num_frames_per_step = (num_frames - 1) // num_steps
    for i in tqdm(range(num_steps)):
        start_rot = rotations[i]
        end_rot = rotations[i + 1]
        t1 = t_list[i]
        t2 = t_list[i + 1]
        t3 = t1 + delta_t
        for j in range(num_frames_per_step):
            t4 = min((t3 - t1) / (t2 - t1), 1)
            interpolated_rot = slerp(start_rot, end_rot, t4)
            new_rotations.append(torch.tensor(interpolated_rot, device='cuda'))
            t3 += delta_t
    new_rotations = torch.stack(new_rotations)  # [C,3,3]

    new_c2w = torch.zeros((C, 4, 4), dtype=torch.float32, device='cuda')    # [C,4,4]
    new_c2w[:, :3, 3] = new_positions
    new_c2w[:, :3, :3] = new_rotations
    new_c2w[:, 3, 3] = 1.0
    viewmats = get_inv(new_c2w)

    width, height = viewpoint_stack[0].image_width, viewpoint_stack[0].image_height
    fx, fy, cx, cy, width, height = rescale_output_resolution(viewpoint_stack[0].fx, viewpoint_stack[0].fy, width / 2, height / 2, width, height, args.width)
    print(f"Video resolution: {width}x{height}")

    K = get_intrinsics_matrix(fx, fy, cx, cy).cuda()
    file_extension = os.path.basename(args.output_path).split('.')[-1]
    fourcc = get_fourcc_by_extension(file_extension)
    video_writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height))

    print(f"sh_degree={gaussians.thermal_sh_degree}")
    print('Rendering images and generating a video:')
    block_size = 60
    for start in tqdm(range(0, C, block_size)):
        end = min(start + block_size, C)
        images = render_images(gaussians, viewmats[start:end], K.repeat(end - start, 1, 1), width, height)
        images *= 255.0
        images.clamp_(min=0, max=255)
        images = images.to(torch.uint8)[..., [2, 1, 0]].cpu().numpy()    # [B,H,W,3]
        for i in tqdm(range(images.shape[0]), leave=False):
            video_writer.write(images[i])

    video_writer.release()
    print(f"Video saved at {args.output_path}")

def render_images(gaussians, viewmats, Ks, W, H):
    if gaussians.thermal_sh_degree > 0:
        sh_degree = gaussians.thermal_sh_degree
    else:
        sh_degree = None
    vanilla_sh = gaussians.thermal_sh_degree > 0
    if vanilla_sh:
        colors = gaussians.get_thermal_features
    else:
        colors = torch.sigmoid(gaussians.get_thermal_features).squeeze(-2)

    render, alpha, _ = rasterization_thermal(
            means=gaussians._xyz,
            quats=gaussians._rotation,   # 实际上不归一化也是可以的
            scales=torch.exp(gaussians._scaling),
            opacities=torch.sigmoid(gaussians._opacity).squeeze(-1),
            colors=colors,
            viewmats=viewmats.cuda(),  # [B, 4, 4]
            Ks=Ks,  # [B, 3, 3]
            width=W,
            height=H,
            tile_size=16,  
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="Thermal",
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",  # 默认是 "classic" 模式
            vanilla_sh=vanilla_sh,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
    alpha = alpha[:, ...]
    background = torch.zeros(3, device=alpha.device)
    rgb = render[:, ..., :3] + (1 - alpha) * background
    rgb = torch.clamp(rgb, 0.0, 1.0)    # [B,H,W,3]
    return rgb
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera interpolation and video generation")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--duration', type=int, required=True, help="Video duration")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the video")
    parser.add_argument('--fps', type=int, required=True, help="Frame rate for video")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument('--width', type=int, required=False, default=640, help="Image width")
    args = parser.parse_args(sys.argv[1:])

    main(args)
