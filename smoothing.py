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
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
import torch
from torch_scatter import scatter_max
import torch.nn.functional as F
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np
import faiss
from collections import deque
import gc

# from autoencoder.model import Autoencoder

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def update_voting_mat(result_dict, language_feature_mask, gt_language_feature, contribution, ids, args):
    # Select only locations where Mask is True
    mask_idx = language_feature_mask.squeeze(0).nonzero(as_tuple=True)
    
    # Get the ID and contributions of the gaussians who contributed from that location
    contrib = contribution[mask_idx]  # shape: [N, 100]
    ray_ids = ids[mask_idx]  # shape: [N, 100]
    gt_feats = gt_language_feature[:, mask_idx[0], mask_idx[1]]  # shape: [3, N]
    
    _, indices = torch.topk(contrib, args.topk, dim=1)
    ray_ids = torch.gather(ray_ids, 1, indices)
    
    # Filter only valid contributions (non-1 IDs and non-0 contributions)
    valid_mask = (ray_ids != -1)
    ray_ids = ray_ids[valid_mask].view(-1)  # shape: [M] (valid Gaussian ID)
    gt_feats = gt_feats.T.unsqueeze(1).repeat(1, args.topk, 1)[valid_mask]  # shape: [M, 3]

    unique_ids = torch.unique(ray_ids)
    
    for uid in unique_ids:
        mask = ray_ids == uid
        if uid.item() not in result_dict:
            result_dict[uid.item()] = [gt_feats[mask]]
        else:
            result_dict[uid.item()].append(gt_feats[mask])

    return result_dict

def compute_average(features):
    averaged_tensor = features.mean(dim=0).unsqueeze(0)  # 평균 계산
    averaged_tensor = averaged_tensor / (averaged_tensor.norm(dim=-1, keepdim=True) + 1e-9)
    return averaged_tensor


def majority_voting(gaussians, scene, pipe, background, dataset, args):
    # 这里 *list 的意思是把列表里的元素依次传入函数
    lf_path = "/" + os.path.join(*dataset.lf_path.split('/')[:-1], "language_features")
    # if args.use_pq:
    #     voting_mat = -1 * torch.ones((gaussians._opacity.shape[0], 17), dtype=torch.uint8, device="cuda")
    # else:
        # 创建一个形状为[N, 3]的全 0 GPU 张量
        # voting_mat = -1 * torch.zeros((gaussians._opacity.shape[0], 3), dtype=torch.float32, device="cuda")
    # viewpoint_stack是Camera对象构成的list
    viewpoint_stack = scene.getTrainCameras().copy()
    
    # defaultdict 是一种特殊的字典，和普通的 dict 类似，但当访问不存在的键时，会自动创建一个默认值
    # 创建了一个 defaultdict，它的默认工厂函数是 list
    # 意味着：如果访问一个不存在的键，会自动分配一个新的空列表 []
    # from collections import defaultdict
    # result_dict = defaultdict(list)
    
    #### code edit ####
    # 对feature进行计数
    num_masks_array = torch.zeros(len(viewpoint_stack), dtype=torch.int, device=gaussians.get_opacity.device)
    for i in range(len(viewpoint_stack)):
        language_feature_name = os.path.join(lf_path, viewpoint_stack[i].image_name)
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))   # [B,D]
        num_masks_array[i] = feature_map.shape[0]   # 每张图像的feature数

    num_masks = torch.sum(num_masks_array)  # 所有图像的feature总数 M
    num_gaussians = len(gaussians.get_opacity)
    features_array = torch.zeros((num_masks,512), device=gaussians.get_opacity.device)   # [M,512]
    allocate_array = torch.zeros((num_gaussians, num_masks), dtype=torch.float32, device=gaussians.get_opacity.device)   # [N,M]
    offset = 0
    for i in tqdm(range(len(viewpoint_stack))):
        viewpoint_cam = viewpoint_stack[i]
        language_feature_name = os.path.join(lf_path, viewpoint_cam.image_name)
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))   # [B,D]
        features_array[offset:offset+num_masks_array[i]] = feature_map  # 所有图像的feature
        
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        ids, contribution = (
            # [H,W,100]
            render_pkg['per_pixel_gaussian_ids'].detach(),
            render_pkg['per_pixel_gaussian_contributions'].detach(), 
        )

        # self._feature_level = -1
        # ParamGroup中对下划线进行了处理
        # # 0:default 1:s 2:m 3:l
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy')).type(torch.int64)[dataset.feature_level].unsqueeze(0).cuda() # [1,H,W]
        seg_map_bool = seg_map != -1    # bool [1,H,W]
        seg_map += offset   # 保证seg_map中的索引与features_array一致

        # 返回一个包含所有 True 值的位置的元组。这个元组包含两个张量：一个是行索引，另一个是列索引，指示布尔张量中 True 值的位置
        mask_idx = seg_map_bool.squeeze(0).nonzero(as_tuple=True)
        
        # Get the ID and contributions of the gaussians who contributed from that location
        contrib = contribution[mask_idx]  # shape: [num_true_elements, 100]
        ray_ids = ids[mask_idx]  # shape: [num_true_elements, 100]

        # topk默认是1
        # Top-K
        gt_segmentations = seg_map[0, mask_idx[0], mask_idx[1]] # 取出所有满足掩码条件的位置上的索引值，形状为[num_true_elements]
        gt_segmentations = gt_segmentations.repeat(args.topk,1).T.reshape(-1)   # [num_true_elements] -> [k,num_true_elements] -> [num_true_elements,k] -> [num_true_elements*k]

        # weights和indices的形状相同
        weights, indices = torch.topk(contrib, args.topk, dim=1)
        # torch.gather(input, dim, index, out=None) 用于根据指定索引从输入张量中选取元素
        # input 张量和 index 张量的形状不需要完全相同，但它们必须满足特定的形状兼容性要求
        # 除指定 dim 维度外，input 和 index 必须具有相同的形状
        ray_ids = torch.gather(ray_ids, 1, indices)

        weights = weights.reshape(-1)
        ray_ids = ray_ids.reshape(-1)
        valid_mask = (ray_ids != -1)    # -1索引为无效索引
        ray_ids = ray_ids[valid_mask]
        weights = weights[valid_mask]
        gt_segmentations = gt_segmentations[valid_mask]

        ray_ids = ray_ids.type(torch.int64)

        # weight_sum = torch.zeros(num_gaussians)
        # 以下划线结尾代表原地操作
        # tensor.index_put_(indices, values, accumulate=False)用于通过索引将新的值填充到原始张量中的指定位置
        # indices：一个包含索引的元组，指定了在哪些位置进行赋值操作。它通常是一个包含多个张量的元组，每个张量代表一个维度上的索引
        # 一个张量，包含你想要放置在 indices 指定位置的值。这个张量的形状必须与 indices 所指定的位置一致
        # accumulate：一个布尔值，表示是否对指定位置的元素进行累加，False的话就直接赋值
        allocate_array.index_put_((ray_ids, gt_segmentations), weights, accumulate=True)

        offset += num_masks_array[i]

    # 将feature vector归一化
    features_array /= (features_array.norm(dim=-1, keepdim=True) + 1e-9)

    weight_sum = torch.sum(allocate_array, 1)
    threshold = 1e-4
    weight_sum_over_zero = weight_sum>0
    weight_sum_under_threshold = weight_sum<threshold
    reweight_index = weight_sum_over_zero * weight_sum_under_threshold
    # 先选择 allocate_array 中对应 reweight_index 为 True 的元素，然后从这些元素中选择大于零的元素，最后将这些大于零的元素的值设置为 1
    allocate_array[reweight_index][allocate_array[reweight_index]>0] = 1

    # 对所有feature进行加权求和
    averaged_tensor = torch.matmul(allocate_array.type(torch.float32) ,features_array)
    averaged_tensor /= (averaged_tensor.norm(dim=-1, keepdim=True) + 1e-9)  # 归一化


    # if args.use_pq:
    #     index = faiss.read_index(args.pq_index)

    #     weight_sum = torch.sum(allocate_array, 1)
    #     threshold = 1e-4
    #     weight_sum_over_zero = weight_sum>0
    #     weight_sum_under_threshold = weight_sum<threshold
    #     reweight_index = weight_sum_over_zero * weight_sum_under_threshold
    #     allocate_array[reweight_index][allocate_array[reweight_index]>0] = 1

    #     averaged_tensor = torch.matmul(allocate_array.type(torch.float32) ,features_array)
    #     averaged_tensor /= (averaged_tensor.norm(dim=-1, keepdim=True) + 1e-9)
    #     invalid_gaussians = torch.sum(averaged_tensor,1) == 0


    #     if args.faiss_add: index.add(averaged_tensor.cpu().numpy())
    #     averaged_tensor = index.sa_encode(averaged_tensor.cpu().numpy())
    #     averaged_tensor = torch.ByteTensor(averaged_tensor).to("cuda")
    #     averaged_tensor[invalid_gaussians,:] = -1
        
    return averaged_tensor

def cosine_similarity_clustering(features, threshold=0.9, block_size=50000):

    N = features.shape[0]
    device = features.device
    free_mask = torch.ones(N, dtype=torch.bool, device=device) # [N]
    cluster_ids = -1 * torch.ones(N, dtype=torch.int, device=device)    # [N]
    cluster_id = 0
    for start in range(0, N, block_size):
        N1 = min(block_size, N - start)
        end = start + N1
        features_sampled = features[start:end]    # [N1, D]
        sim_mask = (features_sampled @ features.T) > threshold   # [N1, N]
        s = sim_mask
        while s.shape[0] > 0:
            grp_mask = s[0] & free_mask     # [N]
            cluster_ids[grp_mask] = cluster_id
            free_mask[grp_mask] = False
            cluster_id += 1
            s = sim_mask[free_mask[start:end]]
        del sim_mask
        gc.collect()

    return cluster_ids
        
def compute_significant_mask(contribution, ids, N, max_threshold=0.01, block_size=10000, use_max_weight=True, sum_threshold=0.25):
    device = contribution.device
    if use_max_weight:
        per_gauss_contrib = torch.zeros(N, device=device, dtype=contribution.dtype)  # [N]
        contribution = contribution.reshape(-1)   # [M*H*W*L]
        ids = ids.reshape(-1)    # [M*H*W*L]
        valid_mask = (ids != -1)    # [M*H*W*L]
        contribution = contribution[valid_mask]
        ids = ids[valid_mask]
        out = scatter_max(contribution, ids.type(torch.long))[0]
        per_gauss_contrib[:out.shape[0]] = out
        significant_mask = per_gauss_contrib > max_threshold  # [N]
    else:
        per_gauss_contrib = torch.zeros(N, device=device, dtype=contribution.dtype)  # [N]
        contribution = contribution.reshape(-1)   # [M*H*W*L]
        ids = ids.reshape(-1)    # [M*H*W*L]
        valid_mask = (ids != -1)    # [M*H*W*L]
        contribution = contribution[valid_mask]
        ids = ids[valid_mask]
        per_gauss_contrib.index_put_((ids,), contribution, accumulate=True)
        significant_mask = per_gauss_contrib > sum_threshold  # [N]
    
    return significant_mask

def laplacian_smoothing(gaussians, cluster_ids, full_significant_mask, lambda_reg=0.5, k_neighbors=5):
    N = gaussians.get_thermal_features.shape[0]
    device = gaussians.get_thermal_features.device
    thermal_features = gaussians.get_thermal_features.clone()   # [N,K,3]

    unique_cluster_ids = torch.unique(cluster_ids).cpu()
    for cluster_id in unique_cluster_ids:
        indices = torch.where(cluster_ids == cluster_id)[0] # [M]
        means = gaussians.get_xyz[indices]  # [M]
        M = means.shape[0]
        k = min(M - 1, k_neighbors)

        euclidean_dists = torch.cdist(means, means)   # [M,M]
        knn_idx = euclidean_dists.topk(k + 1, largest=False).indices   # [M,k+1]
        weights = 1 / (euclidean_dists + 1e-9)  # [M,M]  
        del euclidean_dists
        gc.collect()

        graph_mask = torch.zeros((M, M), device=device, dtype=torch.bool)     # [M,M]
        graph_mask.scatter_(1, knn_idx, True)
        graph_mask = graph_mask | graph_mask.T  # [M,M]

        graph_mask.fill_diagonal_(0)
        adj_matrix = torch.zeros((M, M), device=device, dtype=torch.float32)     # [M,M]
        adj_matrix[graph_mask] = weights[graph_mask]
        del weights, graph_mask
        gc.collect()

        degree_matrix = torch.diag(adj_matrix.sum(dim=1))
        laplacian_matrix = degree_matrix - adj_matrix
        del degree_matrix, adj_matrix
        gc.collect()

        significant_mask = full_significant_mask[indices]    # [M]

        non_significant_mask = ~significant_mask    # [M]
        L_ii = laplacian_matrix[non_significant_mask, :][:, non_significant_mask]  # [M2,M2]
        L_ij = laplacian_matrix[non_significant_mask, :][:, significant_mask]  # [M2,M1]
        del laplacian_matrix
        gc.collect()

        known_features = thermal_features[indices][significant_mask]  # [M1,K,3]
        M1 = known_features.shape[0]
        if M1 == 0:
            print('Warning: there is no known feature in a group')
            continue
        M2 = M - M1
        K = known_features.shape[1]
        known_features = known_features.reshape(M1, -1) # [M1,K*3]

        # 对于非显著节点，使用图拉普拉斯平滑
        # 目标是解决方程: L_ii * unknown_features = -L_ij * known_features
        rhs = -torch.mm(L_ij, known_features)  # [M2,K*3]
        lhs = L_ii + lambda_reg * torch.eye(L_ii.shape[0], device=device)   # [M2,M2]

        try:
            smooth_unknown_features = torch.linalg.solve(lhs, rhs).reshape(M2, K, 3)    # [M2,K*3] -> [M2,K,3]
        except torch.linalg.LinAlgError:
            print('Warning: the input matrix is singular')
            smooth_unknown_features = torch.linalg.lstsq(lhs, rhs)[0].reshape(M2, K, 3)

        del L_ii, L_ij
        gc.collect()

        thermal_features[indices][non_significant_mask] = smooth_unknown_features

    sye.exit(0)
    return thermal_features


def temperature_propagation(gaussians, scene, pipe, background, dataset, args):
    num_gaussians = len(gaussians.get_opacity)
    viewpoint_stack = scene.getTestCameras().copy()
    ids_list = []
    contribution_list = []
    for i in range(len(viewpoint_stack)):
        viewpoint_cam = viewpoint_stack[i]
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        ids, contribution = (
            # [H,W,100]
            render_pkg['per_pixel_gaussian_ids'].detach(),
            render_pkg['per_pixel_gaussian_contributions'].detach(), 
        )
        ids_list.append(ids)
        contribution_list.append(contribution)
    ids_stack = torch.stack(ids_list, dim=0)    # [M,H,W,100]
    contribution_stack = torch.stack(contribution_list, dim=0) # [M,H,W,100]

    cluster_ids = cosine_similarity_clustering(gaussians.get_language_feature)
    significant_mask = compute_significant_mask(contribution_stack, ids_stack, num_gaussians)
    thermal_features = laplacian_smoothing(gaussians, cluster_ids, significant_mask)

    print('temperature_propagation finished')

    return thermal_features
    
# training(lp.extract(args), op.extract(args), pp.extract(args), 
# args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
def smoothing(dataset, opt, pipe, checkpoint, args):
    gaussians = GaussianModel(dataset.sh_degree)    # 简单地给所有属性赋空值
    scene = Scene(dataset, gaussians)
    
    # gaussians.training_setup(opt)

    if not checkpoint:
        raise ValueError("checkpoint missing!!!!!")
    # if opt.include_feature:
    #     if not checkpoint:
    #         raise ValueError("checkpoint missing!!!!!")

    # 若torch.save保存的是元组，那么torch.load返回的也是元组
    # (model_params, first_iter) 与 (gaussians.capture(opt.include_feature), iteration) 相对应
    # 这里的model_params是元组，first_iter后面用不到
    # (model_params, first_iter) = torch.load(checkpoint)
    # if len(model_params) == 12 and opt.include_feature:
    #     first_iter = 0
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
    
    maj_feat = majority_voting(gaussians, scene, pipe, background, dataset, args)
    # 删除feature为零向量的高斯
    gaussians_mask = maj_feat.norm(dim=-1) > 0  # [N]
    gaussians._language_feature = maj_feat[gaussians_mask]
    gaussians._xyz = gaussians._xyz[gaussians_mask]
    gaussians._features_dc = gaussians._features_dc[gaussians_mask]
    gaussians._features_rest = gaussians._features_rest[gaussians_mask]
    gaussians._thermal_features_dc = gaussians._thermal_features_dc[gaussians_mask]
    gaussians._thermal_features_rest = gaussians._thermal_features_rest[gaussians_mask]
    gaussians._scaling = gaussians._scaling[gaussians_mask]
    gaussians._rotation = gaussians._rotation[gaussians_mask]
    gaussians._opacity = gaussians._opacity[gaussians_mask]

    thermal_features = temperature_propagation(gaussians, scene, pipe, background, dataset, args)
    gaussians._thermal_features_dc = thermal_features[:, 0, :]
    gaussians._thermal_features_rest = thermal_features[:, 1:, :]

    gaussians_mask = gaussians_mask.cpu()
    ckpt["pipeline"]["_model.gauss_params.thermal_features_dc"] = gaussians._thermal_features_dc.cpu()
    ckpt["pipeline"]["_model.gauss_params.thermal_features_rest"] = gaussians._thermal_features_rest.cpu()
    ckpt["pipeline"]["_model.gauss_params.means"] = ckpt["pipeline"]["_model.gauss_params.means"][gaussians_mask]
    ckpt["pipeline"]["_model.gauss_params.scales"] = ckpt["pipeline"]["_model.gauss_params.scales"][gaussians_mask]
    ckpt["pipeline"]["_model.gauss_params.quats"] = ckpt["pipeline"]["_model.gauss_params.quats"][gaussians_mask]
    ckpt["pipeline"]["_model.gauss_params.opacities"] = ckpt["pipeline"]["_model.gauss_params.opacities"][gaussians_mask]
    ckpt["pipeline"]["_model.gauss_params.features_dc"] = ckpt["pipeline"]["_model.gauss_params.features_dc"][gaussians_mask]
    ckpt["pipeline"]["_model.gauss_params.features_rest"] = ckpt["pipeline"]["_model.gauss_params.features_rest"][gaussians_mask]
    base_name, ext = os.path.splitext(checkpoint)
    new_checkpoint = base_name + "_origin" + ext
    os.rename(checkpoint, new_checkpoint)
    torch.save(ckpt, checkpoint)
    
    # iteration = 0

    # default=[0]
    # if (iteration in saving_iterations):
    #     print("\n[ITER {}] Saving Gaussians".format(iteration))
    #     scene.save(iteration)

    # default=[0]
    # if (iteration in checkpoint_iterations):
    #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
    #     # capture方法返回一个包含所有高斯属性的tuple
    #     # torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    #     ckpt["pipeline"]["_model.gauss_params.thermal_features_dc"] = gaussians._thermal_features_dc
    #     ckpt["pipeline"]["_model.gauss_params.thermal_features_rest"] = gaussians._thermal_features_rest
    #     base_name, ext = os.path.splitext(checkpoint)
    #     new_checkpoint = base_name + "_smooth" + ext
    #     torch.save(ckpt, new_checkpoint)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, testing_iterations, scene : Scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'testing for iter {iteration}')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) # self.include_feature
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    # parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[0])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # parser.add_argument("--name_extra", type=str, default = None)
    # parser.add_argument("--mode", type=str, default = "mean")
    parser.add_argument("--topk", type=int, default = 1)
    
    # parser.add_argument("--use_pq", action="store_true")
    # parser.add_argument("--pq_index", type=str, default=None)
    
    # parser.add_argument('--encoder_dims',
    #                     nargs = '+',
    #                     type=int,
    #                     default=[256, 128, 64, 32, 3],
    #                     )
    # parser.add_argument('--decoder_dims',
    #                     nargs = '+',
    #                     type=int,
    #                     default=[16, 32, 64, 128, 256, 256, 512],
    #                     )
    # parser.add_argument("--faiss_add", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    # index = faiss.read_index(args.pq_index)

    # try:
    #     args.modelpath = args.model_path + f"_{str(args.feature_level)}_{args.name_extra}_topk{args.topk}_weight_{index.coarsecode_size()+index.code_size}"
    # except :
    #     args.model_path = args.model_path + f"_{str(args.feature_level)}_{args.name_extra}_topk{args.topk}_weight_{index.code_size}"

    # if args.use_pq:
    #     if args.pq_index is None:
    #         raise ValueError("PQ index file is not provided.")
    #     lp._language_features_name = "language_features_pq"

    safe_state(args.quiet)
    torch.set_grad_enabled(False)

    # 在 PyTorch 中，autograd 会记录运算过程并自动求导。有时候在反向传播（loss.backward()）时，会遇到 NaN、inf 或非法操作，但报错信息并不会直接告诉你是在哪个算子里出的问题，调试起来很麻烦
    # 调用 torch.autograd.set_detect_anomaly(True) 后，PyTorch 会在反向传播时逐步检查每个算子 的梯度计算，一旦发现 NaN 或 inf，就会立刻报错，并指出具体是在哪个算子里出现的异常
    # 开启 anomaly detection 会显著降低训练速度（因为要逐步检查每一步运算），通常只在 调试阶段 使用，定位问题后应关闭
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    smoothing(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args)

    # All done
    print("\nSmoothing complete.")

