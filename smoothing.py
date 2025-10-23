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
    averaged_tensor = features.mean(dim=0).unsqueeze(0)  
    averaged_tensor = averaged_tensor / (averaged_tensor.norm(dim=-1, keepdim=True) + 1e-9)
    return averaged_tensor

def visualize_features(features):
    import umap

    features_np = features.cpu().numpy()

    # 使用UMAP进行降维，降到2维
    umap_model = umap.UMAP(
        n_neighbors=15,        # 设置邻居数，根据数据集调整
        min_dist=0.1,          # 设置最小距离，控制聚簇的紧密度
        metric='cosine',       # 使用余弦相似度作为度量
        n_components=2         # 降维到 2D 以便可视化
    )
    reduced_features = umap_model.fit_transform(features_np)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=0.01, c='blue', alpha=0.5)
    plt.title("UMAP Visualization of Feature Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    plt.savefig('umap_visualization.png', dpi=300) 
    plt.close()

def generate_lab_colors(M, device):
    # 在 Lab 空间中，L*：亮度（0到100），a*：红绿色（-128到127），b*：蓝黄色（-128到127）
    L = 50  # 固定亮度（L*）
    a = np.random.randint(-128, 128)  # 随机生成红绿分量 (a*)
    b = np.random.randint(-128, 128)  # 随机生成蓝黄分量 (b*)
    
    # 创建 Lab 颜色 (L*, a*, b*)
    lab_color = np.array([L, a, b])
    
    # 将 RGB 颜色复制 M 次，并转换为 PyTorch tensor
    lab_tensor = torch.tensor(np.tile(lab_color, (M, 1)), dtype=torch.float32, device=device)
    
    return lab_tensor

def majority_voting(gaussians, scene, pipe, background, dataset, args):
    t0 = time.perf_counter()

    folder_name = 'language_features_clip' if args.encoder == 'clip' else 'language_features_dino'
    # 这里 *list 的意思是把列表里的元素依次传入函数
    lf_path = "/" + os.path.join(*dataset.lf_path.split('/')[:-1], folder_name)
    feat_dim = 512 if args.encoder == 'clip' else 768
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
    num_masks_array = torch.zeros(len(viewpoint_stack), dtype=torch.int)
    feature_map_list = []
    seg_map_list = []
    for i in range(len(viewpoint_stack)):
        language_feature_name = os.path.join(lf_path, viewpoint_stack[i].image_name)
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy')).cuda().half()   # [B,D]
        num_masks_array[i] = feature_map.shape[0]   # 每张图像的feature数
        feature_map_list.append(feature_map)

        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy')).type(torch.int64)[dataset.feature_level].unsqueeze(0).cuda() # [1,H,W]
        seg_map_list.append(seg_map)

    features_array = torch.cat(feature_map_list, dim=0)     # [M,D]
    seg_maps = torch.cat(seg_map_list, dim=0)

    num_masks = features_array.shape[0]  # 所有图像的feature总数 M
    num_gaussians = len(gaussians.get_opacity)
    allocate_array = torch.zeros((num_gaussians, num_masks), dtype=torch.float16, device=gaussians.get_opacity.device)   # [N,M]
    offset = 0
    for i in tqdm(range(len(viewpoint_stack))):
        viewpoint_cam = viewpoint_stack[i]
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        ids, contribution = (
            # [H,W,100]
            render_pkg['per_pixel_gaussian_ids'].detach(),
            render_pkg['per_pixel_gaussian_contributions'].detach(), 
        )

        # self._feature_level = -1
        # ParamGroup中对下划线进行了处理
        # # 0:default 1:s 2:m 3:l
        seg_map = seg_maps[i:i + 1] # [1,H,W]
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

        # weight_sum = torch.zeros(num_gaussians)
        # 以下划线结尾代表原地操作
        # tensor.index_put_(indices, values, accumulate=False)用于通过索引将新的值填充到原始张量中的指定位置
        # indices：一个包含索引的元组，指定了在哪些位置进行赋值操作。它通常是一个包含多个张量的元组，每个张量代表一个维度上的索引
        # 一个张量，包含你想要放置在 indices 指定位置的值。这个张量的形状必须与 indices 所指定的位置一致
        # accumulate：一个布尔值，表示是否对指定位置的元素进行累加，False的话就直接赋值
        allocate_array.index_put_((ray_ids, gt_segmentations), weights.half(), accumulate=True)

        offset += num_masks_array[i]

    # 将feature vector归一化
    features_array /= (features_array.norm(dim=-1, keepdim=True) + 1e-9)

    weight_sum = torch.sum(allocate_array, 1)   # [N]
    allocate_array /= (weight_sum[:, None] + 1e-9)    # 对权值进行归一化

    # threshold = 1e-4
    # weight_sum_over_zero = weight_sum>0
    # weight_sum_under_threshold = weight_sum<threshold
    # reweight_index = weight_sum_over_zero * weight_sum_under_threshold
    # 先选择 allocate_array 中对应 reweight_index 为 True 的元素，然后从这些元素中选择大于零的元素，最后将这些大于零的元素的值设置为 1
    # allocate_array[reweight_index][allocate_array[reweight_index]>0] = 1    # 不会起作用

    # 对所有feature进行加权求和
    averaged_tensor = torch.matmul(allocate_array, features_array)
    averaged_tensor /= (averaged_tensor.norm(dim=-1, keepdim=True) + 1e-9)  # 归一化

    t1 = time.perf_counter()
    print(f"majority_voting: {(t1 - t0) * 1000:.3f} ms")

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

# block_size 小 → 矩阵更小，显存带宽压力小，cache 友好，Python 循环处理量也小 → 更快
# block_size 太大 → 矩阵乘法结果过大，显存和布尔运算开销拖慢速度
# block_size 太小 → kernel 启动开销 + Python 循环过多 → 也会变慢
def cosine_similarity_clustering(features, threshold=0.83, block_size=2000):
    t0 = time.perf_counter()

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
        s = sim_mask[free_mask[start:end]]      # Not: s = sim_mask 
        while s.shape[0] > 0:
            grp_mask = s[0] & free_mask     # [N]
            cluster_ids[grp_mask] = cluster_id
            free_mask[grp_mask] = False
            cluster_id += 1
            s = sim_mask[free_mask[start:end]]

    t1 = time.perf_counter()
    print(f"cosine_similarity_clustering: {(t1 - t0) * 1000:.3f} ms   {cluster_id} groups")

    return cluster_ids
        
def compute_significant_mask(viewpoint_stack, gaussians, pipe, background, max_threshold=0.1, use_max_weight=True, sum_threshold=3):
    t0 = time.perf_counter()

    device = gaussians.get_opacity.device
    N = len(gaussians.get_opacity)
    per_gauss_contrib = torch.zeros(N, device=device, dtype=torch.float32)  # [N]

    M = len(viewpoint_stack)
    block_size = 30
    for start in range(0, M, block_size):
        end = min(start + block_size, M)
        ids_list = []
        contribution_list = []
        for i in range(end - start):
            viewpoint_cam = viewpoint_stack[start + i]
            render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
            ids, contribution = (
                # [H,W,100]
                render_pkg['per_pixel_gaussian_ids'].detach(),
                render_pkg['per_pixel_gaussian_contributions'].detach(), 
            )
            ids_list.append(ids)
            contribution_list.append(contribution)
        ids_blk = torch.stack(ids_list, dim=0).reshape(-1)    # [B*H*W*L]
        contribution_blk = torch.stack(contribution_list, dim=0).reshape(-1) # [B*H*W*L]
        valid_mask = (ids_blk != -1)    # [B*H*W*L]
        contribution_blk = contribution_blk[valid_mask]
        ids_blk = ids_blk[valid_mask]
        if use_max_weight:
            out = scatter_max(contribution_blk, ids_blk.type(torch.long))[0]
            per_gauss_contrib[:out.shape[0]] = torch.max(per_gauss_contrib[:out.shape[0]], out)
        else:
            per_gauss_contrib.index_put_((ids_blk,), contribution_blk, accumulate=True)
    if use_max_weight:
        significant_mask = per_gauss_contrib > max_threshold  # [N]
    else:
        significant_mask = per_gauss_contrib > sum_threshold  # [N]

    t1 = time.perf_counter()
    print(f"compute_significant_mask: {(t1 - t0) * 1000:.3f} ms")

    return significant_mask

def laplacian_smoothing(gaussians, cluster_ids, full_significant_mask, lambda_reg=1e-3, k_neighbors=10000):
    t0 = time.perf_counter()

    N = gaussians.get_thermal_features.shape[0]
    device = gaussians.get_thermal_features.device
    # thermal_features = gaussians.get_thermal_features.clone()   # [N,K,3]
    colors = rgb_to_lab(torch.sigmoid(gaussians._thermal_features_dc))   # [N,3]

    unique_cluster_ids = torch.unique(cluster_ids)
    retain_mask = torch.ones(N, device=device, dtype=torch.bool)

    for cluster_id in unique_cluster_ids:
        # 如果在循环里频繁 gc.collect()，几乎等于每一步都在做“冷启动”，完全失去了 PyTorch 缓存的优势    1ms -> 600ms
        indices = torch.where(cluster_ids == cluster_id)[0] # [M]
        # print(f'{indices.shape[0]}')

        # colors[indices] = generate_lab_colors(indices.shape[0], indices.device)
        # continue

        significant_mask = full_significant_mask[indices]    # [M]

        known_colors = colors[indices][significant_mask]  # [M1,3]
        M1 = known_colors.shape[0]
        if M1 == 0:
            # retain_mask[indices] = False
            # print('Warning: there is no known color in a group')
            continue
        non_significant_mask = ~significant_mask    # [M]

        means = gaussians.get_xyz[indices]  # [M]
        M = means.shape[0]
        k = min(M - 1, k_neighbors)

        euclidean_dists = torch.cdist(means, means)   # [M,M]
        knn_idx = euclidean_dists.topk(k + 1, largest=False).indices   # [M,k+1]
        # weights = 1 / (euclidean_dists + 1e-9)  # [M,M]  
        weights = euclidean_dists.pow_(2).add_(1e-9).reciprocal_() # [M,M] 

        graph_mask = torch.zeros((M, M), device=device, dtype=torch.bool)     # [M,M]
        graph_mask.scatter_(1, knn_idx, True)
        graph_mask = graph_mask | graph_mask.T
        del knn_idx

        graph_mask.fill_diagonal_(0).logical_not_()
        weights[graph_mask] = 0.
        adj_matrix = weights
        del graph_mask

        degree_matrix = torch.diag(adj_matrix.sum(dim=1))   # [M,M]
        laplacian_matrix = degree_matrix - adj_matrix   # [M,M]
        del degree_matrix, adj_matrix

        L_ii = laplacian_matrix[non_significant_mask, :][:, non_significant_mask]  # [M2,M2]
        L_ij = laplacian_matrix[non_significant_mask, :][:, significant_mask]  # [M2,M1]
        del laplacian_matrix

        print(L_ii, L_ij)

        rhs = -torch.mm(L_ij, known_colors)  # [M2,3]
        lhs = L_ii + lambda_reg * torch.eye(L_ii.shape[0], device=device)   # [M2,M2]
        del L_ii, L_ij

        try:
            smooth_unknown_colors = torch.linalg.solve(lhs, rhs)    # [M2,3]
        except torch.linalg.LinAlgError:
            # 病态矩阵
            # print('Warning: the input matrix is singular')
            smooth_unknown_colors = torch.linalg.lstsq(lhs, rhs)[0]

        colors[indices[non_significant_mask]] = smooth_unknown_colors

        # colors[indices[non_significant_mask]] = rgb_to_lab(torch.ones_like(smooth_unknown_colors))

        # known_features = thermal_features[indices][significant_mask]  # [M1,K,3]
        # M1 = known_features.shape[0]
        # if M1 == 0:
        #     print('Warning: there is no known feature in a group')
        #     continue
        # M2 = M - M1
        # K = known_features.shape[1]
        # known_features = known_features.reshape(M1, -1) # [M1,K*3]

        # 对于非显著节点，使用图拉普拉斯平滑
        # 目标是解决方程: L_ii * unknown_features = -L_ij * known_features
        # rhs = -torch.mm(L_ij, known_features)  # [M2,K*3]
        # lhs = L_ii + lambda_reg * torch.eye(L_ii.shape[0], device=device)   # [M2,M2]

        # try:
        #     smooth_unknown_features = torch.linalg.solve(lhs, rhs).reshape(M2, K, 3)    # [M2,K*3] -> [M2,K,3]
        # except torch.linalg.LinAlgError:
        #     print('Warning: the input matrix is singular')
        #     smooth_unknown_features = torch.linalg.lstsq(lhs, rhs)[0].reshape(M2, K, 3)

        # del L_ii, L_ij
        # gc.collect()

        # thermal_features[indices][non_significant_mask] = smooth_unknown_features
    
    t1 = time.perf_counter()
    print(f"laplacian_smoothing: {(t1 - t0) * 1000:.3f} ms")

    # return thermal_features
    return  torch.logit(lab_to_rgb(colors), eps=1e-10), retain_mask  # [N,3]

def temperature_propagation(gaussians, scene, pipe, background, dataset, args):
    t0 = time.perf_counter()

    viewpoint_stack = scene.getTestCameras().copy()
    print(f'Use {len(viewpoint_stack)} images for significant mask computation')
    cluster_ids = cosine_similarity_clustering(gaussians.get_language_feature)
    significant_mask = compute_significant_mask(viewpoint_stack, gaussians, pipe, background)
    # thermal_features = laplacian_smoothing(gaussians, cluster_ids, significant_mask)
    thermal_colors, retain_mask = laplacian_smoothing(gaussians, cluster_ids, significant_mask)

    t1 = time.perf_counter()
    print(f"temperature_propagation: {(t1 - t0) * 1000:.3f} ms")

    # return thermal_features
    return thermal_colors, retain_mask
    
# training(lp.extract(args), op.extract(args), pp.extract(args), 
# args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
def smoothing(dataset, opt, pipe, checkpoint, args):
    t0 = time.perf_counter()

    gaussians = GaussianModel(dataset.sh_degree)    # 简单地给所有属性赋空值
    scene = Scene(dataset, gaussians, train_list_file=args.train_list_file)
    
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

    # thermal_features = temperature_propagation(gaussians, scene, pipe, background, dataset, args)
    # gaussians._thermal_features_dc = thermal_features[:, 0, :]
    # gaussians._thermal_features_rest = thermal_features[:, 1:, :]
    gaussians._thermal_features_dc, retain_mask = temperature_propagation(gaussians, scene, pipe, background, dataset, args)

    gaussians_mask = gaussians_mask.cpu()
    retain_mask = retain_mask.cpu()
    ckpt["pipeline"]["_model.gauss_params.thermal_features_dc"] = gaussians._thermal_features_dc[retain_mask].cpu()
    ckpt["pipeline"]["_model.gauss_params.thermal_features_rest"] = gaussians._thermal_features_rest[retain_mask].cpu()
    ckpt["pipeline"]["_model.gauss_params.means"] = ckpt["pipeline"]["_model.gauss_params.means"][gaussians_mask][retain_mask]
    ckpt["pipeline"]["_model.gauss_params.scales"] = ckpt["pipeline"]["_model.gauss_params.scales"][gaussians_mask][retain_mask]
    ckpt["pipeline"]["_model.gauss_params.quats"] = ckpt["pipeline"]["_model.gauss_params.quats"][gaussians_mask][retain_mask]
    ckpt["pipeline"]["_model.gauss_params.opacities"] = ckpt["pipeline"]["_model.gauss_params.opacities"][gaussians_mask][retain_mask]
    ckpt["pipeline"]["_model.gauss_params.features_dc"] = ckpt["pipeline"]["_model.gauss_params.features_dc"][gaussians_mask][retain_mask]
    ckpt["pipeline"]["_model.gauss_params.features_rest"] = ckpt["pipeline"]["_model.gauss_params.features_rest"][gaussians_mask][retain_mask]

    # print('Smoothing test passed.')
    # sys.exit(0)

    # base_name, ext = os.path.splitext(checkpoint)
    # new_checkpoint = base_name + "_origin" + ext
    # os.rename(checkpoint, new_checkpoint)
    os.remove(checkpoint)
    torch.save(ckpt, checkpoint)

    t1 = time.perf_counter()
    print(f"Total: {(t1 - t0) * 1000:.3f} ms")
    
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

def move_checkpoint_file(args):
    # 检查文件是否存在
    if os.path.exists(args.start_checkpoint):
        print(f"File {args.start_checkpoint} exists. Deleting it.")
        os.remove(args.start_checkpoint)  # 删除文件

    # 获取 args.start_checkpoint 所在目录的父目录
    parent_dir = os.path.dirname(os.path.abspath(args.start_checkpoint))
    checkpoint_name = os.path.basename(os.path.abspath(args.start_checkpoint))
    origin_checkpoint_path = os.path.join(os.path.dirname(parent_dir), 'origin', checkpoint_name)

    # 检查 origin/step-000000299.ckpt 是否存在
    if os.path.exists(origin_checkpoint_path):
        print(f"Copying {origin_checkpoint_path} to {parent_dir}")
        shutil.copy2(origin_checkpoint_path, args.start_checkpoint)  # 复制文件
    else:
        print(f"File {origin_checkpoint_path} does not exist.")
        sys.exit(-1)

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, OptimizationParams

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) # self.include_feature
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    # parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[0])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    # parser.add_argument("--name_extra", type=str, default = None)
    # parser.add_argument("--mode", type=str, default = "mean")
    parser.add_argument("--topk", type=int, default = 1)
    parser.add_argument('--encoder', type=str, default="dino")
    parser.add_argument('--train_list_file', type=str, default=None)
    parser.add_argument('--vram', type=int, required=False, default=32)
    
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

    import torch
    from shuffle import calculate_required_elements

    required_elements = calculate_required_elements(args.vram)
    occupied = torch.empty(required_elements, dtype=torch.float32, device='cuda')
    del occupied

    import os
    os.environ["MKL_NUM_THREADS"] = "12"
    os.environ["NUMEXPR_NUM_THREADS"] = "12"
    os.environ["OMP_NUM_THREADS"] = "12"
    import time
    from torch_scatter import scatter_max
    import torch.nn.functional as F
    from random import randint
    from gaussian_renderer import count_render
    from scene import Scene, GaussianModel
    from utils.general_utils import safe_state
    import uuid
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import shutil
    from tsplatter.utils.color_utils import rgb_to_lab, lab_to_rgb

    if args.encoder not in ['dino', 'clip']:
        print('[ ERROR ] Invalid encoder name.')
        sys.exit(-1)

    safe_state(args.quiet)
    torch.set_grad_enabled(False)

    move_checkpoint_file(args)

    # 在 PyTorch 中，autograd 会记录运算过程并自动求导。有时候在反向传播（loss.backward()）时，会遇到 NaN、inf 或非法操作，但报错信息并不会直接告诉你是在哪个算子里出的问题，调试起来很麻烦
    # 调用 torch.autograd.set_detect_anomaly(True) 后，PyTorch 会在反向传播时逐步检查每个算子 的梯度计算，一旦发现 NaN 或 inf，就会立刻报错，并指出具体是在哪个算子里出现的异常
    # 开启 anomaly detection 会显著降低训练速度（因为要逐步检查每一步运算），通常只在 调试阶段 使用，定位问题后应关闭
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    smoothing(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args)

    # All done
    print("\nSmoothing complete.")

