import os
import random
import argparse
import sys

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

# from CLIP.clip import clip

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",   # 使用半精度
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

@dataclass
class DINOv2ModelConfig:
    dino_model_pretrained: str = 'dinov2_vitb14'
    feat_dim: int = 768

class DINOv2Model(nn.Module):
    def __init__(self, config: DINOv2ModelConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load('facebookresearch/dinov2:qasfb-patch-3', config.dino_model_pretrained)
        model.eval()
        self.model = model.to("cuda").half()
    
    @property
    def embedding_dim(self):
        return self.config.feat_dim
        
    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model(processed_input)

def create(image_list, data_list, save_folder, dataset_path):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size = model.embedding_dim
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size)).cuda()    # [N,300,D]
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])).cuda()  # [N,4,H,W]
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        # image_list 是一个形状为 [N, 3, H, W] 的 4D Tensor
        # enumerate(image_list) 会将这个张量按第 0 维（batch 维）进行迭代
        # i 是索引：从 0 到 N-1
        # img 是每一张图片的张量，形状是 [3, H, W]
        # desc="Embedding images"：前缀标题
        # leave=False：运行完后不保留进度条

        # timer += 1
        # try:
        #     # img_embed: {'l': [b, D]}
        #     # seg_map: {'l': [H, W]}
        #     img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
        # except:
        #     seg_maps[i] = -1
        #     total_lengths.append(0)
        #     # raise ValueError(timer)
        #     continue   
        # sys.exit(0)发出的异常被try-except捕获导致程序没有正常退出
        img_embed, seg_map = _embed_clip_dino_sam_tiles(img.unsqueeze(0), sam_encoder)             

        # 在 PyTorch 中，len(tensor) 返回的是：张量第 0 维的大小，即 tensor.shape[0]
        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)
        
        # 动态扩充 img_embeds
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size)).cuda()
            ], dim=1)

        # 获取所有embeddings
        # img_embed [B,D] B = b1+b2+b3+b4
        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        # img_embeds [N,B,D]
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]    # [b1,b2,b3,b4] -> [b1,b1+b2,b1+b2+b3,b1+b2+b3+b4]
        for j, (k, v) in enumerate(seg_map.items()):
            # k没有用到
            if j == 0:
                seg_map_tensor.append(v)
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            # 由于所有level的embeddings都已经被顺序存入img_embeds中，为了保证seg_map中的索引能够正确访问到对应的embedding,
            # 需要给原索引加上一个偏移量获取正确的索引
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(v)
        seg_map = torch.stack(seg_map_tensor, dim=0)    # 形状取决于使用了几个level的分割结果，最多是 [4,H,W]
        # seg_maps[i] = seg_map.repeat(4,1,1) # [1, H, W] -> [4, H, W] 
        seg_maps[i] = seg_map
        # 最大的索引 + 1 = embeddings数目
        assert total_lengths[i] == int(seg_maps[i].max() + 1)

    # mask_generator.predictor.model.to('cpu')
        
    for i in tqdm(range(img_embeds.shape[0])):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],    # [B,D]
            'seg_maps': seg_maps[i] # [4,H,W]
        }
        sava_numpy(save_path, curr)

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].cpu().numpy())
    np.save(save_path_f, data['feature'].cpu().numpy())

def _embed_clip_dino_sam_tiles(image, sam_encoder):
    # image [1,3,H,W]
    aug_imgs = torch.cat([image])   # 虽然语法上是合法的，但它的作用其实没什么意义，结果 aug_imgs 和 image 是一样的
    seg_images, seg_map = sam_encoder(aug_imgs)
    # seg_images: { 'default': [b1,3,224,224], 's': [b2,3,224,224], 'm': [b3,3,224,224], 'l': [b4,3,224,224] }
    # seg_map: {'default': [H,W], 's': [H,W], 'm': [H,W], 'l': [H,W]}

    feat_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
    # for mode in ['l']:
        tiles = seg_images[mode]    # b,3,224,224
        tiles = tiles.to("cuda")
        with torch.no_grad():
            # clip_embed = model.encode_image(tiles)[0]
            # CLIP或者DINO
            feat_embed = model.encode_image(tiles)  # [b, D]
        feat_embed /= feat_embed.norm(dim=-1, keepdim=True)     # 将embedding归一化
        feat_embeds[mode] = feat_embed.detach().half()    # .half()将数据类型转换为 float16（半精度）
    
    # seg_map_l = {}
    # seg_map_l['l'] = seg_map['l']
    return feat_embeds, seg_map

# 从原图中提取一个被掩码（mask）指定的目标区域图像，并将背景设为黑色
def get_seg_img(mask, image):
    image = image.copy()    # 防止原图被修改
    image[mask['segmentation']==0] = np.array([0, 0, 0], dtype=np.uint8)   # 将分割掩码中为 0 的区域设为黑色，相当于“抠出前景”
    x,y,w,h = np.int32(mask['bbox'])    # 将边界框转换为整数，准备裁剪
    seg_img = image[y:y+h, x:x+w, ...]  # 裁剪出掩码所在的矩形区域
    return seg_img

# 将任意大小的彩色图片（非正方形）居中填充为正方形，背景填充为黑色（0）
def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    # [1,3,H,W] -> [H,W,3], BGR -> RGB
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)   # 像素范围是[0,255]
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)

    def mask2segmap(masks, image):
        seg_img_list = []
        # seg_map [H,W]
        seg_map = -torch.ones(image.shape[:2], dtype=torch.int32).cuda() # 创建一个与图像尺寸相同的二维数组 seg_map，并初始化为 -1
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))   # 通过先 pad 成正方形，不拉伸变形原图比例
            seg_img_list.append(pad_seg_img)

            seg_map[torch.from_numpy(masks[i]['segmentation']).cuda()] = i   # 将第 i 个目标的分割区域在 seg_map 中赋值为 i，从而在 seg_map 中标注每个目标的“编号”或“ID”
        seg_imgs = np.stack(seg_img_list, axis=0) # b,224,224,3
        # 在这里实现 [0， 255] -> [0.0, 1.0]
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')   # b,224,224,3 -> b,3,224,224    归一化到[0.0, 1.0]

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

# 设置随机种子（seed）来保证程序运行时的可复现性
def seed_everything(seed_value):
    # 设置 Python 内置的 random 模块的随机种子，确保使用 random 生成的随机数是可重复的
    random.seed(seed_value)
    # 设置 NumPy 的随机种子，确保 numpy.random 生成的随机数可重复
    np.random.seed(seed_value)
    # 设置 PyTorch 的 CPU 随机数生成器的种子，使得在 CPU 上的操作可复现
    torch.manual_seed(seed_value)
    # 设置环境变量 PYTHONHASHSEED，确保 Python 中哈希函数的结果是可预测的（有时影响某些算法的随机性）
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        # 设置 PyTorch 的 GPU 随机种子，使 GPU 上的操作也可复现
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # 强制使用确定性算法（例如在卷积操作中），以保证结果的一致性。但可能会导致速度变慢。
        torch.backends.cudnn.deterministic = True
        # 这个设置开启后，cuDNN 会在训练开始时自动寻找最优算法以加速计算。但这和上面的 deterministic=True 有冲突
        # 在保证可复现性时，通常建议将 benchmark=False，否则 cudnn 会选择最快的实现而不一定是确定的实现
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--encoder', type=str, default="dino")
    args = parser.parse_args()

    if args.encoder not in ['dino', 'clip']:
        print('[ ERROR ] Invalid encoder name.')
        sys.exit(-1)
    encoder = args.encoder

    # 设置默认的浮点数数据类型为 torch.float32（即 32 位浮点数），在没有明确指定 dtype 的情况下，PyTorch 会使用这个默认类型
    # 如果没有设置这个默认类型，有些 PyTorch 版本（尤其在开启 AMP 自动混合精度或使用 float64 时）可能默认是 float64，这样会导致不必要的类型转换或性能损失
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images_rgb')   # xxx/images
    data_list = os.listdir(img_folder)  # 图像名列表
    data_list.sort()

    # CLIP或DINO
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig) if encoder == 'clip' else DINOv2Model(DINOv2ModelConfig)
    # model, preprocess_for_tensor = clip.load("./CLIP/pretrain_models/ViT-B-16.pt", #"./CLIP/pretrain_models/RN50x64.pt", #"./CLIP/pretrain_models/ViT-B-16.pt", #"./CLIP/pretrain_models/RN50x64.pt", #"./CLIP/pretrain_models/ViT-L-14.pt",
    #                                                     device="cuda",
    #                                                     download_root='./CLIP/pretrain_models/',
    #                                                     if_transform_tensor=True)
    for name, param in model.named_parameters():
        param.requires_grad = False

    # SAM
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)  # cv2.imread(image_path) 读取图像时默认使用 BGR 格式，并且 像素值范围是 [0, 255]

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        # torch.from_numpy(image) 会保留 NumPy 的数据类型，所以变成一个 dtype=torch.uint8 的 Tensor，值仍在 [0, 255] 之间
        image = torch.from_numpy(image) # [H,W,3]
        img_list.append(image)
    # [H,W,3] -> [1,3,H,W] -> [N,3,H,W]
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)    # torch.cat(images) 默认在 维度 0（batch 维） 进行拼接

    folder_name = 'language_features_clip' if encoder == 'clip' else 'language_features_dino'
    save_folder = os.path.join(dataset_path, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder, dataset_path)
