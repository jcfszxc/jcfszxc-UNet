




import argparse
import logging
import torch
import glob
import os
import numpy as np
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import time
import torch.nn.functional as F
from PIL import Image
import random

from data_loading import load_preprocessed_data, display_dataset_info, visualize_samples
from utils.dice_score import dice_coeff, dice_loss
from utils.utils import set_seed, set_deterministic_mode

from UNetFamily import (
    UNet,
    AttentionUNet,
    R2UNet,
    R2AttentionUNet,
    BARUNet,
    BIARUNet,
    DenseUNet,
    MCUNet,
    ResUNet,
    FRUNet,
    MultiResUNet,
    BCDUNet,
    SegNet,
    RetinaLiteNet,
    UNetPP,
)




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 自己实现分形维数计算，而不是依赖不存在的库函数
import numpy as np
from scipy.ndimage import zoom
# 不使用pytorch3d，它可能也不在你的环境中
import math

# ========================= 1. 分形多尺度采样策略 =========================
def fractal_sampling(images_data, masks_data, patch_size, batch_size, fractal_levels=3):
    """
    使用基于分形的多尺度采样策略
    
    Args:
        images_data: 图像数据 (N, C, H, W)
        masks_data: 掩码数据 (N, 1, H, W)
        patch_size: 基础块大小
        batch_size: 批次大小
        fractal_levels: 分形级别数量
        
    Returns:
        batch_images: 分形采样的图像批次
        batch_masks: 对应的掩码批次
    """
    n_samples, channels, height, width = images_data.shape
    
    # 初始化结果列表
    batch_images = []
    batch_masks = []
    
    # 计算不同尺度的patch大小 (分形思想：不同尺度下的自相似性)
    scale_factors = [1 / (1.5 ** i) for i in range(fractal_levels)]
    patch_sizes = [int(patch_size * sf) for sf in scale_factors]
    
    # 确保最小patch大小至少为16
    patch_sizes = [max(ps, 16) for ps in patch_sizes]
    
    # 为每个尺度分配样本数量，大尺度更少，小尺度更多 (分形特性)
    # 遵循分形的功率律分布
    sample_distribution = [int(batch_size * (1/2)**i) for i in range(fractal_levels)]
    # 确保总数等于batch_size
    remaining = batch_size - sum(sample_distribution)
    sample_distribution[0] += remaining
    
    for level, (curr_patch_size, num_samples) in enumerate(zip(patch_sizes, sample_distribution)):
        if num_samples <= 0:
            continue
            
        half_patch = curr_patch_size // 2
        
        # 特别关注血管分叉点，这些点有更高的分形维数
        if level == 0:  # 大尺度 - 查找主要血管
            # 找到掩码中值较高的区域 (主血管)
            sample_map = np.where(masks_data > 0.7)
        elif level == 1:  # 中尺度 - 查找分支点
            # 使用梯度计算找到血管分叉区域
            grad_x = np.abs(np.gradient(masks_data.squeeze(), axis=1))
            grad_y = np.abs(np.gradient(masks_data.squeeze(), axis=2))
            gradient_magnitude = grad_x + grad_y
            sample_map = np.where(gradient_magnitude > np.percentile(gradient_magnitude, 90))
        else:  # 小尺度 - 随机采样小毛细血管
            sample_map = np.where(masks_data > 0.3)
        
        # 有效性检查
        valid_samples = (
            (sample_map[1] >= half_patch) & 
            (sample_map[1] < width - half_patch) & 
            (sample_map[2] >= half_patch) & 
            (sample_map[2] < height - half_patch)
        )
        
        filtered_sample_map = (
            sample_map[0][valid_samples],
            sample_map[1][valid_samples],
            sample_map[2][valid_samples],
        )
        
        if len(filtered_sample_map[0]) == 0:
            # 如果没有满足条件的样本，退回到随机采样
            filtered_sample_map = np.where(masks_data > 0.1)
            valid_samples = (
                (filtered_sample_map[1] >= half_patch) & 
                (filtered_sample_map[1] < width - half_patch) & 
                (filtered_sample_map[2] >= half_patch) & 
                (filtered_sample_map[2] < height - half_patch)
            )
            filtered_sample_map = (
                filtered_sample_map[0][valid_samples],
                filtered_sample_map[1][valid_samples],
                filtered_sample_map[2][valid_samples],
            )
            
        # 随机选择样本
        if len(filtered_sample_map[0]) > 0:
            random_indices = np.random.randint(0, len(filtered_sample_map[0]), num_samples)
            
            batch_indices = (
                filtered_sample_map[0][random_indices],
                filtered_sample_map[1][random_indices],
                filtered_sample_map[2][random_indices],
            )
            
            for i in range(num_samples):
                img_idx = batch_indices[0][i]
                x_center = batch_indices[1][i]
                y_center = batch_indices[2][i]
                
                # 提取patch
                x_start = max(0, x_center - half_patch)
                x_end = min(width, x_center + half_patch)
                y_start = max(0, y_center - half_patch)
                y_end = min(height, y_center + half_patch)
                
                img_patch = images_data[img_idx, :, x_start:x_end, y_start:y_end]
                mask_patch = masks_data[img_idx, :, x_start:x_end, y_start:y_end]
                
                # 调整到统一大小 (patch_size x patch_size)
                if img_patch.shape[1] != patch_size or img_patch.shape[2] != patch_size:
                    scale = (1, patch_size/img_patch.shape[1], patch_size/img_patch.shape[2])
                    img_patch = torch.from_numpy(
                        zoom(img_patch, scale, order=1)
                    ).float()
                    mask_patch = torch.from_numpy(
                        zoom(mask_patch, scale, order=0)
                    ).float()
                else:
                    img_patch = torch.from_numpy(img_patch).float()
                    mask_patch = torch.from_numpy(mask_patch).float()
                
                batch_images.append(img_patch)
                batch_masks.append(mask_patch)
                
    # 如果没有足够的样本，则随机填充
    while len(batch_images) < batch_size:
        # 随机选择已有样本复制
        idx = np.random.randint(0, len(batch_images))
        batch_images.append(batch_images[idx])
        batch_masks.append(batch_masks[idx])
    
    # 堆叠批次
    batch_images = torch.stack(batch_images)
    batch_masks = torch.stack(batch_masks)
    
    return batch_images, batch_masks


# ========================= 2. 分形维度特征 =========================
class FractalFeatureExtractor(nn.Module):
    """
    提取分形特征并与常规特征融合
    """
    def __init__(self, in_channels):
        super(FractalFeatureExtractor, self).__init__()
        self.in_channels = in_channels
        
        # 分形特征提取
        self.fractal_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # 多尺度特征提取 (模拟分形的多尺度特性)
        self.scales = [1, 2, 4, 8]
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 16, kernel_size=3, dilation=scale, padding=scale)
            for scale in self.scales
        ])
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(16 * len(self.scales) + 1, in_channels, kernel_size=1)
        
    def forward(self, x):
        # 计算分形特征
        fractal_feat = self.fractal_conv(x)
        
        # 多尺度特征提取
        multi_scale_feats = []
        for conv in self.multi_scale_convs:
            multi_scale_feats.append(F.relu(conv(x)))
        
        # 拼接多尺度特征
        concat_feats = torch.cat(multi_scale_feats + [fractal_feat], dim=1)
        
        # 特征融合
        enhanced_features = self.fusion_conv(concat_feats)
        
        # 残差连接
        return enhanced_features + x


# ========================= 3. 分形增强的损失函数 =========================
class FractalLoss(nn.Module):
    """
    基于分形几何的损失函数
    
    结合了传统损失函数和分形特性的损失
    """
    def __init__(self, alpha=0.3, beta=0.2, gamma=0.5):
        super(FractalLoss, self).__init__()
        self.alpha = alpha  # BCE权重
        self.beta = beta   # Dice损失权重
        self.gamma = gamma  # 分形损失权重
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def box_dimension(self, mask, max_scales=4):
        """估计方框维数，一种分形维数的近似
        使用方框计数法(Box-counting method)估计分形维数
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
            
        # 确保mask是2D数组
        if mask.ndim > 2:
            mask = mask.squeeze()
            
        # 二值化
        mask_binary = (mask > 0.5).astype(np.float32)
        if mask_binary.sum() == 0:
            return 0.0
            
        # 方框计数法估计分形维数
        counts = []
        scales = []
        
        for scale in range(1, max_scales + 1):
            box_size = 2 ** scale
            boxes_x = math.ceil(mask.shape[0] / box_size)
            boxes_y = math.ceil(mask.shape[1] / box_size)
            count = 0
            
            # 计算非空盒子数量
            for i in range(boxes_x):
                for j in range(boxes_y):
                    x_start = i * box_size
                    y_start = j * box_size
                    x_end = min(x_start + box_size, mask.shape[0])
                    y_end = min(y_start + box_size, mask.shape[1])
                    
                    if np.any(mask_binary[x_start:x_end, y_start:y_end] > 0):
                        count += 1
                        
            counts.append(count)
            scales.append(box_size)
            
        if len(counts) <= 1 or min(counts) == 0:
            return 0.0
            
        # 对数线性回归计算分形维数
        log_counts = np.log(np.array(counts) + 1e-10)  # 添加小值防止log(0)
        log_scales = np.log(np.array(scales))
        
        # 使用NumPy的polyfit进行线性回归
        try:
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            return -slope  # 分形维数 = -斜率
        except:
            # 回退到手动计算
            n = len(scales)
            sum_x = log_scales.sum()
            sum_y = log_counts.sum()
            sum_xy = (log_scales * log_counts).sum()
            sum_xx = (log_scales * log_scales).sum()
            
            if (n * sum_xx - sum_x * sum_x) == 0:
                return 0.0
                
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            return -slope
        
    def forward(self, pred, target):
        # 应用sigmoid以获得概率
        pred_sigmoid = torch.sigmoid(pred)
        
        # BCE损失
        bce = self.bce_loss(pred, target)
        
        # Dice损失
        dice = 1 - (2 * (pred_sigmoid * target).sum()) / (
            (pred_sigmoid + target).sum() + 1e-8
        )
        
        # 分形损失计算
        batch_size = pred.size(0)
        fractal_loss = 0.0
        
        # 批量处理 - 只计算一部分样本以提高效率
        sample_size = min(4, batch_size)
        indices = torch.randperm(batch_size)[:sample_size]
        
        for idx in indices:
            target_fractal = self.box_dimension(target[idx].squeeze().detach().cpu().numpy())
            pred_fractal = self.box_dimension(pred_sigmoid[idx].squeeze().detach().cpu().numpy())
            fractal_loss += abs(target_fractal - pred_fractal)
            
        fractal_loss = fractal_loss / sample_size if sample_size > 0 else 0.0
        
        # 合并损失
        total_loss = self.alpha * bce + self.beta * dice + self.gamma * fractal_loss
        
        return total_loss

# ========================= 4. 分形自监督预训练 =========================
class FractalSelfSupervisedLoss(nn.Module):
    """
    分形自监督预训练损失函数
    
    使用分形特性作为自监督信号
    """
    def __init__(self):
        super(FractalSelfSupervisedLoss, self).__init__()
        
    def forward(self, pred_large, pred_small, original_image):
        """
        Args:
            pred_large: 大尺度的预测
            pred_small: 小尺度的预测 (调整大小后)
            original_image: 原始图像
        """
        # 使用分形的自相似性 - 小尺度与大尺度之间的一致性
        consistency_loss = F.mse_loss(pred_large, pred_small)
        
        # 血管分支一致性 - 计算图像梯度来识别分支
        # 使用Sobel算子近似替代torch.gradient (在某些PyTorch版本中可能不可用)
        def sobel_gradients(x):
            # 定义Sobel算子
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
            
            # 复制到各通道
            if x.size(1) > 1:
                sobel_x = sobel_x.repeat(1, x.size(1), 1, 1)
                sobel_y = sobel_y.repeat(1, x.size(1), 1, 1)
            
            # 计算梯度
            pad = nn.ReflectionPad2d(1)
            x_padded = pad(x)
            grad_x = F.conv2d(x_padded, sobel_x, groups=x.size(1))
            grad_y = F.conv2d(x_padded, sobel_y, groups=x.size(1))
            
            return torch.abs(grad_x), torch.abs(grad_y)
        
        # 计算大尺度梯度
        large_grad_x, large_grad_y = sobel_gradients(pred_large)
        large_grad = large_grad_x + large_grad_y
        
        # 计算小尺度梯度
        small_grad_x, small_grad_y = sobel_gradients(pred_small)
        small_grad = small_grad_x + small_grad_y
        
        # 分支结构一致性
        branch_loss = F.mse_loss(large_grad, small_grad)
        
        return consistency_loss + 0.5 * branch_loss

# 将这些组件集成到训练函数中
def train_with_fractal_optimization(
    model,
    device,
    input_data,
    steps=100,
    batch_size=1,
    learning_rate=1e-5,
    val_percent=0.1,
    patch_size=256,
    weight_decay=1e-8,
    momentum=0.999,
    seed=42,
    early_stopping_patience=20,
):
    # 加载数据
    dataset = load_preprocessed_data(input_data)
    
    # 显示数据集信息
    display_dataset_info(dataset)
    
    # 可视化样本
    visualize_samples(dataset, num_samples=3)
    
    # 划分训练集和验证集
    n_samples = len(dataset["images"])
    n_val = int(n_samples * val_percent)
    n_train = len(dataset["images"]) - n_val
    
    # 创建索引并随机打乱
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 分割索引
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # 创建训练集和验证集字典
    train_dataset = {
        key: [dataset[key][i] for i in train_indices] for key in dataset.keys()
    }
    val_dataset = {
        key: [dataset[key][i] for i in val_indices] for key in dataset.keys()
    }
    
    print(f"训练集样本数: {len(train_dataset['images'])}")
    print(f"验证集样本数: {len(val_dataset['images'])}")
    
    logging.info(
        f"""Starting training with fractal optimization:
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    """
    )
    
    # 添加分形特征提取器
    fractal_feature_extractor = FractalFeatureExtractor(model.n_channels).to(device)
    
    # 设置优化器
    optimizer = optim.RMSprop(
        list(model.parameters()) + list(fractal_feature_extractor.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.7,
        patience=5,
        verbose=True,
        threshold=0.01,
        cooldown=2,
    )
    
    # 梯度缩放器
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # 损失函数
    criterion = FractalLoss(alpha=0.3, beta=0.3, gamma=0.4)
    
    # 预处理数据
    images_data_pool = np.array(train_dataset["images"]).transpose(0, 3, 1, 2)
    masks_data_pool = np.array(train_dataset["masks"])
    masks_data_pool = np.expand_dims(masks_data_pool, axis=1)
    
    images_data_pool_val = np.array(val_dataset["images"]).transpose(0, 3, 1, 2)
    masks_data_pool_val = np.array(val_dataset["masks"])
    masks_data_pool_val = np.expand_dims(masks_data_pool_val, axis=1)
    
    # 开始训练
    epoch = 0
    best_dice_score = 0
    patience_counter = 0
    
    while True:
        model.train()
        fractal_feature_extractor.train()
        epoch_loss = 0
        epoch += 1
        
        with tqdm(total=steps, desc=f"Epoch {epoch}", unit="step") as pbar:
            for step in range(steps):
                # 使用分形采样策略
                batch_images, batch_labels = fractal_sampling(
                    images_data_pool, masks_data_pool, patch_size, batch_size, fractal_levels=3
                )
                
                # 将tensor移动到设备上
                batch_images = batch_images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                batch_labels = batch_labels.to(device=device, dtype=torch.float32)
                
                with torch.autocast(device.type if device.type != "mps" else "cpu"):
                    # 提取分形特征
                    enhanced_input = fractal_feature_extractor(batch_images)
                    
                    # 模型预测
                    masks_pred = model(enhanced_input)
                    
                    # 检查NaN
                    if torch.isnan(masks_pred).any():
                        print("NaN in model output before loss calculation!")
                        continue
                        
                    # 计算损失
                    loss = criterion(masks_pred, batch_labels)
                    
                    # 检查NaN损失
                    if torch.isnan(loss).any():
                        print("NaN loss detected!")
                        continue
                
                # 优化步骤
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                epoch_loss += loss.item()
                
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                pbar.update(1)
        
        # 验证
        model.eval()
        fractal_feature_extractor.eval()
        
        # 创建验证批次 (为了简化，这里使用简单采样而非分形采样)
        n_val_samples = min(len(images_data_pool_val), 200)  # 限制验证样本数以加速
        indices = np.random.choice(len(images_data_pool_val), n_val_samples, replace=False)
        
        val_images = torch.from_numpy(images_data_pool_val[indices]).to(
            device=device, dtype=torch.float32, memory_format=torch.channels_last
        )
        val_labels = torch.from_numpy(masks_data_pool_val[indices]).to(
            device=device, dtype=torch.float32
        )
        
        # 验证评估
        with torch.no_grad():
            # 增强输入
            enhanced_input = fractal_feature_extractor(val_images)
            
            # 预测
            masks_pred = model(enhanced_input)
            
            # 二值化
            masks_pred_binary = (torch.sigmoid(masks_pred) > 0.5).float()
            
            # 计算Dice分数
            dice_score = dice_coeff(
                masks_pred_binary, val_labels, reduce_batch_first=False
            )
            
            # 学习率调度
            scheduler.step(dice_score)
        
        # 早停逻辑
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            patience_counter = 0
            
            torch.save(model, "best_model.pth")
            
            # 保存完整模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'fractal_extractor_state_dict': fractal_feature_extractor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "best_fractal_model.pth")
            
            print(f"New best dice score: {best_dice_score:.4f} - Saved model checkpoint")
        else:
            patience_counter += 1
            print(f"Dice score did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs. Best dice score: {best_dice_score:.4f}")
                break
        
        # 打印结果
        print(
            f"Epoch {epoch} - "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} - "
            f"Loss: {epoch_loss/steps:.4g} - "
            f"Dice: {dice_score:.4g} - "
            f"Best Dice: {best_dice_score:.4g}"
        )
        

                    # 可视化一些样本
        if epoch % 5 == 0:
            try:
                # 选择一个样本用于可视化
                sample_num = np.random.randint(0, len(val_images))
                sample_pred = torch.sigmoid(masks_pred).cpu().numpy()[sample_num]
                sample_label = val_labels.cpu().numpy()[sample_num]
                sample_image = val_images.cpu().numpy()[sample_num]
                
                # 获取图像尺寸
                _, h, w = sample_image.shape
                
                # 确保所有图像有相同尺寸
                sample_label = np.repeat(sample_label, 3, axis=0)
                sample_pred = np.repeat(sample_pred, 3, axis=0)
                
                # 创建与样本图像宽度匹配的空白图像
                blank_image = np.zeros((3, 16, w))
                
                # 确保所有数组第三维度相同
                if sample_pred.shape[2] != w:
                    sample_pred = zoom(sample_pred, (1, 1, w/sample_pred.shape[2]), order=0)
                if sample_label.shape[2] != w:
                    sample_label = zoom(sample_label, (1, 1, w/sample_label.shape[2]), order=0)
                
                # 拼接图像
                concat_image = np.concatenate(
                    (sample_image, blank_image, sample_pred, blank_image, sample_label), axis=1
                )
                
                # 转换为RGB格式
                concat_image = np.array(concat_image * 255).astype(np.uint8).transpose(1, 2, 0)
                
                # 保存可视化结果
                os.makedirs("visualizations", exist_ok=True)
                Image.fromarray(concat_image).save(
                    f"visualizations/fractal_{epoch:03d}_{sample_num:03d}.png"
                )
            except Exception as e:
                print(f"可视化过程中出错: {e}")
                print("跳过此次可视化，继续训练")

def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--data-file",
        "-d",
        type=str,
        default="./data/train_eye_dataset.h5",
        help="Path to the h5 dataset",
    )
    # parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-6,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file"
    )
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "--patch-size",
        "-p",
        dest="patch_size",
        type=int,
        default=128,
        help="Size of the patches extracted from the input images",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=100,
        help="Number of steps per epoch. If not provided, all data will be used",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--early-stopping-patience",
        "-esp",
        dest="early_stopping_patience",
        type=int,
        default=20,
        help="Number of epochs with no improvement after which training will be stopped",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    set_seed(args.seed)
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    
    if args.load:
        # 加载模型
        checkpoint = torch.load(args.load, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 加载优化后的模型
            model = UNet.UNet()
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Model loaded from {args.load}")
        else:
            # 加载原始模型
            model = checkpoint
            logging.info(f"Legacy model loaded from {args.load}")
    else:
        # 创建新模型
        model = UNet.UNet()
        
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)
    
    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
    )
    
    # 使用分形优化训练模型
    train_with_fractal_optimization(
        model=model,
        device=device,
        input_data=args.data_file,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_percent=args.val / 100,
        patch_size=args.patch_size,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )