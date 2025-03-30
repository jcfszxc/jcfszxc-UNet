#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/27 15:58
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : 2.train.py
# @Description   :

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
from utils.utils import set_seed

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
    UNetPP
)


def train_model(
    model,
    device,
    input_data: str = "./data/train_eye_dataset.h5",
    # epochs: int = 5,
    steps: int = 100,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    patch_size: int = 256,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    seed: int = 42,
    early_stopping_patience: int = 20,
):
    set_seed(seed)

    # 1. 加载数据    dict_keys(['images', 'masks', 'labels', 'filenames'])
    dataset = load_preprocessed_data(input_data)

    # 2. 显示数据集信息
    display_dataset_info(dataset)

    # 3. 可视化样本
    visualize_samples(dataset, num_samples=3)

    # 4. 划分训练集和验证集
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
        f"""Starting training:
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.7,  # 每次降低到原来的70%
        patience=5,  # 5个epoch没有改善才降低学习率
        verbose=True,  # 打印学习率变化信息
        threshold=0.01,  # 性能改善需要超过1%才算显著改善
        cooldown=2,  # 学习率降低后等待2个epoch再次检查
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    # 设置patch的一半大小
    half_patch = patch_size // 2

    # ================================================================================
    # data池子
    images_data_pool = np.array(train_dataset["images"]).transpose(0, 3, 1, 2)
    masks_data_pool = np.array(train_dataset["masks"])
    labels_data_pool = np.array(train_dataset["labels"])

    n_train, channel, width, height = images_data_pool.shape

    # 获取masks_data_pool上，值不为0的xyz坐标
    sample_map = np.where(masks_data_pool != 0)

    valid_samples = (
        (sample_map[1] >= half_patch)  # x坐标不能太靠左
        & (sample_map[1] < width - half_patch)  # x坐标不能太靠右
        & (sample_map[2] >= half_patch)  # y坐标不能太靠上
        & (sample_map[2] < height - half_patch)  # y坐标不能太靠下
    )

    # 应用掩码，过滤出有效的坐标
    filtered_sample_map = (
        sample_map[0][valid_samples],  # 图像索引
        sample_map[1][valid_samples],  # x坐标
        sample_map[2][valid_samples],  # y坐标
    )

    masks_data_pool = np.expand_dims(masks_data_pool, axis=1)
    labels_data_pool = np.expand_dims(labels_data_pool, axis=1)

    # ================================================================================

    images_data_pool_val = np.array(val_dataset["images"]).transpose(0, 3, 1, 2)
    masks_data_pool_val = np.array(val_dataset["masks"])
    labels_data_pool_val = np.array(val_dataset["labels"])

    n_val, channel, width, height = images_data_pool_val.shape
    xmax = width - half_patch
    ymax = height - half_patch

    # Generate all image indices
    i_coords = np.arange(n_val)

    # Generate x and y coordinates
    x_coords = np.arange(half_patch, width, half_patch)
    x_coords = np.clip(x_coords, half_patch, xmax)
    y_coords = np.arange(half_patch, height, half_patch)
    y_coords = np.clip(y_coords, half_patch, ymax)

    # Create meshgrid for all combinations
    ii, xx, yy = np.meshgrid(i_coords, x_coords, y_coords, indexing="ij")

    # Stack and reshape
    sample_map_val = np.stack((ii, xx, yy), axis=-1).reshape(-1, 3)
    val_size = sample_map_val.shape[0]

    masks_data_pool_val = np.expand_dims(masks_data_pool_val, axis=1)
    labels_data_pool_val = np.expand_dims(labels_data_pool_val, axis=1)

    # 5. Begin training
    # for epoch in range(1, epochs + 1):
    epoch = 0  # Initialize epoch counter manually
    best_dice_score = 0

    while True:
        model.train()
        epoch_loss = 0
        epoch += 1
        with tqdm(total=steps, desc=f"Epoch {epoch}", unit="step") as pbar:
            for step in range(steps):

                tic = time.time()

                # 从sample_map中随机采样batch_size个样本
                random_indices = np.random.randint(
                    0, len(filtered_sample_map[0]), batch_size
                )

                batch_indices = (
                    filtered_sample_map[0][random_indices],
                    filtered_sample_map[1][random_indices],
                    filtered_sample_map[2][random_indices],
                )

                batch_images = []
                batch_labels = []

                for i in range(batch_size):
                    # Get coordinates for this sample
                    img_idx = batch_indices[0][i]
                    x_center = batch_indices[1][i]
                    y_center = batch_indices[2][i]

                    # Extract the patch
                    x_start = x_center - half_patch
                    x_end = x_center + half_patch
                    y_start = y_center - half_patch
                    y_end = y_center + half_patch

                    # Get image patch
                    img_patch = images_data_pool[
                        img_idx, :, x_start:x_end, y_start:y_end
                    ]
                    batch_images.append(img_patch)

                    # Get corresponding mask patch if needed
                    label_patch = labels_data_pool[
                        img_idx, :, x_start:x_end, y_start:y_end
                    ]
                    batch_labels.append(label_patch)

                # First stack the numpy arrays
                batch_images = np.stack(batch_images)
                batch_labels = np.stack(batch_labels)

                # Convert numpy arrays to PyTorch tensors
                batch_images = torch.from_numpy(batch_images)
                batch_labels = torch.from_numpy(batch_labels)

                # Now you can use PyTorch's .to() method to move tensors to the device and set dtype
                batch_images = batch_images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                batch_labels = batch_labels.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != "mps" else "cpu"):
                    masks_pred = model(batch_images)
                    
                    # Check for NaN in model output
                    if torch.isnan(masks_pred).any():
                        print("NaN in model output before loss calculation!")
                        continue
                        
                    # Apply sigmoid first for numerical stability
                    masks_pred_sigmoid = torch.sigmoid(masks_pred)
                    
                    # Calculate BCE loss
                    bce_loss = criterion(masks_pred, batch_labels)
                    
                    # Calculate Dice loss with the sigmoid-activated predictions
                    dice = dice_loss(
                        masks_pred_sigmoid.squeeze(1),
                        batch_labels.squeeze(1),
                        multiclass=False,
                    )
                    
                    # Combine losses with a weighting factor
                    alpha = 0.5  # You can adjust this weight
                    loss = alpha * bce_loss + (1 - alpha) * dice
                    
                    # Check for NaN in loss
                    if torch.isnan(loss).any():
                        print("NaN loss detected! Debugging info:")
                        print(f"BCE loss: {bce_loss.item()}")
                        print(f"Dice loss: {dice.item()}")
                        print(f"Batch images min/max: {batch_images.min()}/{batch_images.max()}")
                        print(f"Batch labels min/max: {batch_labels.min()}/{batch_labels.max()}")
                        print(f"Model output min/max: {masks_pred.min()}/{masks_pred.max()}")
                        continue

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss += loss.item()

                pbar.set_postfix(**{"loss (batch)": loss.item()})
                pbar.update(1)
                # sys.stdout.flush()

            # Clear line to avoid progress bar duplication
            print("\r", end="")

        # Validation at the end of each epoch
        model.eval()
        batch_images_val = []
        batch_labels_val = []

        for img_idx, x_center, y_center in sample_map_val:

            # Extract the patch
            x_start = x_center - half_patch
            x_end = x_center + half_patch
            y_start = y_center - half_patch
            y_end = y_center + half_patch

            # Get image patch
            img_patch = images_data_pool_val[img_idx, :, x_start:x_end, y_start:y_end]
            batch_images_val.append(img_patch)

            # Get corresponding mask patch if needed
            label_patch = labels_data_pool_val[img_idx, :, x_start:x_end, y_start:y_end]
            batch_labels_val.append(label_patch)

        # First stack the numpy arrays
        batch_images_val = np.stack(batch_images_val)
        batch_labels_val = np.stack(batch_labels_val)

        # Convert numpy arrays to PyTorch tensors
        batch_images_val = torch.from_numpy(batch_images_val)
        batch_labels_val = torch.from_numpy(batch_labels_val)

        # move images and labels to correct device and type
        batch_images_val = batch_images_val.to(
            device=device, dtype=torch.float32, memory_format=torch.channels_last
        )
        batch_labels_val = batch_labels_val.to(device=device, dtype=torch.long)

        # Evaluate
        with torch.no_grad():
            masks_pred = model(batch_images_val)
            masks_pred_binary = (torch.sigmoid(masks_pred) > 0.5).float()
            dice_score = dice_coeff(
                masks_pred_binary, batch_labels_val, reduce_batch_first=False
            )

            scheduler.step(dice_score)

            # 计算每个类别的Dice分数
            dice_score_bg = dice_coeff(
                masks_pred_binary, batch_labels_val, reduce_batch_first=False
            )
            masks_pred_binary = (torch.sigmoid(masks_pred) <= 0.5).float()
            dice_score_fg = dice_coeff(
                masks_pred_binary, 1 - batch_labels_val, reduce_batch_first=False
            )

            # 计算平均Dice分数
            dice_score_avg = (dice_score_bg + dice_score_fg) / 2

        # Early stopping logic
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            patience_counter = 0
            # Save the entire model instead of just state_dict
            torch.save(model, "best_model.pth")
            # print(f"New best dice score: {best_dice_score:.4f} - Saved model checkpoint")
        else:
            patience_counter += 1
            print(
                f"Dice score did not improve. Patience: {patience_counter}/{early_stopping_patience}"
            )

            if patience_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {epoch} epochs. Best dice score: {best_dice_score:.4f}"
                )
                break

        # Print validation results in a clean format
        print(
            f"Epoch {epoch} - "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} - "  # 使用科学计数法表示学习率0
            f"Loss: {epoch_loss/steps:.4g} - "  # 使用.4g格式去除不必要的0
            f"Dice: {dice_score:.4g} - "
            f"Avg Dice: {dice_score_avg:.4g} - "
            f"Best Dice: {best_dice_score:.4g}"
        )

        sample_num = 100
        sample_pred = torch.sigmoid(masks_pred).cpu().numpy()[sample_num]
        sample_label = batch_labels_val.cpu().numpy()[sample_num]
        sample_image = batch_images_val.cpu().numpy()[sample_num]

        sample_label = np.repeat(sample_label, 3, axis=0)
        sample_pred = np.repeat(sample_pred, 3, axis=0)

        blank_image = np.zeros((3, 16, patch_size))

        concat_image = np.concatenate(
            (sample_image, blank_image, sample_pred, blank_image, sample_label), axis=1
        )

        concat_image = np.array(concat_image * 255).astype(np.uint8).transpose(1, 2, 0)

        Image.fromarray(concat_image).save(
            f"visualizations/{epoch:03d}_{sample_num:03d}.png"
        )


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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    if args.load:
        # Load the complete model directly
        model = torch.load(args.load, map_location=device)
        logging.info(f"Model loaded from {args.load}")
    else:
        # model = UNet.UNet()  # lr=1e-6 Dice Score: 0.8098
        # model = AttentionUNet.AttentionUNet()  # lr=1e-6 Dice Score: 0.8098
        # model = DenseUNet.DenseUNet()  # lr=1e-6 Dice Score: 0.8115
        # model = MCUNet.MCUNet()  # lr=1e-6 Dice Score: 0.8051
        # model = ResUNet.ResUNet()  # lr=1e-6 Dice Score: 0.7609
        # model = FRUNet.FRUNet()  # lr=1e-6 Dice Score: 0.8227
        # model = MultiResUNet.MultiResUNet()  # lr=1e-6 Dice Score: 0.7778
        # model = SegNet.SegNet()  # lr=1e-6 Dice Score: 0.7325
        
        
        # model = R2UNet.R2UNet()  # 分数低，不适合
        # model = R2AttentionUNet.R2AttentionUNet()  # 分数低，不适合
        # model = BARUNet.BARUNet()  # 分数低，不适合
        # model = BIARUNet.BIARUNet()  # 分数低，不适合
        # model = BCDUNet.BCDU_net_D1(N=args.patch_size)  # 分数低，不适合
        # model = BCDUNet.BCDU_net_D3(N=args.patch_size)  # 分数低，不适合
        # model = RetinaLiteNet.TransFuseNet()  # 分数低，不适合
        # model = UNetPP.NestedUNet()  # 分数低，不适合
        
        
        model = model.to(memory_format=torch.channels_last)

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
    )

    model.to(device=device)
    train_model(
        model=model,
        device=device,
        input_data=args.data_file,
        # epochs=args.epochs,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_percent=args.val / 100,
        patch_size=args.patch_size,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )
