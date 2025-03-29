#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/29 18:42
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : evaluate.py
# @Description   :


import argparse
import logging
import torch
import numpy as np
import os
import time
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import h5py

from data_loading import load_preprocessed_data, display_dataset_info, visualize_samples
from utils.dice_score import dice_coeff
import random

from utils.utils import set_seed


def predict_full_image(model, device, image, patch_size=256, overlap=0.5, batch_size=4):
    """
    Apply the model to a full image using a sliding window approach.

    Args:
        model: The trained model
        device: The device to run on
        image: The input image (HxWxC)
        patch_size: Size of patches to process
        overlap: Overlap between patches (0-1)
        batch_size: Batch size for processing

    Returns:
        pred_mask: The predicted segmentation mask
    """
    # Ensure image is in the right format
    if len(image.shape) == 3:
        # HxWxC to CxHxW
        image = np.transpose(image, (2, 0, 1))

    # Get dimensions
    c, h, w = image.shape

    # Calculate step size based on overlap
    step_size = int(patch_size * (1 - overlap))

    # Initialize prediction mask
    pred_mask = np.zeros((1, h, w))
    count_mask = np.zeros((1, h, w))

    # Calculate patch positions
    patch_positions = []

    for y in range(0, h - patch_size + 1, step_size):
        for x in range(0, w - patch_size + 1, step_size):
            patch_positions.append((y, x))

    # Process in batches
    model.eval()

    with torch.no_grad():
        for i in range(0, len(patch_positions), batch_size):
            batch_positions = patch_positions[i : i + batch_size]
            batch_patches = []

            for y, x in batch_positions:
                patch = image[:, y : y + patch_size, x : x + patch_size]
                batch_patches.append(patch)

            # Convert to tensor
            batch_tensor = torch.from_numpy(np.stack(batch_patches)).to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )

            # Get predictions
            outputs = model(batch_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Accumulate predictions
            for j, (y, x) in enumerate(batch_positions):
                pred_mask[:, y : y + patch_size, x : x + patch_size] += probs[j]
                count_mask[:, y : y + patch_size, x : x + patch_size] += 1

    # Average overlapping regions
    pred_mask = np.divide(
        pred_mask, count_mask, out=np.zeros_like(pred_mask), where=count_mask != 0
    )

    return pred_mask


def visualize_predictions(image, true_mask, pred_mask, output_path):
    """
    Create and save a visualization with the image, true mask, and predicted mask.

    Args:
        image: The input image (CxHxW)
        true_mask: The ground truth mask (1xHxW)
        pred_mask: The predicted mask (1xHxW)
        output_path: Where to save the visualization
    """
    # Create RGB versions for visualization
    if image.shape[0] == 1:
        image_rgb = np.repeat(image, 3, axis=0)
    else:
        image_rgb = image[:3]  # Use first 3 channels if more than 3

    # Create 3-channel representations of masks
    true_mask_rgb = np.zeros((3, true_mask.shape[1], true_mask.shape[2]))
    pred_mask_rgb = np.zeros((3, pred_mask.shape[1], pred_mask.shape[2]))

    # Use different colors for true positive, false positive, false negative
    true_positives = (true_mask > 0.5) & (pred_mask > 0.5)
    false_positives = (true_mask <= 0.5) & (pred_mask > 0.5)
    false_negatives = (true_mask > 0.5) & (pred_mask <= 0.5)

    # Green for true positives
    true_mask_rgb[1, true_mask[0] > 0.5] = 1.0

    # Set colors in predicted mask
    pred_mask_rgb[1, true_positives[0]] = 1.0  # Green for true positives
    pred_mask_rgb[0, false_positives[0]] = 1.0  # Red for false positives
    pred_mask_rgb[2, false_negatives[0]] = 1.0  # Blue for false negatives

    # Create a compound mask showing all errors
    compound_mask = np.zeros((3, true_mask.shape[1], true_mask.shape[2]))
    compound_mask[1, true_positives[0]] = 1.0  # Green for true positives
    compound_mask[0, false_positives[0]] = 1.0  # Red for false positives
    compound_mask[2, false_negatives[0]] = 1.0  # Blue for false negatives

    # Create a blank separator
    blank = np.zeros((3, true_mask.shape[1], 16))

    # Convert image to 0-1 range if needed
    if image_rgb.max() > 1.0:
        image_rgb = image_rgb / 255.0

    # Concatenate horizontally
    concat_image = np.concatenate(
        (image_rgb, blank, true_mask_rgb, blank, pred_mask_rgb, blank, compound_mask),
        axis=2,
    )

    # Convert to uint8 for saving
    concat_image = (concat_image * 255).astype(np.uint8).transpose(1, 2, 0)

    # Save the image
    Image.fromarray(concat_image).save(output_path)

    # Calculate Dice coefficient
    intersection = np.sum((true_mask > 0.5) & (pred_mask > 0.5))
    dice = (2.0 * intersection) / (np.sum(true_mask > 0.5) + np.sum(pred_mask > 0.5))

    return dice


def eval_model(
    model,
    device,
    output_dir,
    input_data: str = "./data/train_eye_dataset.h5",
    seed: int = 42,
    patch_size: int = 256,
    inference_batch_size: int = 32,
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

    # 设置patch的一半大小
    half_patch = patch_size // 2

    # data池子
    images_data_pool = np.array(dataset["images"]).transpose(0, 3, 1, 2)
    masks_data_pool = np.array(dataset["masks"])
    labels_data_pool = np.array(dataset["labels"])

    n_test, channel, width, height = images_data_pool.shape

    xmax = width - half_patch
    ymax = height - half_patch

    # Generate all image indices
    i_coords = np.arange(n_test)

    # Generate x and y coordinates
    x_coords = np.arange(half_patch, width, half_patch)
    x_coords = np.clip(x_coords, half_patch, xmax)
    y_coords = np.arange(half_patch, height, half_patch)
    y_coords = np.clip(y_coords, half_patch, ymax)

    # Create meshgrid for all combinations
    ii, xx, yy = np.meshgrid(i_coords, x_coords, y_coords, indexing="ij")

    # Stack and reshape
    sample_map_test = np.stack((ii, xx, yy), axis=-1).reshape(-1, 3)

    pred_data_map = np.zeros(masks_data_pool.shape)
    pred_count_map = np.zeros(masks_data_pool.shape)

    # masks_data_pool = np.expand_dims(masks_data_pool, axis=1)
    # labels_data_pool = np.expand_dims(labels_data_pool, axis=1)

    model.eval()
    batch_images_test = []
    batch_labels_test = []

    for img_idx, x_center, y_center in sample_map_test:
        # Extract the patch
        x_start = x_center - half_patch
        x_end = x_center + half_patch
        y_start = y_center - half_patch
        y_end = y_center + half_patch

        # Get image patch
        img_patch = images_data_pool[img_idx, :, x_start:x_end, y_start:y_end]
        batch_images_test.append(img_patch)

        # Get corresponding mask patch if needed
        label_patch = images_data_pool[img_idx, :, x_start:x_end, y_start:y_end]
        batch_labels_test.append(label_patch)

    # First stack the numpy arrays
    batch_images_test = np.stack(batch_images_test)
    batch_labels_test = np.stack(batch_labels_test)

    # Convert numpy arrays to PyTorch tensors
    batch_images_test = torch.from_numpy(batch_images_test)
    batch_labels_test = torch.from_numpy(batch_labels_test)

    # move images and labels to correct device and type
    batch_images_test = batch_images_test.to(
        device=device, dtype=torch.float32, memory_format=torch.channels_last
    )
    batch_labels_test = batch_labels_test.to(device=device, dtype=torch.long)

    # Get total number of samples
    total_samples = batch_images_test.shape[0]
    all_predictions = []

    # Evaluate in smaller batches
    with torch.no_grad():
        for start_idx in range(0, total_samples, inference_batch_size):
            end_idx = min(start_idx + inference_batch_size, total_samples)

            # Get current mini-batch
            mini_batch_images = batch_images_test[start_idx:end_idx]

            # Move mini-batch to device
            mini_batch_images = mini_batch_images.to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )

            # Forward pass
            mini_batch_preds = model(mini_batch_images)

            # Store predictions (move to CPU to save GPU memory)
            all_predictions.append(mini_batch_preds.cpu())

            # Optional: Clear GPU cache if needed
            torch.cuda.empty_cache()

        # Concatenate all predictions
        masks_pred = torch.cat(all_predictions, dim=0)
        masks_pred = torch.sigmoid(masks_pred)

        # If needed for evaluation metrics, move labels to device
        batch_labels_test = batch_labels_test.to(device=device, dtype=torch.long)

    # Convert predictions to numpy for easier handling
    masks_pred_np = masks_pred.numpy()

    # Loop through the test samples again to place predictions back into the original image
    for idx, (img_idx, x_center, y_center) in enumerate(sample_map_test):
        # Calculate patch boundaries
        x_start = x_center - half_patch
        x_end = x_center + half_patch
        y_start = y_center - half_patch
        y_end = y_center + half_patch

        # Add prediction to the accumulation map
        pred_data_map[img_idx, x_start:x_end, y_start:y_end] += masks_pred_np[idx, 0]

        # Increment the count map for this region (for averaging later)
        pred_count_map[img_idx, x_start:x_end, y_start:y_end] += 1

    # Average the predictions where patches overlap
    # Avoid division by zero
    mask = pred_count_map > 0
    pred_data_map[mask] = pred_data_map[mask] / pred_count_map[mask]

    pred_data_map = pred_data_map * masks_data_pool

    # print(pred_data_map.shape, labels_data_pool.shape, images_data_pool.shape)

    dice_score_list = []

    for i in range(n_test):
        pred_img = pred_data_map[i]
        label_img = labels_data_pool[i]
        image_img = images_data_pool[i]

        pred_img = np.repeat(np.expand_dims(pred_img, 0), 3, axis=0)
        label_img = np.repeat(np.expand_dims(label_img, 0), 3, axis=0)

        blank_image = np.zeros((3, 16, height))

        concat_image = np.concatenate(
            (image_img, blank_image, pred_img, blank_image, label_img), axis=1
        )

        concat_image = np.array(concat_image * 255).astype(np.uint8).transpose(1, 2, 0)

        Image.fromarray(concat_image).save(f"{output_dir}/prediction_{i}.png")

        masks_pred_binary = np.array(pred_data_map[i] > 0.5) * 1.0
        label_binary = labels_data_pool[i]
        dice_score = dice_coeff(
            torch.from_numpy(masks_pred_binary),
            torch.from_numpy(label_binary),
            reduce_batch_first=False,
        )

        dice_score_list.append(dice_score)

    print(f"Average Dice Score: {np.mean(dice_score_list):.4f}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict on full images using the trained model"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="best_model.pth",
        help="Path to the model file",
    )
    parser.add_argument(
        "--data-file",
        "-d",
        type=str,
        default="./data/test_eye_dataset.h5",
        help="Path to the h5 dataset",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./predictions",
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=4, help="Batch size for prediction"
    )
    parser.add_argument(
        "--patch-size",
        "-p",
        type=int,
        default=128,
        help="Size of patches for prediction",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5, help="Overlap between patches (0-1)"
    )
    parser.add_argument(
        "--num-images", "-n", type=int, default=5, help="Number of images to process"
    )
    parser.add_argument(
        "--image-indices",
        "-i",
        type=str,
        default=None,
        help="Comma-separated list of image indices to process",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=32,
        help="Batch size for inference to avoid GPU memory issues",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Load model
    logging.info(f"Loading model from {args.model}")
    model = torch.load(args.model, map_location=device)
    model.to(device=device)
    model = model.to(memory_format=torch.channels_last)

    eval_model(
        model=model,
        device=device,
        input_data=args.data_file,
        inference_batch_size=args.inference_batch_size,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
    )
