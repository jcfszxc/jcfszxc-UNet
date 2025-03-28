#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description   : Script to run predictions on full images

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

def load_image_from_h5(file_path, image_index=0):
    """
    Load a single image from the h5 dataset.
    
    Args:
        file_path: Path to h5 file
        image_index: Index of the image to load
        
    Returns:
        image: Image data as numpy array
    """
    with h5py.File(file_path, 'r') as h5f:
        images = h5f['images'][image_index]
        masks = h5f['masks'][image_index] if 'masks' in h5f else None
        
    return images, masks

def predict_full_image(
        model,
        device,
        image,
        patch_size=256,
        overlap=0.5,
        batch_size=4
):
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
            batch_positions = patch_positions[i:i+batch_size]
            batch_patches = []
            
            for y, x in batch_positions:
                patch = image[:, y:y+patch_size, x:x+patch_size]
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
                pred_mask[:, y:y+patch_size, x:x+patch_size] += probs[j]
                count_mask[:, y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping regions
    pred_mask = np.divide(pred_mask, count_mask, out=np.zeros_like(pred_mask), where=count_mask!=0)
    
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
    concat_image = np.concatenate((
        image_rgb, 
        blank, 
        true_mask_rgb, 
        blank, 
        pred_mask_rgb,
        blank,
        compound_mask
    ), axis=2)
    
    # Convert to uint8 for saving
    concat_image = (concat_image * 255).astype(np.uint8).transpose(1, 2, 0)
    
    # Save the image
    Image.fromarray(concat_image).save(output_path)
    
    # Calculate Dice coefficient
    intersection = np.sum((true_mask > 0.5) & (pred_mask > 0.5))
    dice = (2. * intersection) / (np.sum(true_mask > 0.5) + np.sum(pred_mask > 0.5))
    
    return dice

def get_args():
    parser = argparse.ArgumentParser(description='Predict on full images using the trained model')
    parser.add_argument('--model', '-m', type=str, default='best_model.pth',
                        help='Path to the model file')
    parser.add_argument('--data-file', '-d', type=str, default='./data/test_eye_dataset.h5',
                        help='Path to the h5 dataset')
    parser.add_argument('--output-dir', '-o', type=str, default='./predictions',
                        help='Directory to save predictions')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                        help='Batch size for prediction')
    parser.add_argument('--patch-size', '-p', type=int, default=256,
                        help='Size of patches for prediction')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap between patches (0-1)')
    parser.add_argument('--num-images', '-n', type=int, default=5,
                        help='Number of images to process')
    parser.add_argument('--image-indices', '-i', type=str, default=None,
                        help='Comma-separated list of image indices to process')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Load model
    logging.info(f'Loading model from {args.model}')
    model = torch.load(args.model, map_location=device)
    model.to(device=device)
    model = model.to(memory_format=torch.channels_last)
    
    # Determine which images to process
    if args.image_indices:
        indices = [int(idx) for idx in args.image_indices.split(',')]
    else:
        # Open the h5 file to get the number of images
        with h5py.File(args.data_file, 'r') as h5f:
            total_images = h5f['images'].shape[0]
        
        # Select random indices if not specified
        indices = np.random.choice(
            total_images, 
            min(args.num_images, total_images), 
            replace=False
        )
    
    # Process each selected image
    dice_scores = []
    
    for idx in indices:
        logging.info(f'Processing image {idx}')
        
        # Load image and mask
        image, mask = load_image_from_h5(args.data_file, idx)
        
        # Predict on the full image
        logging.info(f'Running prediction...')
        pred_mask = predict_full_image(
            model, 
            device, 
            image, 
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size
        )
        
        # Visualize results
        logging.info(f'Creating visualization...')
        output_path = os.path.join(args.output_dir, f'prediction_{idx:03d}.png')
        
        # Prepare for visualization
        image_vis = np.transpose(image, (2, 0, 1))  # HWC to CHW
        mask_vis = np.expand_dims(mask, axis=0)  # Add channel dimension
        
        dice = visualize_predictions(image_vis, mask_vis, pred_mask, output_path)
        dice_scores.append(dice)
        
        logging.info(f'Image {idx} - Dice score: {dice:.4f}')
        
        # Save binary prediction
        binary_pred = (pred_mask > 0.5).astype(np.uint8) * 255
        binary_pred_img = Image.fromarray(binary_pred[0]).convert('L')
        binary_pred_img.save(os.path.join(args.output_dir, f'binary_pred_{idx:03d}.png'))
    
    # Report overall results
    mean_dice = np.mean(dice_scores)
    logging.info(f'Overall mean Dice score: {mean_dice:.4f}')
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'prediction_results.txt'), 'w') as f:
        f.write(f'Prediction Results:\n')
        f.write(f'Data file: {args.data_file}\n')
        f.write(f'Model: {args.model}\n')
        f.write(f'Overall mean Dice score: {mean_dice:.4f}\n\n')
        f.write('Per-image results:\n')
        
        for i, idx in enumerate(indices):
            f.write(f'Image {idx}: Dice = {dice_scores[i]:.4f}\n')