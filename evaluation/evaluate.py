"""
Accuracy Evaluation Script for CSRNet
======================================
Evaluates the pretrained model against ground truth annotations.
Supports both ShanghaiTech and UCF-QNRF datasets.

Metrics computed:
    - MAE  (Mean Absolute Error)
    - MSE  (Mean Squared Error)
    - RMSE (Root Mean Squared Error)

Usage:
    # ShanghaiTech Part A (model was trained on this)
    python evaluate.py --model model_best.pth.tar --dataset shanghai --data data/ShanghaiTech/part_A/test_data

    # ShanghaiTech Part B
    python evaluate.py --model model_best.pth.tar --dataset shanghai --data data/ShanghaiTech/part_B/test_data

    # UCF-QNRF
    python evaluate.py --model model_best.pth.tar --dataset ucf-qnrf --data data/UCF-QNRF_ECCV18 --split Test

    # Quick test (limit to N images)
    python evaluate.py --model model_best.pth.tar --dataset shanghai --data data/ShanghaiTech/part_A/test_data --max-images 10
"""

import argparse
import os
import glob
import time

import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.io
from PIL import Image

from model import CrowdStampedeYantra as CSRNet


def load_gt_shanghaitech(ann_path):
    """Load ground truth from ShanghaiTech .mat file.
    Structure: image_info[0][0][0][0][0] -> (N, 2) array of head locations."""
    mat = scipy.io.loadmat(ann_path)
    locations = mat['image_info'][0][0][0][0][0]
    return locations.shape[0]


def load_gt_ucfqnrf(ann_path):
    """Load ground truth from UCF-QNRF .mat file.
    Structure: annPoints -> (N, 2) array of head locations."""
    mat = scipy.io.loadmat(ann_path)
    return mat['annPoints'].shape[0]


def get_image_gt_pairs(dataset, data_dir, split='Test'):
    """
    Returns list of (image_path, gt_ann_path) tuples for the given dataset.
    """
    pairs = []

    if dataset == 'shanghai':
        # ShanghaiTech is distributed in two different directory layouts
        # depending on source/mirror:
        #   "classic": data_dir/images/IMG_1.jpg, data_dir/ground-truth/GT_IMG_1.mat
        #   "flat":    data_dir/IMG_1.jpg, data_dir/GT_IMG_1.mat (same directory)
        # Detected rather than assumed, so pointing this at either layout works
        # instead of silently matching zero pairs when the assumed layout
        # (classic) isn't the one actually present.
        img_dir = os.path.join(data_dir, 'images')
        gt_dir = os.path.join(data_dir, 'ground-truth')
        if not (os.path.isdir(img_dir) and os.path.isdir(gt_dir)):
            img_dir = data_dir
            gt_dir = data_dir

        image_paths = sorted(glob.glob(os.path.join(img_dir, 'IMG_*.jpg')))
        for img_path in image_paths:
            img_name = os.path.basename(img_path)  # IMG_1.jpg
            gt_name = 'GT_' + img_name.replace('.jpg', '.mat')  # GT_IMG_1.mat
            gt_path = os.path.join(gt_dir, gt_name)
            if os.path.exists(gt_path):
                pairs.append((img_path, gt_path))

    elif dataset == 'ucf-qnrf':
        # UCF-QNRF: data_dir = .../UCF-QNRF_ECCV18, split = Test or Train
        split_dir = os.path.join(data_dir, split)
        image_paths = sorted(glob.glob(os.path.join(split_dir, 'img_*.jpg')))
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            ann_name = img_name.replace('.jpg', '_ann.mat')
            ann_path = os.path.join(split_dir, ann_name)
            if os.path.exists(ann_path):
                pairs.append((img_path, ann_path))

    return pairs


def evaluate(model_path, dataset, data_dir, split='Test', device=None, max_images=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load model
    print(f"Loading model from {model_path}...")
    model = CSRNet(load_weights=True)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Get image/GT pairs
    pairs = get_image_gt_pairs(dataset, data_dir, split)
    if max_images is not None:
        pairs = pairs[:max_images]

    # Choose GT loader
    load_gt = load_gt_shanghaitech if dataset == 'shanghai' else load_gt_ucfqnrf

    total = len(pairs)
    print(f"Dataset: {dataset.upper()}")
    print(f"Evaluating on {total} images from {data_dir}...\n")

    abs_errors = []
    sq_errors = []
    results = []

    print(f"{'#':>4}  {'Image':<20}  {'GT':>6}  {'Pred':>6}  {'Error':>8}  {'Time':>6}")
    print("-" * 70)

    for idx, (img_path, gt_path) in enumerate(pairs):
        img_name = os.path.basename(img_path)
        gt_count = load_gt(gt_path)

        t0 = time.time()
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            density_map = model(img_tensor)
        pred_count = int(round(density_map.sum().item()))
        elapsed = time.time() - t0

        error = pred_count - gt_count
        abs_errors.append(abs(error))
        sq_errors.append(error ** 2)
        results.append({
            'image': img_name,
            'gt': gt_count,
            'pred': pred_count,
            'error': error,
        })

        print(f"{idx+1:>4}  {img_name:<20}  {gt_count:>6}  {pred_count:>6}  {error:>+8}  {elapsed:>5.1f}s")

    # Summary
    mae = np.mean(abs_errors)
    mse = np.mean(sq_errors)
    rmse = np.sqrt(mse)

    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS ({dataset.upper()}, {len(abs_errors)} images)")
    print(f"{'='*70}")
    print(f"  MAE  (Mean Absolute Error):  {mae:.2f}")
    print(f"  MSE  (Mean Squared Error):   {mse:.2f}")
    print(f"  RMSE (Root Mean Sq. Error):  {rmse:.2f}")
    print(f"{'='*70}")

    if results:
        sorted_by_error = sorted(results, key=lambda x: abs(x['error']))
        print(f"\n  Top 5 BEST predictions (lowest error):")
        for r in sorted_by_error[:5]:
            print(f"    {r['image']}: GT={r['gt']}, Pred={r['pred']}, Error={r['error']:+d}")

        print(f"\n  Top 5 WORST predictions (highest error):")
        for r in sorted_by_error[-5:][::-1]:
            print(f"    {r['image']}: GT={r['gt']}, Pred={r['pred']}, Error={r['error']:+d}")

    print()
    return mae, mse, rmse


def main():
    parser = argparse.ArgumentParser(description='Evaluate CSRNet accuracy')
    parser.add_argument('--model', '-m', required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', choices=['shanghai', 'ucf-qnrf'], default='shanghai',
                        help='Dataset type: shanghai or ucf-qnrf')
    parser.add_argument('--data', '-d', required=True, help='Path to dataset directory')
    parser.add_argument('--split', '-s', default='Test', help='Split for UCF-QNRF (Test/Train)')
    parser.add_argument('--device', default=None, help='Device: cuda or cpu')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to evaluate')
    args = parser.parse_args()

    evaluate(args.model, args.dataset, args.data, args.split, args.device, args.max_images)


if __name__ == '__main__':
    main()
