#!/usr/bin/env python3
"""Inference and Evaluation script with empirical stride mapping & sub-pixel refinement."""

import argparse
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models


class CustomSiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        
        # FIX: Change input channels from 512 to 128 here as well
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # <--- Changed 512 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, ref, search):
        feat_ref = self.backbone(ref)
        feat_search = self.backbone(search)
        fused = torch.abs(feat_ref - feat_search)
        return self.head(fused)


def get_empirical_mapping(input_size=(256, 256)):
    """Dynamically computes exact feature map strides to avoid hardcoded offset errors."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone = torch.nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2
    ).eval()

    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        out = backbone(dummy_input)
        out_h, out_w = out.shape[2], out.shape[3]
       
    stride_y = input_size[0] / out_h
    stride_x = input_size[1] / out_w
    return stride_x, stride_y, (out_h, out_w)


def subpixel_refinement(heatmap, max_y, max_x):
    """Applies local center-of-mass adjustment for sub-pixel accuracy."""
    h, w = heatmap.shape
    if max_y <= 0 or max_y >= h - 1 or max_x <= 0 or max_x >= w - 1:
        return float(max_x), float(max_y)
    
    patch = heatmap[max_y-1:max_y+2, max_x-1:max_x+2]
    patch_sum = patch.sum()
    if patch_sum == 0:
        return float(max_x), float(max_y)
        
    dy = (torch.sum(torch.arange(-1, 2, device=patch.device) * patch.sum(dim=1)) / patch_sum).item()
    dx = (torch.sum(torch.arange(-1, 2, device=patch.device) * patch.sum(dim=0)) / patch_sum).item()
    
    return max_x + dx, max_y + dy


def predict_single_pair(ref_img, search_img, model, device, input_size=(256, 256)):
    # Preprocess images (Grayscale -> 3-Channel -> Normalization)
    if ref_img.shape != input_size:
        ref_img = cv2.resize(ref_img, input_size)
    if search_img.shape != input_size:
        search_img = cv2.resize(search_img, input_size)

    ref_img = np.stack([ref_img, ref_img, ref_img], axis=-1)
    search_img = np.stack([search_img, search_img, search_img], axis=-1)

    ref_tensor = torch.from_numpy(ref_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    search_tensor = torch.from_numpy(search_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    ref_tensor, search_tensor = ref_tensor.to(device), search_tensor.to(device)

    with torch.no_grad():
        heatmap = model(ref_tensor, search_tensor).squeeze()

    # Find peak activation
    max_val_flat = torch.argmax(heatmap)
    max_loc_y, max_loc_x = torch.unravel_index(max_val_flat, heatmap.shape)

    # Sub-pixel refinement
    refined_x, refined_y = subpixel_refinement(heatmap, max_loc_y.item(), max_loc_x.item())

    # Map back using empirical stride
    stride_x, stride_y, _ = get_empirical_mapping(input_size)
    pred_x = refined_x * stride_x
    pred_y = refined_y * stride_y

    return pred_x, pred_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="./output/train_manifest.csv")
    parser.add_argument("--weights", default="siamese_wafer_epoch_10.pth")
    parser.add_argument("--tolerance", type=float, default=4.0)
    parser.add_argument("--input-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading weights from [{args.weights}] to [{device}]...")

    model = CustomSiameseNetwork().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    df = pd.read_csv(args.manifest)
    errors = []
    success_count = 0
    total_time_ms = 0.0

    print(f"Evaluating {len(df)} samples...")
    for _, row in df.iterrows():
        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)

        t0 = time.perf_counter()
        pred_x, pred_y = predict_single_pair(ref_img, search_img, model, device, (args.input_size, args.input_size))
        t1 = time.perf_counter()

        total_time_ms += (t1 - t0) * 1000.0
        gt_x, gt_y = row['gt_x'], row['gt_y']

        error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        errors.append(error)

        if error <= args.tolerance:
            success_count += 1

    accuracy = (success_count / len(df)) * 100.0
    avg_latency = total_time_ms / len(df)
    mean_error = np.mean(errors)

    print("=" * 50)
    print(f"Custom Siamese Accuracy (<= {args.tolerance}px): {accuracy:.2f}%")
    print(f"Mean Navigation Error:             {mean_error:.3f} px")
    print(f"Average Inference Latency:          {avg_latency:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()