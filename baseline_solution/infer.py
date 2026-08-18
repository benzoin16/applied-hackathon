#!/usr/bin/env python3
import argparse
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models


class SiameseTracker(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
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


def locate_target_center_from_heatmap(heatmap_tensor):
    if heatmap_tensor.ndim == 4:
        heatmap_tensor = heatmap_tensor.squeeze()
        
    max_val = torch.max(heatmap_tensor)
    if max_val < 0.01:
        h, w = heatmap_tensor.shape
        return float(w / 2), float(h / 2)
        
    flat_idx = torch.argmax(heatmap_tensor)
    h, w = heatmap_tensor.shape
    cy = (flat_idx // w).float().item()
    cx = (flat_idx % w).float().item()
    return cx, cy


def predict_single_pair(ref_img, search_img, model, device, input_size=(256, 256)):
    h, w = input_size
    ref_resized = cv2.resize(ref_img, (w, h))
    search_resized = cv2.resize(search_img, (w, h))

    ref_tensor = torch.from_numpy(ref_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    search_tensor = torch.from_numpy(search_resized).float().unsqueeze(0).unsqueeze(0) / 255.0

    if ref_tensor.shape[1] == 1:
        ref_tensor = ref_tensor.repeat(1, 3, 1, 1)
        search_tensor = search_tensor.repeat(1, 3, 1, 1)

    ref_tensor = ref_tensor.to(device)
    search_tensor = search_tensor.to(device)

    model.eval()
    with torch.no_grad():
        heatmap = model(ref_tensor, search_tensor)
        
        # --- PUT YOUR TEST SNIPPET HERE ---
        print("min/max/std:", heatmap.min().item(), heatmap.max().item(), heatmap.std().item())
        # -----------------------------------

    cx_out, cy_out = locate_target_center_from_heatmap(heatmap)
    
    out_h, out_w = heatmap.shape[-2], heatmap.shape[-1]
    pred_x = (cx_out / out_w) * w
    pred_y = (cy_out / out_h) * h

    return pred_x, pred_y


def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom Siamese Network on SEM Dataset")
    parser.add_argument("--manifest", type=str, required=True, help="Path to evaluation manifest CSV")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--tolerance", type=float, default=4.0, help="Pixel error tolerance threshold")
    parser.add_argument("--input-size", type=int, default=256, help="Model input resolution")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading weights from [{args.weights}] to [{args.device}]...")
    model = SiameseTracker().to(args.device)
    state_dict = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    df = pd.read_csv(args.manifest)
    print(f"Evaluating {len(df)} samples...")

    success_count = 0
    total_time_ms = 0.0
    errors = []

    for _, row in df.iterrows():
        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)

        if ref_img is None or search_img is None:
            continue

        orig_h, orig_w = search_img.shape[:2]

        t0 = time.perf_counter()
        pred_x, pred_y = predict_single_pair(ref_img, search_img, model, args.device, (args.input_size, args.input_size))
        t1 = time.perf_counter()

        total_time_ms += (t1 - t0) * 1000.0

        scale_x = orig_w / args.input_size
        scale_y = orig_h / args.input_size
        pred_x_orig = pred_x * scale_x
        pred_y_orig = pred_y * scale_y

        gt_x, gt_y = row['gt_x'], row['gt_y']

        error = np.sqrt((pred_x_orig - gt_x)**2 + (pred_y_orig - gt_y)**2)
        errors.append(error)

        if error <= args.tolerance:
            success_count += 1

    accuracy = (success_count / len(df)) * 100.0 if len(df) > 0 else 0.0
    avg_latency = total_time_ms / len(df) if len(df) > 0 else 0.0
    mean_error = np.mean(errors) if len(errors) > 0 else 0.0

    print("=" * 50)
    print(f"Custom Siamese Accuracy (<= {args.tolerance}px): {accuracy:.2f}%")
    print(f"Mean Navigation Error:             {mean_error:.3f} px")
    print(f"Average Inference Latency:          {avg_latency:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()