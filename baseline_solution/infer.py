import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
import time
import argparse

class SiameseTracker(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pre-trained ResNet18 as the twin feature extractor
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Extract layers up to layer2 (Spatial downsample factor of 8)
        # We don't want to go deeper, otherwise we lose too much spatial resolution
        self.backbone = nn.Sequential(*list(resnet.children())[:6])
        
        # Freeze for zero-shot inference (Change to True if you write a training loop)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, template, search):
        # Extract deep feature maps
        z = self.backbone(template)  # Shape: [1, C, H_z, W_z]
        x = self.backbone(search)    # Shape: [1, C, H_x, W_x]
        
        # Cross-Correlation: Slide the template feature map over the search feature map
        heatmap = F.conv2d(x, z)     # Shape: [1, 1, H_out, W_out]
        return heatmap

def predict_single_pair(ref_img: np.ndarray, search_img: np.ndarray, model: nn.Module, device: torch.device) -> tuple[float, float]:
    """Helper method for both evaluation and compare scripts."""
    # 1. Scale reference down 10x to exactly match search field physical scale (100x100)
    ref_10x = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    
    # 2. Convert grayscale to RGB (ResNet expects 3 channels)
    ref_rgb = cv2.cvtColor(ref_10x, cv2.COLOR_GRAY2RGB)
    search_rgb = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)
    
    # 3. Convert to normalized tensors
    ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).float()[None].to(device) / 255.0
    search_tensor = torch.from_numpy(search_rgb).permute(2, 0, 1).float()[None].to(device) / 255.0
    
    # 4. ImageNet Standardization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    ref_tensor = (ref_tensor - mean) / std
    search_tensor = (search_tensor - mean) / std
    
    # 5. Get location heatmap
    with torch.inference_mode():
        heatmap = model(ref_tensor, search_tensor)
        
    # 6. Find peak coordinate in heatmap
    heatmap_np = heatmap.squeeze().cpu().numpy()
    _, _, _, max_loc = cv2.minMaxLoc(heatmap_np)
    
    # 7. Convert feature-map coordinates back to original image pixel coordinates
    # ResNet up to layer2 has a downsample stride of 8.
    stride = 8.0
    
    # max_loc gives the top-left corner of the match. 
    # Add 50 pixels (half of the 100x100 template) to target the exact center.
    pred_x = (max_loc[0] * stride) + 50.0
    pred_y = (max_loc[1] * stride) + 50.0
    
    return float(pred_x), float(pred_y)

def evaluate_siamese_pipeline(manifest_csv: str, tolerance_px: float = 4.0):
    df = pd.read_csv(manifest_csv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading Siamese Tracker on device [{device}]...")
    model = SiameseTracker().to(device)
    
    success_count = 0
    total_time_ms = 0.0
    errors = []
    
    for idx, row in df.iterrows():
        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
        
        t0 = time.perf_counter()
        pred_x, pred_y = predict_single_pair(ref_img, search_img, model, device)
        t1 = time.perf_counter()
        
        total_time_ms += (t1 - t0) * 1000.0
        
        gt_x, gt_y = row['gt_x'], row['gt_y']
        
        error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        errors.append(error)
        
        if error <= tolerance_px:
            success_count += 1
            
    accuracy = (success_count / len(df)) * 100.0
    avg_latency = total_time_ms / len(df)
    mean_error = np.mean(errors)
    
    print("=" * 50)
    print(f"Siamese Accuracy (<= {tolerance_px}px): {accuracy:.2f}%")
    print(f"Mean Navigation Error:      {mean_error:.3f} px")
    print(f"Average Inference Latency:   {avg_latency:.2f} ms")
    print("=" * 50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_root", default="../output/train/manifest.csv")
    ap.add_argument("--tolerance", type=float, default=4.0)
    args = ap.parse_args()
    
    evaluate_siamese_pipeline(args.csv_root, tolerance_px=args.tolerance)

if __name__ == "__main__":
    main()