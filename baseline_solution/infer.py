import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
import argparse

class SiameseTracker(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize ResNet without pre-trained weights (we will load our own)
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:6])
        self.backbone.eval()

    def forward(self, template, search):
        z = self.backbone(template)  
        x = self.backbone(search)    
        
        # L2 Normalize to bound cross-correlation to [-1, 1]
        z = F.normalize(z, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        
        heatmap = F.conv2d(x, z)     
        return heatmap

def predict_single_pair(ref_img: np.ndarray, search_img: np.ndarray, model: nn.Module, device: torch.device) -> tuple[float, float]:
    ref_10x = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    
    ref_rgb = cv2.cvtColor(ref_10x, cv2.COLOR_GRAY2RGB)
    search_rgb = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)
    
    ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).float()[None].to(device) / 255.0
    search_tensor = torch.from_numpy(search_rgb).permute(2, 0, 1).float()[None].to(device) / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    ref_tensor = (ref_tensor - mean) / std
    search_tensor = (search_tensor - mean) / std
    
    with torch.inference_mode():
        heatmap = model(ref_tensor, search_tensor)
        
    heatmap_np = heatmap.squeeze().cpu().numpy()
    _, _, _, max_loc = cv2.minMaxLoc(heatmap_np)
    
    stride = 8.0
    pred_x = (max_loc[0] * stride) + 50.0
    pred_y = (max_loc[1] * stride) + 50.0
    
    return float(pred_x), float(pred_y)

def evaluate_siamese_pipeline(manifest_csv: str, weights_path: str, tolerance_px: float = 4.0):
    df = pd.read_csv(manifest_csv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading Custom Siamese Weights from [{weights_path}] to [{device}]...")
    model = SiameseTracker().to(device)
    
    # Load custom trained weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
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
    print(f"Custom Siamese Accuracy (<= {tolerance_px}px): {accuracy:.2f}%")
    print(f"Mean Navigation Error:             {mean_error:.3f} px")
    print(f"Average Inference Latency:          {avg_latency:.2f} ms")
    print("=" * 50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_root", default="../output/train/manifest.csv")
    ap.add_argument("--weights", default="siamese_wafer_epoch_10.pth")
    ap.add_argument("--tolerance", type=float, default=4.0)
    args = ap.parse_args()
    
    evaluate_siamese_pipeline(args.csv_root, args.weights, tolerance_px=args.tolerance)

if __name__ == "__main__":
    main()