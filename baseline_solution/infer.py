import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import time
import argparse

class WaferLoFTRDataset(Dataset):
    def __init__(self, manifest_csv: str, target_size=(1024, 1024)):
        self.df = pd.read_csv(manifest_csv)
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load grayscale images
        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
        
        # Resize to 1024x1024
        ref_resized = cv2.resize(ref_img, self.target_size, interpolation=cv2.INTER_AREA)
        search_resized = cv2.resize(search_img, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to Tensors [1, 1, H, W] in [0, 1] range
        ref_tensor = torch.from_numpy(ref_resized).float()[None, None] / 255.0
        search_tensor = torch.from_numpy(search_resized).float()[None, None] / 255.0
        
        # Scale ground truth coordinates to match 1024x1024 frame
        sx = self.target_size[0] / 1000.0
        sy = self.target_size[1] / 1000.0
        
        return {
            'image0': ref_tensor,
            'image1': search_tensor,
            'gt_x': row['gt_x'] * sx,
            'gt_y': row['gt_y'] * sy,
            'scale': (sx, sy)
        }


import kornia.feature as KF

def run_loftr_matching(ref_tensor: torch.Tensor, search_tensor: torch.Tensor, matcher, confidence_thresh: float = 0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = matcher.to(device)
    
    input_dict = {
        'image0': ref_tensor.to(device),
        'image1': search_tensor.to(device)
    }
    
    with torch.inference_mode():
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    conf = correspondences['confidence'].cpu().numpy()
    
    # Filter keypoints by confidence threshold
    valid_mask = conf >= confidence_thresh
    return mkpts0[valid_mask], mkpts1[valid_mask]

def locate_target_center(mkpts0: np.ndarray, mkpts1: np.ndarray, ref_shape=(1024, 1024)) -> tuple[float, float]:
    if len(mkpts0) < 4:
        # Fallback to search center if insufficient matches
        return ref_shape[0] / 2.0, ref_shape[1] / 2.0
    
    # Robust homography estimation using MAGSAC++
    H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 3.0)
    
    if H is None:
        return ref_shape[0] / 2.0, ref_shape[1] / 2.0
        
    # Reference image center point [X, Y, 1]
    ref_center = np.array([ref_shape[1] / 2.0, ref_shape[0] / 2.0, 1.0], dtype=np.float32)
    
    # Transform center point to search image space
    pred_center = H @ ref_center
    pred_center /= pred_center[2]
    
    return float(pred_center[0]), float(pred_center[1])


def evaluate_loftr_pipeline(manifest_csv: str, tolerance_px: float = 2.0):
    dataset = WaferLoFTRDataset(manifest_csv)
    matcher = KF.LoFTR(pretrained="indoor_new")
    
    success_count = 0
    total_time_ms = 0.0
    errors = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        t0 = time.perf_counter()
        mkpts0, mkpts1 = run_loftr_matching(sample['image0'], sample['image1'], matcher)
        pred_x_1024, pred_y_1024 = locate_target_center(mkpts0, mkpts1)
        t1 = time.perf_counter()
        
        total_time_ms += (t1 - t0) * 1000.0
        
        # Convert predictions back to native 1000x1000 scale
        pred_x_1000 = pred_x_1024 / sample['scale'][0]
        pred_y_1000 = pred_y_1024 / sample['scale'][1]
        gt_x_1000 = sample['gt_x'] / sample['scale'][0]
        gt_y_1000 = sample['gt_y'] / sample['scale'][1]
        
        error = np.sqrt((pred_x_1000 - gt_x_1000)**2 + (pred_y_1000 - gt_y_1000)**2)
        errors.append(error)
        
        if error <= tolerance_px:
            success_count += 1
            
    accuracy = (success_count / len(dataset)) * 100.0
    avg_latency = total_time_ms / len(dataset)
    mean_error = np.mean(errors)
    
    print(f"LoFTR Accuracy (<= {tolerance_px}px): {accuracy:.2f}%")
    print(f"Mean Navigation Error: {mean_error:.3f} px")
    print(f"Average Inference Latency: {avg_latency:.2f} ms")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv_root", default="../output/train/manifest.csv")

    args = ap.parse_args()
    evaluate_loftr_pipeline(args.csv_root, tolerance_px=4.0)

if __name__ == "__main__":
    main()