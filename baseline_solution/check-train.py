#!/usr/bin/env python3
import cv2
import torch
import pandas as pd
import numpy as np
from infer import SiameseTracker, predict_single_pair

def main():
    manifest_path = "./output/train_manifest.csv"
    weights_path = "siamese_wafer_epoch_10.pth"
    input_size = 256
    
    df = pd.read_csv(manifest_path)
    if len(df) == 0:
        print("Manifest is empty!")
        return

    # Grab the very first training sample
    row = df.iloc[0]
    ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)

    orig_h, orig_w = search_img.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseTracker().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Run prediction
    pred_x, pred_y = predict_single_pair(ref_img, search_img, model, device, (input_size, input_size))

    # Scale back to original image space
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size
    pred_x_orig = pred_x * scale_x
    pred_y_orig = pred_y * scale_y

    gt_x, gt_y = row['gt_x'], row['gt_y']
    error = np.sqrt((pred_x_orig - gt_x)**2 + (pred_y_orig - gt_y)**2)

    print("=" * 50)
    print(f"Training Sample 0 Diagnostics:")
    print(f"Ground Truth (Original): X={gt_x:.2f}, Y={gt_y:.2f}")
    print(f"Predicted (Original):    X={pred_x_orig:.2f}, Y={pred_y_orig:.2f}")
    print(f"Absolute Error:          {error:.3f} pixels")
    print("=" * 50)

    if error > 10.0:
        print("-> Result: WILDLY WRONG on a training sample.")
        print("-> Conclusion: The bug is in the target coordinate math/offset generation during training.")
    else:
        print("-> Result: CORRECT on a training sample!")
        print("-> Conclusion: Training target math is correct. Any eval errors are due to split differences or evaluation coordinate mapping.")

if __name__ == "__main__":
    main()