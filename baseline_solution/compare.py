import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from infer import SiameseTracker, predict_single_pair


def visualize_dataset(manifest_csv: str, num_samples: int = 5, save_dir: str = "visualizations"):
    df = pd.read_csv(manifest_csv)
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Siamese Tracker on [{device}]...")
    model = SiameseTracker().to(device)

    # Pick random samples from manifest
    sample_indices = np.random.choice(len(df), size=min(num_samples, len(df)), replace=False)
    
    # 100x100 px reference footprint in search space (10x FOV scale)
    half_box = 50  

    for idx in sample_indices:
        row = df.iloc[idx]
        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)

        # Run Siamese prediction
        pred_x, pred_y = predict_single_pair(ref_img, search_img, model, device)

        gt_x, gt_y = row['gt_x'], row['gt_y']
        error_px = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

        # Draw overlays on RGB copy of search field
        search_rgb = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)

        # 1. Ground Truth (Green Box & Center Point)
        cv2.circle(search_rgb, (int(gt_x), int(gt_y)), 6, (0, 255, 0), -1)
        cv2.rectangle(
            search_rgb,
            (int(gt_x - half_box), int(gt_y - half_box)),
            (int(gt_x + half_box), int(gt_y + half_box)),
            color=(0, 255, 0),
            thickness=2
        )

        # 2. Model Prediction (Red Box & Center Point)
        cv2.circle(search_rgb, (int(pred_x), int(pred_y)), 4, (255, 0, 0), -1)
        cv2.rectangle(
            search_rgb,
            (int(pred_x - half_box), int(pred_y - half_box)),
            (int(pred_x + half_box), int(pred_y + half_box)),
            color=(255, 0, 0),
            thickness=2
        )

        # Plot Side-by-Side
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        axes[0].imshow(ref_img, cmap='gray')
        axes[0].set_title(f"Sample #{idx} | Reference Crop (1000x1000)")
        axes[0].axis('off')

        axes[1].imshow(search_rgb)
        axes[1].set_title(f"Search Field (Siamese) | Green=GT, Red=Pred | Error: {error_px:.2f} px")
        axes[1].axis('off')

        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"sample_{idx}_siamese.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
        plt.close()
        
        print(f"Saved visual comparison to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Prediction vs Ground Truth Overlays")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to render")
    parser.add_argument("--save_dir", type=str, default="visualizations")

    args = parser.parse_args()
    visualize_dataset(args.manifest, args.num_samples, args.save_dir)