#!/usr/bin/env python3
"""CLI to generate a Drift-Sense synthetic dataset split.

Example:
    python generate_dataset.py --num-samples 1000 --split train \
        --architectures dram_1x finfet_10nm --output-dir ./output --seed 42
"""

import argparse
import csv
import os

import cv2
import numpy as np

from src.pipeline import GenerationParams, generate_sample
from src.presets import PRESETS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-samples", type=int, default=1000, help="Number of synthetic sample pairs to generate")
    p.add_argument("--architectures", nargs="+", default=list(PRESETS.keys()), choices=list(PRESETS.keys()))
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--seed", type=int, default=42)
    
    # --- Physical Parameter Ranges (Base defaults for CLI overrides) ---
    p.add_argument("--beam-spot-size-nm", type=float, default=3.0)
    p.add_argument("--collapse-threshold-nm", type=float, default=GenerationParams.collapse_threshold_nm)
    p.add_argument("--dose-reference", type=float, default=1.0)
    p.add_argument("--dose-search", type=float, default=1.0)
    p.add_argument("--shear-amplitude-px", type=float, default=GenerationParams.shear_amplitude_px)
    p.add_argument("--drift-jitter-px", type=float, default=1.5)
    p.add_argument("--astigmatism-ratio", type=float, default=1.1)
    p.add_argument("--vignette-strength", type=float, default=GenerationParams.vignette_strength)
    p.add_argument("--gamma", type=float, default=GenerationParams.gamma)
    p.add_argument("--barrel-distortion-k", type=float, default=GenerationParams.barrel_distortion_k)
    p.add_argument("--charging-streak-prob", type=float, default=0.2)
    p.add_argument("--charging-streak-intensity", type=float, default=GenerationParams.charging_streak_intensity)
    p.add_argument("--speckle-sigma", type=float, default=GenerationParams.speckle_sigma)
    p.add_argument("--salt-pepper-prob", type=float, default=GenerationParams.salt_pepper_prob)
    p.add_argument("--linewidth-bias-nm", type=float, default=GenerationParams.linewidth_bias_nm)
    p.add_argument("--corner-rounding-px", type=float, default=5.0)
    p.add_argument("--mat-size-nm", type=float, default=GenerationParams.mat_size_nm)
    p.add_argument("--strip-width-nm", type=float, default=GenerationParams.strip_width_nm)
    p.add_argument("--boundary-bias", type=float, default=GenerationParams.boundary_bias)
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Setup directories
    split_dir = os.path.join(args.output_dir, args.split)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, f"{args.split}_manifest.csv")
    
    # Initialize random number generator with seed
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.num_samples} diverse samples for split [{args.split}]...")

    fieldnames = [
        "id", "reference_path", "search_path", "gt_x", "gt_y", 
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h", 
        "architecture", "beam_spot_size_nm", "collapse_threshold_nm", 
        "dose_reference", "dose_search", "shear_amplitude_px", 
        "drift_jitter_px", "astigmatism_ratio", "vignette_strength", 
        "gamma", "barrel_distortion_k", "charging_streak_prob", 
        "charging_streak_intensity", "speckle_sigma", "salt_pepper_prob", 
        "linewidth_bias_nm", "corner_rounding_px", "mat_size_nm", 
        "strip_width_nm", "boundary_bias", "seed"
    ]

    with open(manifest_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(args.num_samples):
            # --- Per-Iteration Dynamic Randomization ---
            # This guarantees high visual variation across your dataset
            params = GenerationParams(
                beam_spot_size_nm=float(rng.uniform(1.5, 5.0)),
                collapse_threshold_nm=float(args.collapse_threshold_nm),
                dose_reference=float(rng.uniform(0.5, 2.0)),
                dose_search=float(rng.uniform(0.4, 2.0)),
                shear_amplitude_px=float(rng.uniform(0.0, 4.0)),
                drift_jitter_px=float(rng.uniform(0.0, 5.0)),
                astigmatism_ratio=float(rng.uniform(1.0, 1.4)),
                vignette_strength=float(rng.uniform(0.0, 0.3)),
                gamma=float(rng.uniform(0.7, 1.3)),
                barrel_distortion_k=float(rng.uniform(-0.03, 0.03)),
                charging_streak_prob=float(rng.choice([0.0, 0.15, 0.3], p=[0.5, 0.3, 0.2])),
                charging_streak_intensity=float(rng.uniform(0.1, 0.6)),
                speckle_sigma=float(args.speckle_sigma),
                salt_pepper_prob=float(rng.uniform(0.0, 0.005)),
                linewidth_bias_nm=float(rng.uniform(-2.0, 2.0)),
                corner_rounding_px=float(rng.uniform(2.0, 8.0)),
                mat_size_nm=float(rng.choice([1000.0, 2000.0, 4000.0])),
                strip_width_nm=float(rng.uniform(50.0, 200.0)),
                boundary_bias=float(rng.uniform(0.1, 0.9)),
            )

            architecture = args.architectures[int(rng.integers(0, len(args.architectures)))]
            sample = generate_sample(architecture, rng, params)

            ref_path = os.path.join(ref_dir, f"{i:05d}.png")
            search_path = os.path.join(search_dir, f"{i:05d}.png")
            cv2.imwrite(ref_path, sample["reference_img"])
            cv2.imwrite(search_path, sample["search_img"])

            gx0, gy0, gw, gh = sample["gt_box"]
            
            # Map parameters object back to dictionary for CSV writing
            param_dict = {field: getattr(params, field) for field in params.__dataclass_fields__ if field in fieldnames}

            writer.writerow({
                "id": i,
                "reference_path": ref_path,
                "search_path": search_path,
                "gt_x": sample["gt_x"],
                "gt_y": sample["gt_y"],
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "architecture": architecture,
                **param_dict,
                "seed": args.seed,
            })
            
            if (i + 1) % 100 == 0 or (i + 1) == args.num_samples:
                print(f"Progress: [{i + 1}/{args.num_samples}] samples generated.")

    print(f"\nDataset generation complete! Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()