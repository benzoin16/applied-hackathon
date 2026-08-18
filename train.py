#!/usr/bin/env python3
"""Training script for the Custom Siamese SEM Localization Network."""

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import numpy as np

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class WaferDataset(Dataset):
    def __init__(self, manifest_path, img_size=(256, 256)):
        self.df = pd.read_csv(manifest_path)
        self.img_size = img_size  # (height, width)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load reference and search images as grayscale
        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            raise FileNotFoundError(f"Could not load images for index {idx}: {row['reference_path']} or {row['search_path']}")

        orig_h, orig_w = search_img.shape[:2]
        target_h, target_w = self.img_size
        
        # 2. Resize images to model input size (OpenCV resize expects width, height)
        if ref_img.shape[:2] != (target_h, target_w):
            ref_img = cv2.resize(ref_img, (target_w, target_h))
        if search_img.shape[:2] != (target_h, target_w):
            search_img = cv2.resize(search_img, (target_w, target_h))
            
        # 3. Scale ground truth coordinates from original image space to model input space
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        scaled_gt_x = row['gt_x'] * scale_x
        scaled_gt_y = row['gt_y'] * scale_y

        # 4. Generate target Gaussian heatmap centered at scaled GT coordinates
        # ResNet-18 backbone (conv1 to layer2) downsamples spatial dimensions by a factor of 8
        out_h, out_w = target_h // 8, target_w // 8
        gt_hx = (scaled_gt_x / target_w) * out_w
        gt_hy = (scaled_gt_y / target_h) * out_h
        
        yy, xx = torch.meshgrid(torch.arange(out_h), torch.arange(out_w), indexing='ij')
        sigma = 2.0
        target_heatmap = torch.exp(-((xx - gt_hx)**2 + (yy - gt_hy)**2) / (2 * sigma**2))
        
        # 5. Convert to PyTorch tensors and normalize to [0, 1]
        ref_tensor = torch.from_numpy(ref_img).float().unsqueeze(0) / 255.0
        search_tensor = torch.from_numpy(search_img).float().unsqueeze(0) / 255.0
        
        # If your ResNet backbone expects 3 channels (due to pretrained ImageNet weights):
        if ref_tensor.shape[0] == 1:
            ref_tensor = ref_tensor.repeat(3, 1, 1)
            search_tensor = search_tensor.repeat(3, 1, 1)

        # Return images and the target heatmap with channel dimension [1, out_h, out_w]
        return ref_tensor, search_tensor, target_heatmap.unsqueeze(0)

class CustomSiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        
        # FIX: Change input channels from 512 to 128 (since layer2 outputs 128 channels)
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
        heatmap = self.head(fused)
        return heatmap


class CenterNetWeightedMSELoss(nn.Module):
    """Downweights easy background negatives, upweights peaks and near-peak pixels."""
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_heatmap, target_heatmap):
        weights = torch.where(target_heatmap > 0.01, self.alpha, self.beta)
        loss = weights * (pred_heatmap - target_heatmap) ** 2
        return loss.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="./output/train_manifest.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = WaferDataset(args.manifest)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = CustomSiameseNetwork().to(device)
    criterion = CenterNetWeightedMSELoss(alpha=10.0, beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for ref, search, target in dataloader:
            ref, search, target = ref.to(device), search.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(ref, search)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {epoch_loss:.6f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"siamese_wafer_epoch_{epoch+1}.pth")

    print("Training complete. Weights saved.")

if __name__ == "__main__":
    main()