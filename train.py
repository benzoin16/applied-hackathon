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

class WaferDataset(Dataset):
    def __init__(self, manifest_path, img_size=(256, 256)):
        self.df = pd.read_csv(manifest_path)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load grayscale SEM images
        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
        
        # Resize if necessary
        if ref_img.shape != self.img_size:
            ref_img = cv2.resize(ref_img, self.img_size)
        if search_img.shape != self.img_size:
            search_img = cv2.resize(search_img, self.img_size)
            
        # Convert grayscale to 3-channel for ResNet backbone
        ref_img = np.stack([ref_img, ref_img, ref_img], axis=-1)
        search_img = np.stack([search_img, search_img, search_img], axis=-1)
        
        # To tensor & normalize to [0, 1]
        ref_tensor = torch.from_numpy(ref_img).permute(2, 0, 1).float() / 255.0
        search_tensor = torch.from_numpy(search_img).permute(2, 0, 1).float() / 255.0
        
        # Generate target Gaussian heatmap centered at GT coordinates
        # (Assuming output feature map is stride 8 downscaled)
        out_h, out_w = self.img_size[0] // 8, self.img_size[1] // 8
        target_heatmap = torch.zeros((out_h, out_w), dtype=torch.float32)
        
        # Scale GT coordinates to heatmap space
        gt_hx = (row['gt_x'] / self.img_size[1]) * out_w
        gt_hy = (row['gt_y'] / self.img_size[0]) * out_h
        
        # Populate Gaussian peak
        yy, xx = torch.meshgrid(torch.arange(out_h), torch.arange(out_w), indexing='ij')
        sigma = 2.0
        target_heatmap = torch.exp(-((xx - gt_hx)**2 + (yy - gt_hy)**2) / (2 * sigma**2))

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