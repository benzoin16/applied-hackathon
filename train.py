import argparse
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision.models as models

# ---------------------------------------------------------
# 1. Dataset & Ground Truth Heatmap Generator
# ---------------------------------------------------------
def create_gaussian_heatmap(target_size, center_x, center_y, sigma=2.0):
    w, h = target_size
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    y = y[:, np.newaxis]
    
    # Generate 2D Gaussian
    heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    return heatmap

class WaferSiameseTrainDataset(Dataset):
    def __init__(self, manifest_csv: str):
        self.df = pd.read_csv(manifest_csv)
        
        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
        
        # 1. Scale reference to 100x100 to match scale
        ref_10x = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        search_resized = cv2.resize(search_img, (1024, 1024), interpolation=cv2.INTER_AREA)
        
        # 2. Convert to RGB Tensors
        ref_rgb = cv2.cvtColor(ref_10x, cv2.COLOR_GRAY2RGB)
        search_rgb = cv2.cvtColor(search_resized, cv2.COLOR_GRAY2RGB)
        
        ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).float() / 255.0
        search_tensor = torch.from_numpy(search_rgb).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        ref_tensor = (ref_tensor - self.mean) / self.std
        search_tensor = (search_tensor - self.mean) / self.std
        
        # 3. Calculate GT Heatmap coordinates
        # Search scales from raw to 1024x1024
        sx = 1024.0 / search_img.shape[1]
        sy = 1024.0 / search_img.shape[0]
        gt_x_1024 = row['gt_x'] * sx
        gt_y_1024 = row['gt_y'] * sy
        
        # The feature map is downsampled by 8x (ResNet up to layer2)
        # Template anchors at its top-left, so we offset by 50 (half the 100x100 template)
        hm_x = (gt_x_1024 - 50.0) / 8.0
        hm_y = (gt_y_1024 - 50.0) / 8.0
        
        # The output heatmap dimension for a 1024 search and 100 template is 116x116
        heatmap = create_gaussian_heatmap((116, 116), hm_x, hm_y, sigma=2.0)
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0) # [1, 116, 116]
        
        return ref_tensor, search_tensor, heatmap_tensor

# ---------------------------------------------------------
# 2. Trainable Siamese Network
# ---------------------------------------------------------
class TrainableSiameseTracker(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Keep up to layer2 (stride of 8)
        self.backbone = nn.Sequential(*list(resnet.children())[:6])
        
        # Unfreeze all layers so we can fine-tune
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, template, search):
        z = self.backbone(template)
        x = self.backbone(search)
        
        # Batch cross-correlation
        b, c, h, w = x.shape
        heatmap = []
        for i in range(b):
            # cross-correlate each item in batch
            hm = F.conv2d(x[i].unsqueeze(0), z[i].unsqueeze(0))
            heatmap.append(hm)
            
        return torch.cat(heatmap, dim=0)

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def train(manifest_csv: str, epochs: int = 10, batch_size: int = 8, lr: float = 1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on {device}...")
    
    dataset = WaferSiameseTrainDataset(manifest_csv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = TrainableSiameseTracker().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Mean Squared Error for heatmap regression
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for batch_idx, (ref, search, gt_heatmap) in enumerate(dataloader):
            ref, search, gt_heatmap = ref.to(device), search.to(device), gt_heatmap.to(device)
            
            optimizer.zero_grad()
            
            # Predict heatmap
            pred_heatmap = model(ref, search)
            
            # Calculate loss against Gaussian GT
            loss = criterion(pred_heatmap, gt_heatmap)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f} | Time: {time.time()-t0:.1f}s")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"siamese_wafer_epoch_{epoch+1}.pth")
        
    print("Training Complete. Weights saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="../output/train/manifest.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args.manifest, args.epochs, args.batch_size, args.lr)