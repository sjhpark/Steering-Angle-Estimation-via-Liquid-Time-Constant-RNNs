from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse

from utils import DrivingDataset, augment
from models import SimpleCNN
from ncps.wirings import AutoNCP 
from ncps.torch import LTC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
    parser.add_argument("--N", type=int, default=10, help="image sequence length")
    parser.add_argument("--units", type=int, default=16, help="number of units in NCP")
    parser.add_argument("--log_freq", type=int, default=100, help="logging frequency")
    args = parser.parse_args()

    # Hyperparameters
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    N = args.N
    units = args.units
    log_freq = args.log_freq
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    augmentation = augment()
    train_dataloader = DataLoader(DrivingDataset(N, augmentation, "train"), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(DrivingDataset(N, augmentation, "test"), batch_size=batch_size, shuffle=True)

    # Feature Extractor
    feature_extractor = SimpleCNN()

    # LTC in NCP
    in_features = 1 # 1 type of in_features: extracted features from CNN
    out_features = 1 # 1 type of out_features: steering angle
    wiring = AutoNCP(units, out_features)  # 16 units, 1 motor neuron
    ltc_model = LTC(in_features, wiring, batch_first=True)

    # Criterion and Optimizer
    criterion = nn.MSELoss()
    ltc_optimizer = torch.optim.Adam(ltc_model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        # training
        ltc_model.train()
        loss = 0.0
        count = 0
        for i, (img, angle) in tqdm(enumerate(train_dataloader), desc="Training...", total=len(train_dataloader)):
            img = img.to(device) # (B,N,C,H,W)
            C, H, W = img.shape[-3:]
            img = img.view(-1, C, H, W) # (B*N,C,H,W)

            angle = angle.to(device) # (B,N)
            angle = angle.view(-1, 1) # (B*N,1)

            # Feature Extraction
            features = feature_extractor(img) # (B*N,1)
            features = features.view(-1, N, 1) # (B,N,1)

            # NCP
            ltc_optimizer.zero_grad()
            pred_angle, hx = ltc_model(features) # (B,N,1)
            pred_angle = pred_angle.view(-1, 1) # (B*N,1)

            loss = criterion(pred_angle, angle)
            loss.backward()
            ltc_optimizer.step()
            loss += loss.item()
            count += 1
            if i % log_freq == 0:
                print(f"Epoch {epoch}, Training Loss {loss / count:.4f}")
        # validation
        ltc_model.eval()
        loss = 0.0
        count = 0
        with torch.no_grad():
            for i, (img, angle) in tqdm(enumerate(test_dataloader), desc="Testing...", total=len(test_dataloader)):
                img = img.to(device)
                img = img.view(-1, C, H, W)
                angle = angle.to(device)
                angle = angle.view(-1, 1)

                features = feature_extractor(img) # (B*N,1)
                features = features.view(-1, N, 1) # (B,N,1)

                pred_angle, hx = ltc_model(features) # (B,N,1)
                pred_angle = pred_angle.view(-1, 1) # (B*N,1)
                
                loss = criterion(pred_angle, angle)
                loss += loss.item()
                count += 1
        print(f"Epoch {epoch}, Testing Loss {loss / count:.4f}")
