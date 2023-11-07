import os
from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse

from utils import DrivingDataset, augment
from models import Network
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
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers for dataloader")
    args = parser.parse_args()

    # Hyperparameters
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    N = args.N
    units = args.units
    log_freq = args.log_freq
    num_workers = args.num_workers
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    augmentation = augment()
    train_dataloader = DataLoader(DrivingDataset(N, augmentation, "train"), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(DrivingDataset(N, augmentation, "test"), batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Wiring
    out_features = 1 # steering angle 
    wiring = AutoNCP(units, out_features)  # arguments: units, motor neurons

    # Network
    model = Network(wiring=wiring).to(device)

    # Criterion and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        # training
        model.train()
        loss = 0.0
        count = 0
        for i, (img, angle) in tqdm(enumerate(train_dataloader), desc="Training...", total=len(train_dataloader)):
            img = img.to(device) # (B,N,C,H,W)
            C, H, W = img.shape[-3:]
            img = img.view(-1, C, H, W) # (B*N,C,H,W)

            angle = angle.to(device) # (B,N)
            angle = angle.unsqueeze(-1) # (B,N,1)

            optimizer.zero_grad()
            pred_angle, hx = model(img, batch_size, N) # (B,N,1)

            loss = criterion(pred_angle, angle)
            loss.backward()
            optimizer.step()
            loss += loss.item()
            count += 1
            if i % log_freq == 0:
                print(f"Epoch {epoch}, Training Loss {loss / count:.4f}")
        
        # save model
        save_dir = "out"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        now = datetime.datetime.now()
        date = now.strftime("%Y%m%d")
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}_{date}.pth"))

        # validation
        model.eval()
        loss = 0.0
        count = 0
        with torch.no_grad():
            for i, (img, angle) in tqdm(enumerate(test_dataloader), desc="Testing...", total=len(test_dataloader)):
                img = img.to(device)
                img = img.view(-1, C, H, W)
                angle = angle.to(device)
                angle = angle.unsqueeze(-1)

                pred_angle, hx = model(img, batch_size, N)
                
                loss = criterion(pred_angle, angle)
                loss += loss.item()
                count += 1
        print(f"Epoch {epoch}, Testing Loss {loss / count:.4f}")
