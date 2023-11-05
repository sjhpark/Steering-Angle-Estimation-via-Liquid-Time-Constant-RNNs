import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

def data_process():
    datadir = "dataset"
    imgdir = os.path.join(datadir, "data")
    data = os.path.join(datadir, "data.txt")

    if not os.path.exists(os.path.join(datadir, "processed")):
        os.makedirs(os.path.join(datadir, "processed"))
    processdir = os.path.join(datadir, "processed")

    """Road View Image Data Statistics"""
    print("# of images: ", len(os.listdir(imgdir)))

    """Data (image names & steering angles) Preprocessing"""
    # convert txt to pandas dataframe
    data_df = pd.read_csv(data, sep=",", header=None, names=["image angle", "timestamp"])
    # drop timestamp column
    data_df = data_df.drop("timestamp", axis=1)
    # space split
    data_df = data_df["image angle"].apply(lambda x: pd.Series(x.split(" ")))
    # set header names
    data_df.columns = ["image", "angle"]

    """Train and Validation Data Split using sklearn"""
    # split train and validation data without shuffling
    train_df, test_df = train_test_split(data_df, test_size=0.2, shuffle=False)
    # convert to parquet table
    train_table = pa.Table.from_pandas(train_df)
    test_table = pa.Table.from_pandas(test_df)
    # Save as parquet file
    pq.write_table(train_table, os.path.join(processdir, "train.parquet"))
    pq.write_table(test_table, os.path.join(processdir, "test.parquet"))

class DrivingDataset(Dataset):
    def __init__(self, img_dir, data_dir):
        self.img_dir = img_dir
        self.data_dir = data_dir
        self.data = pq.read_table(data_dir).to_pandas()
        self.transforms = transforms.Compose([transforms.ToTensor()])
   
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data["image"].iloc[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)) # image
        img = self.transforms(img)
        angle = self.data["angle"].iloc[idx] # steering angle
        angle = torch.tensor(float(angle))
        return img, angle
