import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

def data_process():
    """Dataset is from https://github.com/SullyChen/driving-datasets"""
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
    def __init__(self, N=10, transform=transforms.ToTensor(), flag="train"):
        self.img_dir = "dataset/data"
        self.flag = flag
        if flag == "train":
            self.data_dir = "dataset/processed/train.parquet"
        elif flag == "test":
            self.data_dir = "dataset/processed/test.parquet"
        else:
            raise ValueError("flag must be either 'train' or 'test'")
        self.data = pq.read_table(self.data_dir).to_pandas() # data (image names & steering angles)

        self.transform = transform # image augmentation

        self.N = N # sequence length
        self.total_sequences = len(self.data["image"]) - N + 1 # total number of sequences

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        img_sequence = []
        angle_sequence = []
        for i in range(self.N):
            img_name = self.data["image"].iloc[idx + i]
            img = Image.open(os.path.join(self.img_dir, img_name)) # image
            if self.flag == "train":
                img = self.transform(img)
            elif self.flag == "test":
                img = transforms.ToTensor()(img)
            img_sequence.append(img)

            angle = self.data["angle"].iloc[idx + i] # steering angle
            angle = torch.tensor(float(angle))
            angle_sequence.append(angle)

        img_sequence = torch.stack(img_sequence)
        angle_sequence = torch.stack(angle_sequence)
        return img_sequence, angle_sequence

# data augmentation using torchvision.transforms
def augment():
    """Data Augmentation on PIL image"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # TODO: Find mean & std dev of training dataset; torchvision.transforms.Normalize doesn't support PIL.Image, so has to be done after transform.ToTensor()
        ])
    return transform