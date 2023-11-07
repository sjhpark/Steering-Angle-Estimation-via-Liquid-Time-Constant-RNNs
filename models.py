import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.wirings import AutoNCP 
from ncps.torch import LTC
import os
import seaborn as sns
import matplotlib.pyplot as plt

# simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 110, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 110)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Network(nn.Module):
    def __init__(self, wiring):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 110, 120)
        self.wiring = wiring

    def forward(self, x, batch_size, N):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 110)
        x = torch.relu(self.fc1(x)) # (B*N,120)
        x = x.view(batch_size, N, -1) # (B,N,120)
        
        # LTC
        device = x.device
        in_features = 1
        ltc_model = LTC(in_features, self.wiring, batch_first=True).to(device)

        # save plot if plot does not exist
        save_dir = "out"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, "wiring.png")):
            sns.set_style("white")
            plt.figure(figsize=(6, 4))
            legend_handles = self.wiring.draw_graph(draw_labels=True,  neuron_colors={"command": "tab:cyan"})
            plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "wiring.png"), dpi=300)
            plt.close()

        x = ltc_model(x) # (B,N,1)
        return x