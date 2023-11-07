import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.wirings import AutoNCP 
from ncps.torch import LTC

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
    def __init__(self, units):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 110, 120)
        self.units = units

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 110)
        x = torch.relu(self.fc1(x))

        self.in_features = x.shape[-1]
        self.out_features = 1 # steering angle 
        device = x.device
        wiring = AutoNCP(self.units, self.out_features)  # arguments: units, motor neurons
        ltc_model = LTC(self.in_features, wiring, batch_first=True).to(device)
        x = ltc_model(x)
        return x