import torch
from torch import nn
import numpy

class FoodConvNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 12, 5) # Data has 3 channels (RGB), 12 output features, 5 kernel
        # 224 pixels - 5 pixels = 219 + 1 = 220. new shape = [12, 220, 220]
        self.pool = nn.MaxPool2d(2, 2) # takes 2x2 area and extracts it into 1 pixel (which will divide the dimensions of the image by 2: [12, 110, 110]

        self.conv2 = nn.Conv2d(12, 24, 5) # (110-5)+1 = (24, 106, 106) --> maxPooling: [24, 53, 53] --> flatten: [24 * 53 * 53]

        self.fullyConnected1 = nn.Linear(24 * 53 * 53, 120)
        self.fullyConnected2 = nn.Linear(120, 110)
        self.output = nn.Linear(110, 101)



    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fullyConnected1(x))
        x = nn.functional.relu(self.fullyConnected2(x))
        x = self.output(x)

        return x