import torch
from torch import nn
import numpy

class FoodConvNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 12, 11) # Data has 3 channels (RGB), 12 output features, 10x10 kernel
        # 224 pixels - 11 pixels = 213 + 1 = 214. new shape = [12, 214, 214]
        self.pool = nn.MaxPool2d(2, 2) # takes 2x2 area and extracts it into 1 pixel (which will divide the dimensions of the image by 2: [12, 107, 107]

        self.conv2 = nn.Conv2d(12, 24, 10) # (107-10)+1 = (24, 98, 98) --> maxPooling: [24, 49, 49] --> flatten: [24 * 49 * 49]

        self.fullyConnected1 = nn.Linear(24 * 49 * 49, 120)
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