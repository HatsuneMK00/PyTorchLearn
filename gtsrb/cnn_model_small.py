# -*- coding: utf-8 -*-
# created by makise, 2022/2/22

"""
Define a smaller CNN model for the GTSRB dataset using pytorch without dropout layer.
"""

import torch.nn as nn
import torch.nn.functional as F

# define the output size of the network
OUTPUT_DIM = 43

# define the small CNN model for the GTSRB dataset
class SmallCNNModel(nn.Module):
    def __init__(self):
        super(SmallCNNModel, self).__init__()
        self.output_dim = OUTPUT_DIM
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # define the linear layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten the input
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
