import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils import weight_norm

"""
This file sets up the neural network of the DQN agent in PyTorch.

Source: https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda
"""


class QNetwork(nn.Module):
    """ Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_unit=64,
                 fc2_unit=64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()  # calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        # self.fc1 = weight_norm(nn.Linear(state_size, fc1_unit), name='weight')
        # self.fc2 = weight_norm(nn.Linear(fc1_unit, fc2_unit), name='weight')
        # self.fc3 = weight_norm(nn.Linear(fc2_unit, action_size), name='weight')
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        # self.fc25 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, x):
        """
        Build a network that maps state -> action values.
        """
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        # x = f.relu(self.fc25(x))
        return self.fc3(x)
