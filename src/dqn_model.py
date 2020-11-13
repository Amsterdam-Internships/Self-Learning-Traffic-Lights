import torch
import torch.nn as nn
import torch.nn.functional as f

"""
This file sets up the neural network of the DQN agent in PyTorch.

Source: https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda
"""


class QNetwork(nn.Module):
    """ Actor (Policy) Model."""

    # TODO input list of sizes instead of fixed 2 layers
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
        self.batch_norm0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.batch_norm1 = nn.BatchNorm1d(fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.batch_norm2 = nn.BatchNorm1d(fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, x):
        """
        Build a network that maps state -> action values.
        """

        # x = self.batch_norm0(x)
        # x = self.batch_norm1(f.relu(self.fc1(x)))
        # x = self.batch_norm2(f.relu(self.fc2(x)))
        # return self.fc3(x)

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.fc3(x)
