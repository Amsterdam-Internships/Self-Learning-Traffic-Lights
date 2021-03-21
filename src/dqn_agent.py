import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.optim as optim

from src.dqn_model import QNetwork

"""
This file contains the Deep Q-leaning agent in PyTorch.
It learns by updating the parameters of its neural network by backpropagation,
by taking samples from the replay memory filled by the training loop in dqn_train.py.
Soft-updates are used to update the target network every training iteration.

Source: https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda
"""

GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_decay = 0.999  # learning rate decay
LR_STEP_TIMES = 500  # how often learning rate decays
FREEZE_TARGET = 10000  # how often to replace the target network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from environment."""

    def __init__(self, state_size, action_size, seed, lr=1e-3, batch_size=128, rm_size=36000, learn_every=4):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            lr (float): start learning rate
            batch_size (int): batch size
        """

        self.state_size = state_size
        self.action_size = action_size

        # The Q-Networks to be trained.
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=LR, momentum=0.9)
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=LR_STEP_TIMES, gamma=0.3)

        self.memory = ReplayBuffer(action_size, rm_size, batch_size)

        # Initialize time steps (for updating every UPDATE_EVERY and FREEZE_TARGET steps)
        self.train_step = 0
        self.update_step = 0

        self.loss = 0
        self.training_step = 0
        self.acting_step = 0
        self.batch_size = batch_size
        self.learn_every = learn_every

    def step(self, state, action, reward, next_step):

        # Save experience in replay memory.
        self.memory.add(state, action, reward, next_step)

        # Learn every UPDATE_EVERY time steps.
        self.train_step = (self.train_step + 1) % self.learn_every
        if self.train_step == 0:

            # If enough samples are available in memory, get random subset and learn.
            if len(self.memory) > self.batch_size:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def act(self, state, eps=0):
        """Returns action for given state as per current policy.

        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.FloatTensor(state).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy()), np.amax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size)), None

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s') tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_state = experiences

        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        # Shape of output from the model is (batch_size, action_dim).
        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            # .detach() ->  Returns a new Tensor, detached from the current graph.
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (gamma * labels_next)

        loss = criterion(predicted_targets, labels).to(device)
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.qnetwork_local.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target network.
        self.soft_update(TAU)

        # # Update target network every FREEZE_TARGET time steps.
        # self.update_step = (self.update_step + 1) % FREEZE_TARGET
        # if self.update_step == 0:
        #     self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        #     print("UPDATE TARGET NETWORK")

    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target

        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(self.qnetwork_target.parameters(),
                                             self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed -size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state"])

    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        experience = self.experiences(state, action, reward, next_state)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.FloatTensor([e.state for e in experiences if e is not None]).to(device)
        actions = torch.LongTensor([[e.action] for e in experiences if e is not None]).to(device)
        rewards = torch.FloatTensor([[e.reward] for e in experiences if e is not None]).to(device)
        next_states = torch.FloatTensor([e.next_state for e in experiences if e is not None]).to(device)

        return states, actions, rewards, next_states

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
