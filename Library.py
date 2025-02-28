import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, minibatch):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, minibatch)
        self.fc2_mean = nn.Linear(minibatch, action_dim)
        self.fc2_logstd = nn.Linear(minibatch, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        log_std = self.fc2_logstd(x)
        std = torch.exp(log_std)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, minibatch):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, minibatch)
        self.fc2 = nn.Linear(minibatch, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x