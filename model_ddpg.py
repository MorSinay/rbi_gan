import torch
from torch import nn
from config import consts
import torch.nn.functional as F


action_space = consts.action_space


class DuelNet(nn.Module):

    def __init__(self):

        super(DuelNet, self).__init__()

        self.fc1 = nn.Linear(action_space * action_space, 64)
        self.fc2 = nn.Linear(64 + action_space, 32)
        self.fc3 = nn.Linear(32, 1)

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, s, pi):
        s = s.view(-1, action_space*action_space)
        pi = pi.view(-1, action_space)
        x = F.relu(self.fc1(s))
        x = torch.cat([x, pi], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class BehavioralNet(nn.Module):

    def __init__(self):

        super(BehavioralNet, self).__init__()

        self.beta = nn.Sequential(
            nn.Linear(action_space * action_space, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_space),
        )

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, s):
        s = s.view(-1, action_space * action_space)
        beta = self.beta(s)

        return beta
