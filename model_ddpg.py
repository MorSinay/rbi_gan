import torch
from torch import nn
from config import consts
import torch.nn.functional as F


action_space = consts.action_space


class DuelNet(nn.Module):

    def __init__(self):

        super(DuelNet, self).__init__()

        self.fc1 = nn.Linear(action_space, action_space)
        self.fc2 = nn.Linear(action_space, action_space)
        self.fc3 = nn.Linear(action_space, 1)

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, pi):
        pi = pi.view(-1, action_space)
        x = F.relu(self.fc1(pi))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class DuelNet_1(nn.Module):

    def __init__(self):

        super(DuelNet_1, self).__init__()

        self.fc1 = nn.Linear(action_space, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, pi):
        pi = pi.view(-1, action_space)

        x = F.relu(self.fc1(pi))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


