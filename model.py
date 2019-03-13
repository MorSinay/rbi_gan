import torch
from torch import nn
from config import consts, args
import numpy as np


action_space = consts.action_space


class DuelNet(nn.Module):

    def __init__(self):

        super(DuelNet, self).__init__()

        # q net
        self.fc_q = nn.Sequential(
            nn.Linear(action_space*action_space, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_space),
        )

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, s):
        s = s.view(-1, action_space*action_space)
        s = self.fc_q(s)

        return s


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
