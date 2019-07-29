import torch
import os
import sys
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils import data
import itertools
import argparse
from model_ddpg import DuelNet,DuelNet_1
from config import args


# trajectory_dir = "traj/explore/trajectory/"
# data_dir = "traj/"
# action_space = 10
#
# # size = 354460
# # rec_type = np.dtype([('fr', np.int64),
# #                          ('r', np.float32), ('acc', np.float32), ('t', np.float32), ('pi', np.float32, action_space),
# #                          ('pi_explore', np.float32, action_space), ('traj', np.int64), ('ep', np.int64)])
# #
# # replay_buffer = np.array([], dtype=rec_type)
# # replay = np.concatenate([np.load(os.path.join(trajectory_dir, "%d.npy" % traj)) for traj in range(size)], axis=0)
# # replay_buffer = np.concatenate([replay_buffer, replay], axis=0)
# #
# # a = np.zeros(354460,10)
# #
# # np.save(os.path.join(os.path.join(data_dir, "pi.npy")), replay_buffer['pi'])
# # np.save(os.path.join(os.path.join(data_dir, "pi_explore.npy")), replay_buffer['pi_explore'])
# # np.save(os.path.join(os.path.join(data_dir, "r.npy")), replay_buffer['r'])
#
# pi_explore = np.load(os.path.join(os.path.join(data_dir, "pi_explore.npy")))
# r = np.load(os.path.join(os.path.join(data_dir, "r.npy")))
#
# pi_explore = torch.Tensor(pi_explore)
# r = torch.Tensor(r)
#
# d = list(zip(pi_explore, r))
#
# train = d[:-10000]
# test = d[-10000:]
#
# test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=64, shuffle=False)
# train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=64, shuffle=True)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, pi, reward):
        'Initialization'
        self.pi = pi
        self.reward = reward

    def __len__(self):
        'Denotes the total number of samples'
        return self.pi.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.pi[index]
        y = self.reward[index]
        return x, y


def train(model, device, train_loader, optimizer, args):
    model.train()
    loss_val = 0

    if args.metric == 'L1':
        loss_func = nn.L1Loss(reduction='none')
    elif args.metric == 'MSE':
        loss_func = nn.MSELoss(reduction='none')
    elif args.metric == 'SMOOTH':
        loss_func = nn.SmoothL1Loss(reduction='none')
    else:
        raise NotImplementedError


    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        value = model(data).view(-1)

        loss = loss_func(value, target)
        loss_val += loss.sum().item()
        loss.mean().backward()
        optimizer.step()

    train_len = len(train_loader.dataset)
    loss_val /= train_len
    return loss_val
    #print("TrainLoss %.4f" % loss_val)


def test(model, device, test_loader, args):
    model.eval()
    loss_val = 0

    if args.metric == 'L1':
        loss_func = nn.L1Loss(reduction='none')
    elif args.metric == 'MSE':
        loss_func = nn.MSELoss(reduction='none')
    elif args.metric == 'SMOOTH':
        loss_func = nn.SmoothL1Loss(reduction='none')
    else:
        raise NotImplementedError


    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            value = model(data).view(-1)

            loss = loss_func(value, target)

            loss_val += loss.sum().item()

    test_len = len(test_loader.dataset)
    loss_val /= test_len
    return loss_val
    #print("TestLoss %.4f" % loss_val)


def main():

    data_dir = "../traj/"

    pi_explore = np.load(os.path.join(os.path.join(data_dir, "pi_explore.npy")))
    r = np.load(os.path.join(os.path.join(data_dir, "r.npy")))

    pi_explore = torch.Tensor(pi_explore)
    r = torch.Tensor(r)

    train_set = Dataset(pi_explore[:-10000], r[:-10000])
    test_set = Dataset(pi_explore[-10000:], r[-10000:])

    device = torch.device("cuda")

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1024, num_workers=96, shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1024, num_workers=96, shuffle=True)


    if args.architecture == 'SAME':
        value_net = DuelNet().to(device)
    elif args.architecture == 'BIGGER':
        value_net = DuelNet_1().to(device)
    else:
        raise NotImplementedError

    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=0.0001)

    prev_loss = sys.maxsize
    for epoch in (itertools.count()):
    #for epoch in range(2):
        train_loss_val = train(value_net, device, train_loader, optimizer_value, args)
        test_loss_val = test(value_net, device, test_loader, args)
        print("TrainLoss %.4f TestLoss %.4f" % (train_loss_val, test_loss_val))
        if test_loss_val < prev_loss:
            prev_loss = test_loss_val
        elif epoch > 10:
            print("finished after %d epochs" % epoch)
            break


if __name__ == '__main__':
    main()
