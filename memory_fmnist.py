from torchvision import datasets, transforms
import torch
import random
import numpy as np
import os
from config import consts, args
import pwd

class Memory(torch.utils.data.Dataset):

    def __init__(self, benchmark):
        super(Memory, self).__init__()

        self.rawdata = consts.rawdata

        if benchmark == 'fmnist':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

            data_set = datasets.FashionMNIST(root=self.rawdata, train=True, transform=transform, download=False)

        elif benchmark == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

            data_set = datasets.MNIST(root=self.rawdata, train=True, transform=transform, download=False)

        elif benchmark == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))])

            data_set = datasets.CIFAR10(root=self.rawdata, train=True, transform=transform, download=False)
        else:
            assert False, "no valid benchmark"

        y = [i[1] for i in data_set]
        data_dict = {i: list() for i in range(consts.action_space)}
        for i in range(len(y)):
            data_dict[y[i].item()].append(data_set[i])

        self.data_dict = data_dict
        self.len = np.zeros(consts.action_space)
        #s_len = 100
        for i in range(consts.action_space):
            self.len[i] = len(self.data_dict[i])
            #self.len[i] = s_len
            #s_len += 50

        #self.len = len(self.data_dict[0])

    def __len__(self):
        return self.len[0]

    def __getitem__(self, label):
        assert (label < consts.action_space), "assert sample in DummyGen"
        i = random.randint(0, self.len[label]-1)
        return self.data_dict[label][i]

    def get_item(self, action_batch):
        assert (action_batch.max() < consts.action_space), "assert sample in DummyGen"

        gen_pic = list()
        for i in range(len(action_batch)):
            gen_pic.append(self.__getitem__(action_batch[i]))

        data, label = zip(*gen_pic)
        return torch.stack(data), torch.stack(label)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton_Mem(metaclass=Singleton):
    def __init__(self, benchmark):
        self.memory = Memory(benchmark)

    def get_item(self,action_batch):

        return self.memory.get_item(action_batch)

