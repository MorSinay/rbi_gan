from torchvision import datasets, transforms
import torch
import random
import numpy as np
import os
from config import consts, args, DirsAndLocksSingleton, lock_file, release_file

class Memory(torch.utils.data.Dataset):

    def __init__(self):
        super(Memory, self).__init__()
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        rawdata = os.path.join('/dev/shm/', 'elkayam', 'fmnist')

        mnist_data = datasets.FashionMNIST(root=rawdata,
                                           train=True,
                                           transform=transform,
                                           download=True)

        y = [i[1] for i in mnist_data]
        mnist_dict = {i: list() for i in range(consts.action_space)}
        for i in range(len(y)):
            mnist_dict[y[i].item()].append(mnist_data[i])

        self.data_dict = mnist_dict
        self.len = np.zeros(consts.action_space)
        s_len = 1000
        for i in range(consts.action_space):
            self.len[i] = s_len
            s_len += 500

        #self.len = len(self.data_dict[0])

    def __len__(self):
        return self.len[0]

    def __getitem__(self, label):
        assert (label < 10), "assert sample in DummyGen"
        i = random.randint(0, self.len[label]-1)
        return self.data_dict[label][i]

    def get_item(self, action_batch):
        assert (action_batch.max() < 10), "assert sample in DummyGen"

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
    def __init__(self):
        self.memory = Memory()

    def get_item(self,action_batch):

        return self.memory.get_item(action_batch)

