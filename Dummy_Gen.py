from torchvision import datasets, transforms
import torch
import random
import numpy as np
import os


class DummyGen():
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor()  # ,
                                        # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                                        ])

        rawdata = os.path.join('/dev/shm/', 'elkayam', 'fmnist')

        mnist_data = datasets.FashionMNIST(root=rawdata,
                                           train=True,
                                           transform=transform,
                                           download=True)

        y = [i[1] for i in mnist_data]
        mnist_dict = {i: list() for i in range(10)}
        for i in range(len(y)):
            mnist_dict[y[i].item()].append(mnist_data[i])

        self.data_dict = mnist_dict

        random.seed(23)

    def sample(self, label):
        assert (label < 10), "assert sample in DummyGen"
        i = random.randint(0, len(self.data_dict[label]) - 1)
        return self.data_dict[label][i]

    def gen(self, action_batch):
        assert (action_batch.max() < 10), "assert sample in DummyGen"

        gen_pic = list()
        for i in range(len(action_batch)):
            gen_pic.append(self.sample(action_batch[i]))

        data, label = zip(*gen_pic)
        return torch.stack(data), torch.stack(label)
