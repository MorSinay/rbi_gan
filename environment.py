import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from memory_fmnist import Singleton_Mem
from Net import Net, ResNet_Cifar
import torch.optim as optim
import os
import random
import numpy as np
import collections

from config import consts, args

class Env(object):

    def __init__(self):
        self.memory = Singleton_Mem(args.benchmark)
        self.output_size = consts.action_space
        self.batch_size = args.env_batch_size
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = Model()
        self.acc = 0
        self.reward = None
        self.iterations = args.env_iterations
        self.t = 0

        self.test_func = self.model.test

        if args.acc == 'all':
            self.acc_func = self.acc_calc
        else:
            self.acc_func = self.label_calc

        self.reward_func = self.reward_final

        self.reset()

    def acc_calc(self, cm):
        return np.trace(cm)/np.sum(cm)

    def label_calc(self, cm):
        labels = [0,5,9]
        acc = 0.0
        for l in labels:
            acc += cm[l][l]/cm.sum(axis=1)[l]

        acc /= len(labels)

        return acc

    def reward_final(self, next_acc):
        self.reward = min(-np.log2(1-next_acc),10)

    def reset(self):
        #self.model.load_model()
        self.model.reset_model()
        self.acc = 0
        self.reward = 0

        self.t = 0

    def step_policy(self, policy):

        assert(policy.shape[1] == self.output_size), "action error"

        for _ in range(self.iterations):
            # TODO: need to check about the option of a sampler that sample from the policy distribution
            action_batch = np.random.choice(self.output_size, self.batch_size, p=policy[0])
            data_gen, label_gen = self.memory.get_item(action_batch)

            self.model.train_batch(data_gen, label_gen)

        cm = self.test_func()
        next_acc = self.acc_func(cm)
        self.reward_func(next_acc)
        self.acc = next_acc

        self.t = 1

class Model():
    def __init__(self):

        if args.benchmark == 'fmnist':
            self.data_func = datasets.FashionMNIST
            self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        elif args.benchmark == 'mnist':
            self.data_func = datasets.MNIST
            self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        elif args.benchmark == 'cifar10':
            self.data_func = datasets.CIFAR10
            self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))])
        else:
            assert False, "wrong benchmark"

        self.outputs = consts.action_space
        self.optimizer = None
        self.test_loader = None
        self.test_loader_batch = 256 #args.test_loader_batch_size
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.lr = args.model_lr
        self.betas = (args.model_beta1, args.model_beta2)
        self.model_name = "net.pkl"
        self.model_dir = consts.modeldir
        self.load_test_loader(None)

        self.create_model()
        #self.load_model()

    def train(self, train_loader):
        self.model.train()
        for batch_idx, (data, label) in tqdm(enumerate(tqdm(train_loader))):
            data, label = data.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, label)
            loss.backward()
            self.optimizer.step()

    def train_batch(self, data, label):
        self.model.train()

        data, label = data.to(self.device), label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()

    def test(self):
        self.model.eval()
        test_loss = 0
        pred_list = []
        target_list = []
        with torch.no_grad():
            for data, target in self.test_loader:
                target_list.append(target.numpy())
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1].view(-1) # get the index of the max log-probability
                pred_list.append(pred.cpu().numpy())

        target_list = np.concatenate(target_list)
        pred_list = np.concatenate(pred_list)

        cm = confusion_matrix(target_list, pred_list, labels=range(self.outputs))
        return cm

    def reset_model(self):
        self.model.reset()
        self.optimizer.state = collections.defaultdict(dict)

    def create_model(self):
        if args.benchmark == 'fmnist':
            self.model = Net().to(self.device)
        elif args.benchmark == 'mnist':
            self.model = Net().to(self.device)
        elif args.benchmark == 'cifar10':
            self.model = ResNet_Cifar().to(self.device)
        else:
            assert False, "wrong benchmark"
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.reset_model()

    def load_model(self):
        if self.model is None:
            assert False, 'no model'

        model_path = os.path.join(self.model_dir, self.model_name)

        if not os.path.exists(model_path):
            print("-------------NO MODEL FOUND--------------")
            assert False, 'load_model'

        save_dict = torch.load(model_path)
        self.model.load_state_dict(save_dict['state_dict'])
        self.optimizer.load_state_dict(save_dict['optimizer'])

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        save_dict = {'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}

        torch.save(save_dict, os.path.join(self.model_dir, self.model_name))

    def load_test_loader(self, sample_per_class):

        self.test_loader = get_subset_data_loader(self.data_func, self.transform, False, self.test_loader_batch, sample_per_class)


def get_subset_data_loader(dataset_func, transform, train, batch_size, sample_per_class=None):

    dataset = dataset_func(root=consts.rawdata, train=train, transform=transform, download=False)

    if sample_per_class is None:
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    else:
        y = [i[1] for i in dataset]
        data_dict = {i: list() for i in range(consts.action_space)}
        for i in range(len(y)):
            data_dict[y[i].item()].append(i)

        subset_index = list()
        for i in range(consts.action_space):
            subset_index.append(random.sample(data_dict[i], sample_per_class))

        subset_indexes_flatten = [y for x in subset_index for y in x]
        data_subset = torch.utils.data.Subset(dataset, subset_indexes_flatten)
        return torch.utils.data.DataLoader(dataset=data_subset, batch_size=batch_size, shuffle=True)
