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
        self.state = None
        self.acc = 0
        self.reward = None
        self.iterations = args.env_iterations
        self.t = 0
        self.k = 0
        self.max_k = args.env_max_k
        self.max_acc = args.env_max_acc

        self.list_reward = [0.3, 0.4, 0.45, 0.5, 1.0, 10, 100]
        self.list_level = [0.3, 0.4, 0.6, 0.8, 0.9, 0.95, 1.]
        self.level_achieved = [False, False, False, False, False, False, False]

        if args.evaluate:
            self.test_func = self.model.test
        else:
            self.test_func = self.model.test
            #self.test_func = self.model.test_only_one_batch

        if args.acc == 'all':
            self.acc_func = self.acc_calc
        else:
            self.acc_func = self.label_calc

        if args.reward == 'shape':
            self.reward_func = self.reward_shape
        elif args.reward == 'no_shape':
            self.reward_func = self.reward_no_shape
        elif args.reward == 'final':
            self.reward_func = self.reward_final
        else:
            self.reward_func = self.reward_step

        self.reset()

    def acc_calc(self, cm):
        return np.trace(cm)/np.sum(cm)

    def label_calc(self, cm):
        label1 = 9
        label2 = 5
        return 0.5*cm[label1][label1]/cm.sum(axis=1)[label1] + 0.5*cm[label2][label2]/cm.sum(axis=1)[label2]

    def reward_shape(self, next_acc):
        self.reward = np.float32((next_acc - self.acc) / (1 - self.acc))

    def reward_no_shape(self, next_acc):
        self.reward = np.float32(next_acc - self.acc)

    def reward_step(self, next_acc):

        self.reward = 0

        items = [i for i, x in enumerate(self.level_achieved) if (not x) and (next_acc >= self.list_level[i])]
        if len(items) != 0:
            self.reward = self.list_reward[items[0]]
            self.list_level[items[0]] = True

    def reward_final(self, next_acc):

        self.reward = 0
        if self.t == 1:
            self.reward = min(-np.log2(next_acc), 10)

    def reset(self):
        #self.model.load_model()
        self.model.reset_model()
        cm = self.test_func()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        self.acc = 0
        self.reward = 0
        self.state = torch.Tensor(cm).view(-1, self.output_size * self.output_size)

        self.t = 0
        self.k = 0

    def step_policy(self, policy):

        assert(policy.shape[1] == self.output_size), "action error"

        for _ in range(self.iterations):
            # TODO: need to check about the option of a sampler that sample from the policy distribution
            action_batch = np.random.choice(self.output_size, self.batch_size, p=policy[0])
            data_gen, label_gen = self.memory.get_item(action_batch)

            self.model.train_batch(data_gen, label_gen)

            # TODO: need to check about the option to save all the test on a GPU
            # TODO: maybe save a GPU to tun only the test
            cm = self.test_func()
            next_acc = self.acc_func(cm)

            self.k += 1
            if self.k >= self.max_k or next_acc >= self.max_acc:
                self.t = 1

            self.reward_func(next_acc)
            self.acc = next_acc
            new_state = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.state = torch.Tensor(new_state).view(-1, self.output_size * self.output_size)

    def step(self, a):

        for _ in range(self.iterations):

            data_gen, label_gen = self.memory.get_item(a)

            self.model.train_batch(data_gen, label_gen)

            # TODO: need to check about the option to save all the test on a GPU
            # TODO: maybe save a GPU to tun only the test
            cm = self.test_func()
            next_acc = self.acc_func(cm)

            self.k += 1

            if self.k >= self.max_k or next_acc >= self.max_acc:
                self.t = 1

            self.reward_func(next_acc)
            self.acc = next_acc
            new_state = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.state = torch.Tensor(new_state).view(-1, self.output_size * self.output_size)


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
        self.load_test_loader()

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

    def test_only_one_batch(self):
        self.model.eval()
        test_loss = 0
        pred_list = []
        target_list = []

        for _ in range(2):
            with torch.no_grad():
                data, target = next(iter(self.test_loader))
                target_list.append(target.numpy())
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1].view(-1) # get the index of the max log-probability
                pred_list.append(pred.cpu().numpy())

        target_list = np.concatenate(target_list)
        pred_list = np.concatenate(pred_list)

        cm = confusion_matrix(target_list, pred_list, labels=range(self.outputs))
        #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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

    def load_test_loader(self):

        self.test_loader = get_subset_data_loader(self.data_func, self.transform, False, self.test_loader_batch, 100)


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

def train_primarily_model_fmnist(sumples_per_class, batch_size,epochs):
    model = Model()
    #model.create_model()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    mnist_data = datasets.FashionMNIST(root=consts.rawdata,
                                       train=True,
                                       transform=transform,
                                       download=False)

    y = [i[1] for i in mnist_data]
    mnist_dict = {i: list() for i in range(consts.action_space)}
    for i in range(len(y)):
        mnist_dict[y[i].item()].append(i)

    subset_mnist_index = list()
    for i in range(consts.action_space):
        subset_mnist_index.append(random.sample(mnist_dict[i], sumples_per_class))

    subset_indexes_flatten = [y for x in subset_mnist_index for y in x]
    mnist_data_subset = torch.utils.data.Subset(mnist_data, subset_indexes_flatten)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data_subset,
                                              batch_size=batch_size)

    for epoch in range(epochs):
        model.train(data_loader)
        testcm = model.test()
        testcm = torch.tensor(testcm).item()/consts.action_space
        print("test epoch {} accuracy {:.2f}".format(epoch, testcm))

    model.save_model()


def train_primarily_model_mnist(sumples_per_class, batch_size,epochs):
    model = Model()
    #model.create_model()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    mnist_data = datasets.MNIST(root=consts.rawdata,
                                       train=True,
                                       transform=transform,
                                       download=False)

    y = [i[1] for i in mnist_data]
    mnist_dict = {i: list() for i in range(consts.action_space)}
    for i in range(len(y)):
        mnist_dict[y[i].item()].append(i)

    subset_mnist_index = list()
    for i in range(consts.action_space):
        subset_mnist_index.append(random.sample(mnist_dict[i], sumples_per_class))

    subset_indexes_flatten = [y for x in subset_mnist_index for y in x]
    mnist_data_subset = torch.utils.data.Subset(mnist_data, subset_indexes_flatten)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data_subset,
                                              batch_size=batch_size)

    for epoch in range(epochs):
        model.train(data_loader)
        testcm = model.test()
        testcm = torch.tensor(testcm)
        print("test epoch {} accuracy {:.2f}".format(epoch, torch.trace(testcm).item()))

    model.save_model()

def train_primarily_model_cifar10(sumples_per_class, batch_size,epochs):
    model = Model()
    #model.create_model()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))])

    data_set = datasets.CIFAR10(root=consts.rawdata,
                                       train=True,
                                       transform=transform,
                                       download=False)

    y = [i[1] for i in data_set]
    data_dict = {i: list() for i in range(consts.action_space)}
    for i in range(len(y)):
        data_dict[y[i]].append(i)

    subset_index = list()
    for i in range(consts.action_space):
        subset_index.append(random.sample(data_dict[i], sumples_per_class))

    subset_indexes_flatten = [y for x in subset_index for y in x]
    data_subset = torch.utils.data.Subset(data_set, subset_indexes_flatten)
    data_loader = torch.utils.data.DataLoader(dataset=data_subset,
                                              batch_size=batch_size)

    for epoch in range(epochs):
        model.train(data_loader)
        testcm = model.test()
        testcm = torch.tensor(testcm).item()/consts.action_space
        print("test epoch {} accuracy {:.2f}".format(epoch, testcm))

    model.save_model()


def train_label_0_model_fmnist(sumples_per_class, batch_size,epochs):
    model = Model()
    #model.create_model()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    mnist_data = datasets.FashionMNIST(root=consts.rawdata,
                                       train=True,
                                       transform=transform,
                                       download=False)

    y = [i[1] for i in mnist_data]
    mnist_dict = {i: list() for i in range(consts.action_space)}
    for i in range(len(y)):
        mnist_dict[y[i].item()].append(i)

    subset_mnist_index = list()
    for i in range(consts.action_space):
        subset_mnist_index.append(random.sample(mnist_dict[i], sumples_per_class))

    subset_indexes_flatten = [y for x in subset_mnist_index for y in x]
    mnist_data_subset = torch.utils.data.Subset(mnist_data, subset_indexes_flatten)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data_subset,
                                              batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train(data_loader)
        cm = model.test()
        testcm = torch.tensor(cm)
        acc_label_0 = cm[9][9]
        print("test epoch {} accuracy label 0 {:.2f}".format(epoch, acc_label_0))

    subset_mnist_index = list()
    subset_mnist_index.append(random.sample(mnist_dict[0], 20))

    subset_indexes_flatten = [y for x in subset_mnist_index for y in x]
    mnist_data_subset = torch.utils.data.Subset(mnist_data, subset_indexes_flatten)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data_subset,
                                              batch_size=batch_size, shuffle=True)

    for epoch in range(3):
        model.train(data_loader)
        cm = model.test()
        testcm = torch.tensor(cm)
        acc_label_0 = cm[9][9]
        print("test epoch {} accuracy label 0 {:.2f}".format(epoch, acc_label_0))

# class Singleton(type):
#     _instances = {}
#
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]
#
#
# class Singleton_Loader(metaclass=Singleton):
#     def __init__(self):
#         transform = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
#
#         dataset = datasets.FashionMNIST(root=consts.rawdata, train=False, transform=transform,
#                                         download=False)
#
#
#         self.test_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                         batch_size=args.test_loader_batch_size, shuffle=True,
#                                                        num_workers=args.cpu_workers,pin_memory=False)
#
#         # self.test_loader = torch.utils.data.DataLoader(dataset=dataset,
#         #                                                batch_size=self.test_loader_batch, shuffle=True, num_workers=10,
#         #                                                pin_memory=False)
#
#
