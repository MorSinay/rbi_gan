import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import datasets, transforms
from memory_fmnist import Singleton_Mem
from Net import Net
import torch.optim as optim
import os
import random
import numpy as np

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
        self.acc = None
        self.reward = None
        self.f1 = None
        self.iterations = args.env_iterations
        self.t = 0
        self.k = 0
        self.max_k = args.env_max_k
        self.max_acc = args.env_max_acc
        self.reward_metric = args.reward
        self.reset()

    def reset(self):
        self.model.load_model()
        cm, self.f1 = self.model.test_only_one_batch()

        if self.reward_metric == 'label0':
            self.acc = cm[0][0] / cm.sum(axis=1)[0]
        else:
            self.acc = np.trace(cm)

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
            new_state, next_f1 = self.model.test_only_one_batch()

            if self.reward_metric == 'label0':
                next_acc = new_state[0][0] / new_state.sum(axis=1)[0]
                self.reward = np.float32(next_acc - self.acc)
            elif self.reward_metric == 'acc':
                next_acc = np.trace(new_state)
                self.reward = np.float32((next_acc - self.acc) / (1 - self.acc))
            elif self.reward_metric == 'f1':
                next_acc = np.trace(new_state)
                self.reward = next_f1 - self.f1
            else:
                assert False, "reward not defined"

            self.state = torch.Tensor(new_state).view(-1, self.output_size * self.output_size)
            self.acc = next_acc
            self.f1 = next_f1
            self.k += 1

            if self.k >= self.max_k or self.acc >= self.max_acc:
                self.t = 1

    def step(self, a):

        for _ in range(self.iterations):

            data_gen, label_gen = self.memory.get_item(a)

            self.model.train_batch(data_gen, label_gen)

            # TODO: need to check about the option to save all the test on a GPU
            # TODO: maybe save a GPU to tun only the test
            new_state, next_f1 = self.model.test_only_one_batch()

            if self.reward_metric == 'label0':
                next_acc = new_state[0][0] / new_state.sum(axis=1)[0]
                self.reward = np.float32(next_acc - self.acc)
            elif self.reward_metric == 'acc':
                next_acc = np.trace(new_state)
                self.reward = np.float32((next_acc - self.acc) / (1 - self.acc))
            elif self.reward_metric == 'f1':
                next_acc = np.trace(new_state)
                self.reward = next_f1 - self.f1
            else:
                assert False, "reward not defined"

            self.state = torch.Tensor(new_state).view(-1, self.output_size * self.output_size)
            self.acc = next_acc
            self.f1 = next_f1
            self.k += 1

            if self.k >= self.max_k or self.acc >= self.max_acc:
                # TODO: is it right?
                self.t = 1

class Model():
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.test_loader = None
        self.test_loader_batch = args.test_loader_batch_size
        self.outputs = consts.action_space
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.lr = args.model_lr
        self.betas = (args.model_beta1, args.model_beta2)
        self.model_name = "net.pkl"
        self.model_dir = consts.modeldir

        if args.benchmark == 'fmnist':
            self.load_test_loader_fmnist()
        else:
            self.load_test_loader_cifar10()
        #self.single_test_loader = Singleton_Loader()
        self.load_model()

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
            for data, target in tqdm(self.test_loader):
                target_list.append(target.numpy())
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1].view(-1) # get the index of the max log-probability
                pred_list.append(pred.cpu().numpy())

        target_list = np.concatenate(target_list)
        pred_list = np.concatenate(pred_list)

        cm = confusion_matrix(target_list, pred_list, labels=range(self.outputs))
        f1 = f1_score(target_list,pred_list, labels=range(self.outputs), average='macro')
        return cm/cm.sum(),f1


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
        f1 = f1_score(target_list,pred_list, labels=range(self.outputs), average='macro')
        return cm / cm.sum(), f1


    def create_model(self):
        self.model = Net(self.outputs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)

    def load_model(self):
        if self.model is None:
            self.create_model()
            return

        model_path = os.path.join(self.model_dir, self.model_name)

        if not os.path.exists(model_path):
            print("-------------NO MODEL FOUND--------------")
            if args.benchmark == 'fmnist':
                train_primarily_model_fmnist(250, 128, 4)
            else:
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

    def load_test_loader_fmnist(self):

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


        dataset = datasets.FashionMNIST(root=consts.rawdata, train=False, transform=transform,
                                        download=False)

     #   self.test_loader = torch.utils.data.DataLoader(dataset=dataset,
      #                                                 batch_size=self.test_loader_batch, shuffle=True,
       #                                                num_workers=args.cpu_workers)

        self.test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=self.test_loader_batch, shuffle=True)

    def load_test_loader_cifar10(self):

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))])

        dataset = datasets.CIFAR10(root=consts.rawdata, train=False, transform=transform,
                                        download=False)

        #   self.test_loader = torch.utils.data.DataLoader(dataset=dataset,
        #                                                 batch_size=self.test_loader_batch, shuffle=True,
        #                                                num_workers=args.cpu_workers)

        self.test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.test_loader_batch, shuffle=True)


def train_primarily_model_fmnist(sumples_per_class, batch_size,epochs):
    model = Model()
    model.create_model()

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
        testcm,f1 = model.test()
        testcm = torch.tensor(testcm)
        print("test epoch {} accuracy {:.2f} f1 {}".format(epoch, torch.trace(testcm).item(), f1))

    model.save_model()

def train_primarily_model_cifar10(sumples_per_class, batch_size,epochs):
    model = Model()
    model.create_model()

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
        testcm,f1 = model.test()
        testcm = torch.tensor(testcm)
        print("test epoch {} accuracy {:.2f} f1 {}".format(epoch, torch.trace(testcm).item(), f1))

    model.save_model()


def train_label_0_model_fmnist(sumples_per_class, batch_size,epochs):
    model = Model()
    model.create_model()

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
        cm,f1 = model.test()
        testcm = torch.tensor(cm)
        acc_label_0 = cm[0][0] / cm.sum(axis=1)[0]
        print("test epoch {} accuracy label 0 {:.2f} f1 {}".format(epoch, acc_label_0, f1))

    subset_mnist_index = list()
    subset_mnist_index.append(random.sample(mnist_dict[0], 20))

    subset_indexes_flatten = [y for x in subset_mnist_index for y in x]
    mnist_data_subset = torch.utils.data.Subset(mnist_data, subset_indexes_flatten)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data_subset,
                                              batch_size=batch_size, shuffle=True)

    for epoch in range(3):
        model.train(data_loader)
        cm,f1 = model.test()
        testcm = torch.tensor(cm)
        acc_label_0 = cm[0][0] / cm.sum(axis=1)[0]
        print("test epoch {} accuracy label 0 {:.2f} f1 {}".format(epoch, acc_label_0, f1))

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
