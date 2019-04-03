import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from Dummy_Gen import DummyGen
from Net import Net
import torch.optim as optim
import os
import random
import numpy as np

from config import consts, args

class Env(object):

    def __init__(self):
        self.dummyG = DummyGen()
        self.output_size = consts.action_space
        self.batch_size = args.env_batch_size
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = Model()
        self.state = None
        self.acc = None
        self.reward = None
        self.iterations = args.env_iterations
        self.t = 0
        self.k = 0
        self.max_k = 10 #5600
        self.reset()

    def reset(self):
        self.model.load_model()
        self.state = torch.Tensor(self.model.test())
        self.acc = torch.trace(self.state).item()
        self.t = np.int64(0)
        self.k = 0

    def step_policy(self, policy):

        assert(policy.shape[0] == self.output_size), "action error"

        for _ in range(self.iterations):
            # TODO: need to check about the option of a sampler that sample from the policy distribution
            action_batch = np.random.choice(self.output_size, self.batch_size, p=policy)

            #actions_one_hot = np.zeros((self.batch_size, self.output_size))
            #actions_one_hot[np.arange(self.batch_size), action_batch] = 1

            data_gen, label_gen = self.dummyG.gen(action_batch)

            #self.model.train_batch(data_gen, actions_one_hot)
            self.model.train_batch(data_gen, label_gen)

            # TODO: need to check about the option to save all the test on a GPU
            # TODO: maybe save a GPU to tun only the test
            new_state = torch.tensor(self.model.test())

            next_acc = torch.trace(new_state).item()
            self.reward = np.float32((next_acc - self.acc)/(1-self.acc))

            # TODO: add boost for the reward after some threshold
            assert ((self.reward <= 1) and (self.reward >= -1)), "step function - accuracy problem"

            self.state = new_state
            self.acc = next_acc
            self.k += 1

            if self.k >= self.max_k:
                # TODO: is it right?
                self.t = np.int64(1)
                self.k = 0

    def step(self, a):

        for _ in range(self.iterations):

            data_gen, label_gen = self.dummyG.gen(a)

            #self.model.train_batch(data_gen, actions_one_hot)
            self.model.train_batch(data_gen, label_gen)

            # TODO: need to check about the option to save all the test on a GPU
            # TODO: maybe save a GPU to tun only the test
            new_state = torch.tensor(self.model.test(), dtype=torch.float)

            next_acc = torch.trace(new_state).item()
            self.reward = np.float32((next_acc - self.acc)/(1-self.acc))

            # TODO: add boost for the reward after some threshold
            assert ((self.reward <= 1) and (self.reward >= -1)), "step function - accuracy problem"

            self.state = new_state
            self.acc = next_acc
            self.k += 1

            if self.k >= self.max_k:
                # TODO: is it right?
                self.t = np.int64(1)
                self.k = 0

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

        self.load_test_loader()
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
        cm = None
        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                if cm is None:
                    cm = confusion_matrix(target.view_as(pred), pred, labels=range(self.outputs))
                else:
                    cm += confusion_matrix(target.view_as(pred), pred, labels=range(self.outputs))

        #test_loss /= len(self.test_loader.dataset)
        #print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #    test_loss, correct, len(self.test_loader.dataset),
        #    100. * correct / len(self.test_loader.dataset)))

        cm = cm / len(self.test_loader.dataset)
        return cm

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
            train_primarily_model(250, 128, 5)

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

        transform = transforms.Compose([transforms.ToTensor()  # ,
                                        # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                                        ])


        dataset = datasets.FashionMNIST(root=consts.rawdata, train=False, transform=transform,
                                        download=False)

        self.test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.test_loader_batch, shuffle=False)


def train_primarily_model(sumples_per_class, batch_size,epochs):
    model = Model()
    model.create_model()

    transform = transforms.Compose([transforms.ToTensor()  # ,
                                    # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                                    ])

    mnist_data = datasets.FashionMNIST(root=consts.rawdata,
                                       train=True,
                                       transform=transform,
                                       download=True)

    y = [i[1] for i in mnist_data]
    mnist_dict = {i: list() for i in range(10)}
    for i in range(len(y)):
        mnist_dict[y[i].item()].append(i)

    subset_mnist_index = list()
    for i in range(10):
        subset_mnist_index.append(random.sample(mnist_dict[i], sumples_per_class))

    subset_indexes_flatten = [y for x in subset_mnist_index for y in x]
    mnist_data_subset = torch.utils.data.Subset(mnist_data, subset_indexes_flatten)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data_subset,
                                              batch_size=batch_size)

    for epoch in range(epochs):
        model.train(data_loader)
        testcm = torch.tensor(model.test())
        print("test epoch {} accuracy {:.2f}".format(epoch, torch.trace(testcm).item()))

    model.save_model()
