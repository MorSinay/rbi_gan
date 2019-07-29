import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from config import consts, args, lock_file, release_file
import psutil

from model_ddpg import DuelNet_1 as DuelNet

from memory_gan import Memory, ReplayBatchSampler
from agent import Agent
from environment import Env
import os
import time

import itertools
mem_threshold = consts.mem_threshold


class GANAgent(Agent):

    def __init__(self, exp_name, player=False, choose=False, checkpoint_value=None, checkpoint_beta=None):

        reward_str = "BANDIT"
        print("Learning POLICY method using {} with GANAgent".format(reward_str))
        super(GANAgent, self).__init__(exp_name, checkpoint_value, checkpoint_beta)

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.pi_rand = np.ones(self.action_space, dtype=np.float32) / self.action_space

        if args.beta_init == 'uniform':
            init = torch.tensor(self.pi_rand).to(self.device)
        elif args.beta_init == 'label':
            init = torch.zeros(self.action_space, dtype=torch.float32).to(self.device)
            init[0] = 1
        elif args.beta_init == 'rand':
            init = torch.rand(self.action_space, dtype=torch.float32).to(self.device)
            init /= torch.sum(init)
        else:
            raise ImportError

        self.value_net = DuelNet()
        self.value_net.to(self.device)


        self.q_loss = nn.SmoothL1Loss(reduction='none')

        if player:
            self.beta_net = nn.Parameter(init)
            # play variables
            self.env = Env()
            # TODO Mor: look
            self.n_replay_saved = 1
            self.frame = 0
            self.save_to_mem = args.save_to_mem
            self.explore_only = choose

        else:
            # datasets
            self.train_dataset = Memory()
            self.train_sampler = ReplayBatchSampler(exp_name)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_sampler,
                                                            num_workers=args.cpu_workers, pin_memory=True,
                                                            drop_last=False)

        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer_value = torch.optim.SGD(self.value_net.parameters(), lr=self.value_lr)
        #self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=0.001, eps=1.5e-4, weight_decay=0)

        if player:
            self.optimizer_beta = torch.optim.Adam([self.beta_net], lr=self.beta_lr)
        #self.optimizer_beta = torch.optim.Adam([self.beta_net], lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.n_offset = 0

    def save_value_checkpoint(self, path, aux=None):

        state = {#'beta_net': self.beta_net,
                 'value_net': self.value_net.state_dict(),
                 'optimizer_value': self.optimizer_value.state_dict(),
                 #'optimizer_beta': self.optimizer_beta.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_value_checkpoint(self, path):

        if not os.path.exists(path):
            #return {'n':0}
            assert False, "load_value_checkpoint"

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        #self.beta_net = state['beta_net'].to(self.device)
        self.value_net.load_state_dict(state['value_net'])

        #self.optimizer_beta.load_state_dict(state['optimizer_beta'])
        self.optimizer_value.load_state_dict(state['optimizer_value'])
        self.n_offset = state['aux']['n']

        return state['aux']

    def save_beta_checkpoint(self, path, aux=None):

        state = {'beta_net': self.beta_net,
                 'optimizer_beta': self.optimizer_beta.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_beta_checkpoint(self, path):

        if not os.path.exists(path):
            # return {'n':0}
            assert False, "load_beta_checkpoint"

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        self.beta_net = state['beta_net'].to(self.device)
        self.optimizer_beta.load_state_dict(state['optimizer_beta'])

        return state['aux']

    def learn(self, n_interval, n_tot):
        self.value_net.train()

        results = {'n': [], 'loss_q': [],
                   'r': [], 't': [], 'pi': [], 'pi_tag': [], 'acc': [],
                   'q': [], 'q_onehot': []}

        onehot = torch.zeros(self.action_space, self.action_space).to(self.device)
        onehot[torch.arange(self.action_space), torch.arange(self.action_space)] = 1

        for n, sample in tqdm(enumerate(self.train_loader)):

            r = sample['r'].to(self.device, non_blocking=True)
            t = sample['t'].to(self.device, non_blocking=True)
            pi_explore = sample['pi_explore'].to(self.device, non_blocking=True)
            pi = sample['pi'].to(self.device, non_blocking=True)
            acc = sample['acc'].to(self.device, non_blocking=True)
            pi_tag = sample['pi_tag'].to(self.device, non_blocking=True)

            self.optimizer_value.zero_grad()

            q_value = self.value_net(pi_explore).view(-1)

            loss_q = self.q_loss(q_value, r).mean()
            loss_q.backward()
            self.optimizer_value.step()

            if not n % self.update_memory_interval:
                # save agent state
                self.save_value_checkpoint(self.checkpoint_value, {'n': n})

            # collect actions statistics
            if not n % 50:
                # add results
                results['r'].append(r.data.cpu().numpy())
                results['t'].append(t.data.cpu().numpy())
                results['pi'].append(pi.data.cpu().numpy())
                results['acc'].append(acc.data.cpu().numpy())
                results['pi_tag'].append(pi_tag.data.cpu().numpy())
                results['q'].append(q_value.detach().cpu().numpy())

                # add results
                results['loss_q'].append(loss_q.data.cpu().numpy())

                if not n % n_interval:
                    results['r'] = np.concatenate(results['r'])
                    results['t'] = np.concatenate(results['t'])
                    results['acc'] = np.average(np.concatenate(results['acc']))

                    results['pi'] = np.average(np.concatenate(results['pi']), axis=0).flatten()
                    results['pi_tag'] = np.average(np.concatenate(results['pi_tag']), axis=0).flatten()
                    results['q'] = np.average(np.concatenate(results['q']), axis=0).flatten()
                    results['n'] = n

                    q_onehot = self.value_net(onehot)
                    results['q_onehot'] = q_onehot.data.cpu().numpy().reshape(self.action_space)

                    yield results
                    self.value_net.train()
                    results = {key: [] for key in results}

                    if n >= n_tot:
                        self.save_value_checkpoint(self.checkpoint_value, {'n': n})
                        break

        print("Learn Finish")

    def pca_explore(self, n_players, pca_pi, pi):

        pca = PCA(n_components=2)
        pca.fit(pca_pi)
        explore = pca.transform(pi) + (0.9 ** (7 * np.array(range(n_players))).reshape(n_players, 1)) * np.random.rand(n_players, 2)
        explore = pca.inverse_transform(explore)
        explore = np.clip(explore, a_min=0.0001, a_max=1)
        pi_explore = explore / np.repeat(explore.sum(axis=1, keepdims=True), self.action_space, axis=1)
        return pi_explore

    def grad_explore(self, n_players):
        self.optimizer_beta.zero_grad()
        beta_policy = F.softmax(self.beta_net, dim=0)
        loss_beta = self.value_net(beta_policy)
        loss_beta.backward()

        grads = self.beta_net.grad.detach().cpu().numpy().copy()

        explore_factor = self.delta * grads + self.epsilon * np.random.randn(n_players, self.action_space)
        explore_factor *= 0.9 ** (2*np.array(range(n_players))).reshape(n_players,1)
        beta_explore = self.beta_net.detach().cpu().numpy() + explore_factor
        pi_explore =  np.exp(beta_explore) / np.sum(np.exp(beta_explore), axis=1).reshape(n_players,1)

        return (pi_explore, grads)

    def rand_explore(self, n_players):
        pi_explore = np.random.randn(n_players, self.action_space)
        pi_explore = np.exp(pi_explore) / np.sum(np.exp(pi_explore), axis=1).reshape(n_players,1)

        return pi_explore

    def uniform_explore(self, n_players, pi):
        explore = np.random.rand(n_players, self.action_space)
        # explore = np.exp(explore) / np.sum(np.exp(explore), axis=1).reshape(n_players, 1)
        explore = explore / np.sum(explore, axis=1).reshape(n_players, 1)

        pi_explore = self.epsilon * explore + (1 - self.epsilon) * pi
        pi_explore = pi_explore / np.repeat(pi_explore.sum(axis=1, keepdims=True), self.action_space, axis=1)

        return pi_explore

    def multiplay(self):

        n_players = self.n_players

        mp_env = [Env() for _ in range(n_players)]
        self.frame = 0

        rewards = [[[]] for _ in range(n_players)]
        accs = [[[]] for _ in range(n_players)]
        episode = [[] for _ in range(n_players)]
        ts = [[[]] for _ in range(n_players)]
        policies = [[[]] for _ in range(n_players)]
        explore_policies = [[[]] for _ in range(n_players)]
        trajectory = [[] for _ in range(n_players)]
        grads = [[[]] for _ in range(n_players)]

        # set initial episodes number
        # lock read
        fwrite = lock_file(self.episodelock)
        current_num = np.load(fwrite).item()
        episode_num = current_num + np.arange(n_players)
        fwrite.seek(0)
        np.save(fwrite, current_num + n_players)
        release_file(fwrite)

#        for i in range(n_players):
 #           mp_env[i].reset()

        if self.save_beta:
            self.save_beta_checkpoint(self.checkpoint_beta, {'n': 0})
            print("save beta")

        grad = np.zeros(10)

        for self.frame in tqdm(itertools.count()):

            pi = F.softmax(self.beta_net.detach(), dim=0)
            pi = pi.data.cpu().numpy().reshape(-1, self.action_space)
            # to assume there is no zero policy
            pi = (1 - self.eta) * pi + self.eta / self.action_space

            if self.explore_only or self.n_offset <= self.n_rand:
                #pi_explore = self.rand_explore(n_players)
                pi_explore = self.uniform_explore(n_players, pi)

            elif len(policies[0][0]) > 20 and args.exploration == 'PCA':
                pca_pi = np.vstack(np.concatenate(np.vstack(policies)))
                pi_explore = self.pca_explore(n_players, pca_pi, pi)

            elif self.n_offset > self.n_rand and args.exploration == 'GRAD':
                (pi_explore, grad) = self.grad_explore(n_players)

            elif self.n_offset > self.n_rand and args.exploration == 'UNIFORM':
                pi_explore = self.uniform_explore(n_players, pi)
            else:
                #pi_explore = self.rand_explore(n_players)
                pi_explore = self.uniform_explore(n_players, pi)

            for i in range(n_players):
                grads[i][-1].append(grad)

                env = mp_env[i]
                env.step_policy(np.expand_dims(pi_explore[i], axis=0))
                rewards[i][-1].append(env.reward)
                accs[i][-1].append(env.acc)
                ts[i][-1].append(env.t)
                policies[i][-1].append(pi)
                explore_policies[i][-1].append(pi_explore[i])

                episode_format = np.array((self.frame, rewards[i][-1][-1],
                                           accs[i][-1][-1], ts[i][-1][-1], policies[i][-1][-1],
                                           explore_policies[i][-1][-1], -1, episode_num[i]), dtype=consts.rec_type)

                episode[i].append(episode_format)

                if env.t:

                    episode_df = np.stack(episode[i])
                    trajectory[i].append(episode_df)

                    # write if enough space is available
                    if psutil.virtual_memory().available >= mem_threshold:
                        # read
                        fwrite = lock_file(self.writelock)
                        traj_num = np.load(fwrite).item()
                        fwrite.seek(0)
                        np.save(fwrite, traj_num + 1)
                        release_file(fwrite)

                        traj_to_save = np.concatenate(trajectory[i])
                        traj_to_save['traj'] = traj_num

                        traj_file = os.path.join(self.trajectory_dir, "%d.npy" % traj_num)
                        np.save(traj_file, traj_to_save)

                        fread = lock_file(self.readlock)
                        traj_list = np.load(fread)
                        fread.seek(0)
                        np.save(fread, np.append(traj_list, traj_num))
                        release_file(fread)

                        fwrite = lock_file(self.episodelock)
                        episode_num[i] = np.load(fwrite).item()
                        fwrite.seek(0)
                        np.save(fwrite, episode_num[i] + 1)
                        release_file(fwrite)

                        if (i == 0) and (not self.frame % 50):
                            print("actor {} player {} - acc {}, n-offset {}, frame {}, episode {}, r {}, pi_explore {}".format(
                                    self.actor_index, i, env.acc, self.n_offset, self.frame, episode_num[i],env.reward, pi_explore[i]))
                    else:
                        assert False, "memory error available memory {}".format(psutil.virtual_memory().available)

                    env.reset()
                    trajectory[i] = []
                else:
                    assert False, "env.t error"

                if not (self.frame % self.load_memory_interval):
                    try:
                        self.load_value_checkpoint(self.checkpoint_value)
                    except:
                        pass

                    self.value_net.eval()

                if self.n_offset > self.n_rand:
                    for _ in range(10):
                        self.optimizer_beta.zero_grad()
                        beta_policy = F.softmax(self.beta_net, dim=0)
                        loss_beta = -self.value_net(beta_policy)
                        loss_beta.backward()
                        self.optimizer_beta.step()

            if self.save_beta:
                self.save_beta_checkpoint(self.checkpoint_beta,  {'n': self.frame})
                print("save beta")

            if not self.frame % self.player_replay_size:
                beta_policy = F.softmax(self.beta_net, dim=0).detach().cpu().numpy()
                a = np.hstack(accs)
                max_player = np.argmax(a)
                p = np.hstack(policies)
                ep = np.hstack(explore_policies)
                results  = {}
                results['actor'] = self.actor_index
                results['acc'] = a[0,max_player]
                results['frame'] = self.frame
                results['n_offset'] = self.n_offset
                results['pi'] = p[0,max_player]
                results['epi'] = ep[0,max_player]
                results['beta'] = beta_policy

                yield results

                if self.save_beta: # and self.frame>30:
                    __pi = np.vstack(np.concatenate(np.vstack(policies)))
                    __pi_explore = np.vstack(np.concatenate(np.vstack(explore_policies)))
                    __grads = np.vstack(np.concatenate(np.vstack(grads)))
                    pi_file = os.path.join(self.root_dir, "pi.npy")
                    pi_explore_file = os.path.join(self.root_dir, "pi_explore.npy")
                    grad_file = os.path.join(self.root_dir, "grad.npy")
                    np.save(pi_file, __pi)
                    np.save(pi_explore_file, __pi_explore)
                    np.save(grad_file, __grads)

                    # # flaten = [val for sublist in policies for val in sublist]
                    # pi = np.vstack(np.concatenate(np.vstack(policies)))
                    # pi_explore = np.vstack(np.concatenate(np.vstack(explore_policies)))
                    # pca = PCA(n_components=2)
                    # np.seterr(divide='ignore', invalid='ignore')
                    # X_train = pca.fit_transform(pi)
                    # X_test = pca.transform(pi_explore)
                    # plt.plot(X_train[0], X_train[1], 'ro', label='pi')
                    # plt.plot(X_test[0], X_test[1], 'bo', label='pi_explore')
                    # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
                    # path = os.path.join(self.root_dir, 'pca.png')
                    # plt.savefig(path, bbox_inches='tight')
                    # plt.close()

                if self.n_offset >= self.n_tot:
                    break

    def demonstrate(self, n_tot):
        return

    def train(self, n_interval, n_tot):
        return

    def evaluate(self):

        onehot = torch.zeros(self.action_space, self.action_space).to(self.device)
        onehot[torch.arange(self.action_space), torch.arange(self.action_space)] = 1

        beta_checkpoint_n = 0
        try:
            self.load_value_checkpoint(self.checkpoint_value)
            beta_checkpoint_n = (self.load_beta_checkpoint(self.checkpoint_beta))['n']
        except:
            print("Error in load checkpoint")
            pass

        results = {'n': [], 'pi': [], 'beta': [], 'q': [], 'acc': [], 'r': [], 'score': [], 'q_onehot': [],
                   'beta_checkpoint': []}

        #for _ in tqdm(itertools.count()):
        for _ in itertools.count():

            try:
                self.load_value_checkpoint(self.checkpoint_value)
                beta_checkpoint_n = (self.load_beta_checkpoint(self.checkpoint_beta))['n']
            except:
                print("Error in load checkpoint")
                continue

            self.env.reset()
            self.value_net.eval()

            for _ in tqdm(itertools.count()):

                beta = self.beta_net
                beta = F.softmax(beta.detach(), dim=0)

                q = self.value_net(beta)

                q_onehot = self.value_net(onehot)
                q_onehot = q_onehot.data.cpu().numpy().reshape(self.action_space)

                beta = beta.data.cpu().numpy().reshape(self.action_space)
                q = q.data.cpu().numpy()

                pi = beta
                pi = np.expand_dims(pi, axis=0)
                self.env.step_policy(pi)

                results['pi'].append(pi)
                results['beta'].append(beta)
                results['q'].append(q)
                results['r'].append(self.env.reward)
                results['acc'].append(self.env.acc)
                results['q_onehot'].append(q_onehot)

                if self.env.t:
                    break

            if self.env.t:
                results['pi'] = np.average(np.asarray(results['pi']), axis=0).flatten()
                results['beta'] = np.average(np.asarray(results['beta']), axis=0).flatten()
                results['q'] = np.average(np.asarray(results['q']), axis=0).flatten()
                results['q_onehot'] = np.average(np.asarray(results['q_onehot']), axis=0).flatten()
                results['acc'] = np.asarray(results['acc'])
                results['r'] = np.asarray(results['r'])

                results['beta_checkpoint'] = beta_checkpoint_n
                results['n'] = self.n_offset
                results['score'] = results['q']
                #results['acc'] = self.env.acc

                yield results
                results = {key: [] for key in results}

            if self.n_offset >= args.n_tot:
                break

            time.sleep(60)
