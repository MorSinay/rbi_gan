import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn as nn

from config import consts, args, lock_file, release_file
import psutil

from model_ddpg import DuelNet

from memory_gan import Memory, ReplayBatchSampler
from agent import Agent
from environment import Env
import os
import time

import itertools
mem_threshold = consts.mem_threshold


class GANAgent(Agent):

    def __init__(self, exp_name, player=False, choose=False, checkpoint=None):

        reward_str = "BANDIT"
        print("Learning POLICY method using {} with GANAgent".format(reward_str))
        super(GANAgent, self).__init__(exp_name, checkpoint)

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.pi_rand = np.ones(self.action_space, dtype=np.float32) / self.action_space

        self.beta_net = nn.Parameter(torch.tensor(self.pi_rand))
        self.value_net = DuelNet()

        self.beta_net = self.beta_net.to(self.device)
        self.value_net.to(self.device)


        self.q_loss = nn.SmoothL1Loss(reduction='none')

        if player:
            # play variables
            self.env = Env()
            # TODO Mor: look
            self.n_replay_saved = 1
            self.frame = 0
            self.save_to_mem = args.save_to_mem

        else:
            # datasets
            self.train_dataset = Memory()
            self.train_sampler = ReplayBatchSampler(exp_name)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_sampler,
                                                            num_workers=args.cpu_workers, pin_memory=True,
                                                            drop_last=False)

        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.optimizer_beta = torch.optim.Adam([self.beta_net], lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.n_offset = 0

    def save_checkpoint(self, path, aux=None):

        state = {'beta_net': self.beta_net,
                 'value_net': self.value_net.state_dict(),
                 'optimizer_value': self.optimizer_value.state_dict(),
                 'optimizer_beta': self.optimizer_beta.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        if not os.path.exists(path):
            return {'n':0}
        #    assert False, "load_checkpoint"

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        self.beta_net = state['beta_net'].to(self.device)
        self.value_net.load_state_dict(state['value_net'])

        self.optimizer_beta.load_state_dict(state['optimizer_beta'])
        self.optimizer_value.load_state_dict(state['optimizer_value'])
        self.n_offset = state['aux']['n']

        return state['aux']

    def learn(self, n_interval, n_tot):
        self.save_checkpoint(self.snapshot_path, {'n': 0})

      #  self.beta_net.to(self.device)
        self.value_net.train()

        results = {'n': [], 'loss_q': [], 'loss_beta': [],
                   'r': [], 't': [], 'pi': [], 'pi_tag': [], 'acc': [], 'beta': [],
                   'q': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            r = sample['r'].to(self.device, non_blocking=True)
            t = sample['t'].to(self.device, non_blocking=True)
            pi_explore = sample['pi_explore'].to(self.device, non_blocking=True)
            pi = sample['pi'].to(self.device, non_blocking=True)
            acc = sample['acc'].to(self.device, non_blocking=True)
            pi_tag = sample['pi_tag'].to(self.device, non_blocking=True)

            self.optimizer_beta.zero_grad()
            self.optimizer_value.zero_grad()

            beta_policy = F.softmax(self.beta_net)
            q_value = self.value_net(pi_explore).view(-1)

            loss_q = self.q_loss(q_value, r).mean()
            loss_q.backward()
            self.optimizer_value.step()

            loss_beta = -self.value_net(beta_policy)
            loss_beta.backward()
            self.optimizer_beta.step()

            # collect actions statistics
            if not n % 50:
                # add results
                results['r'].append(r.data.cpu().numpy())
                results['t'].append(t.data.cpu().numpy())
                results['pi'].append(pi.data.cpu().numpy())
                results['acc'].append(acc.data.cpu().numpy())
                results['pi_tag'].append(pi_tag.data.cpu().numpy())
                results['beta'].append(beta_policy.detach().cpu().numpy())
                results['q'].append(q_value.detach().cpu().numpy())

                # add results
                results['loss_beta'].append(loss_beta.data.cpu().numpy())
                results['loss_q'].append(loss_q.data.cpu().numpy())

                if not n % self.update_memory_interval:
                    # save agent state
                    self.save_checkpoint(self.snapshot_path, {'n': n})

                if not n % n_interval:
                    results['r'] = np.concatenate(results['r'])
                    results['t'] = np.concatenate(results['t'])
                    results['acc'] = np.average(np.concatenate(results['acc']))

                    results['pi'] = np.average(np.concatenate(results['pi']), axis=0).flatten()
                    results['pi_tag'] = np.average(np.concatenate(results['pi_tag']), axis=0).flatten()
                    results['beta'] = np.average(np.concatenate(results['beta']), axis=0).flatten()
                    results['q'] = np.average(np.concatenate(results['q']), axis=0).flatten()
                    results['n'] = n

                    yield results
                    self.value_net.train()
                    results = {key: [] for key in results}

                    if n >= n_tot:
                        self.save_checkpoint(self.snapshot_path, {'n': n})
                        break

        print("Learn Finish")

    def multiplay(self):

        n_players = self.n_players

        mp_env = [Env() for _ in range(n_players)]
        self.frame = 0

        range_players = np.arange(n_players)
        rewards = [[[]] for _ in range(n_players)]
        accs = [[[]] for _ in range(n_players)]
        states = [[[]] for _ in range(n_players)]
        episode = [[] for _ in range(n_players)]
        ts = [[[]] for _ in range(n_players)]
        policies = [[[]] for _ in range(n_players)]
        explore_policies = [[[]] for _ in range(n_players)]
        trajectory = [[] for _ in range(n_players)]

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

        for self.frame in tqdm(itertools.count()):

            if not (self.frame % self.load_memory_interval):
                try:
                    self.load_checkpoint(self.snapshot_path)
                except:
                    pass

                self.value_net.eval()

            if self.n_offset <= self.n_rand:
                pi_explore = np.repeat(self.pi_rand, n_players, axis=0).reshape(n_players,self.action_space)
                pi = pi_explore
            else:
                beta = self.beta_net
                beta = F.softmax(beta.detach())
                beta = beta.data.cpu().numpy().reshape(-1,self.action_space)

                pi = beta.copy()

                #to assume there is no zero policy
                pi = (1 - self.eta) * pi + self.eta / self.action_space

                explore = np.random.rand(n_players, self.action_space)
                #explore = np.exp(explore) / np.sum(np.exp(explore), axis=1).reshape(n_players, 1)
                explore = explore / np.sum(explore, axis=1).reshape(n_players, 1)

                pi_explore = self.epsilon * explore + (1 - self.epsilon) * pi
                pi_explore = pi_explore / np.repeat(pi_explore.sum(axis=1, keepdims=True), self.action_space, axis=1)


            for i in range(n_players):

                env = mp_env[i]

                env.step_policy(np.expand_dims(pi_explore[i], axis=0))
                rewards[i][-1].append(env.reward)
                accs[i][-1].append(env.acc)
                ts[i][-1].append(env.t)
                policies[i][-1].append(pi[i])
                explore_policies[i][-1].append(pi_explore[i])

                episode_format = np.array((self.frame, rewards[i][-1][-1],
                                           accs[i][-1][-1], ts[i][-1][-1], policies[i][-1][-1],
                                           explore_policies[i][-1][-1], -1, episode_num[i]), dtype=consts.rec_type)

                episode[i].append(episode_format)

                if env.t:
                #if env.t:

                    #print("player {} - acc {}, n-offset {}, frame {}, episode {} k {}".format(i, env.acc, self.n_offset, self.frame, episode_num[i], env.k))

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
                    else:
                        assert False, "memory error available memory {}".format(psutil.virtual_memory().available)

                    trajectory[i] = []

                    if env.t:

                        fwrite = lock_file(self.episodelock)
                        episode_num[i] = np.load(fwrite).item()
                        fwrite.seek(0)
                        np.save(fwrite, episode_num[i] + 1)
                        release_file(fwrite)

                        env.reset()

            if not self.frame % self.player_replay_size:
                yield True
                if self.n_offset >= self.n_tot:
                    break

    def demonstrate(self, n_tot):
        return

    def train(self, n_interval, n_tot):
        return

    def evaluate(self):

        try:
            self.load_checkpoint(self.snapshot_path)
        except:
            print("Error in load checkpoint")
            pass

        results = {'n': [], 'pi': [], 'beta': [], 'q': [], 'acc': [], 'k': [], 'r': [], 'score': []}

        #for _ in tqdm(itertools.count()):
        for _ in itertools.count():

            try:
                self.load_checkpoint(self.snapshot_path)
            except:
                print("Error in load checkpoint")
                continue

            self.env.reset()
            self.value_net.eval()

            for _ in tqdm(itertools.count()):

                beta = self.beta_net
                beta = F.softmax(beta.detach())

                q = self.value_net(beta)

                beta = beta.data.cpu().numpy().reshape(self.action_space)
                q = q.data.cpu().numpy()

                if self.n_offset <= self.n_rand:
                    pi = self.pi_rand
                else:
                    pi = beta

                pi = np.expand_dims(pi, axis=0)
                self.env.step_policy(pi)

                results['pi'].append(pi)
                results['beta'].append(beta)
                results['q'].append(q)
                results['r'].append(self.env.reward)
                results['acc'].append(self.env.acc)

                if self.env.t:
                    break

            if self.env.t:
                results['pi'] = np.average(np.asarray(results['pi']), axis=0).flatten()
                results['beta'] = np.average(np.asarray(results['beta']), axis=0).flatten()
                results['q'] = np.average(np.asarray(results['q']), axis=0).flatten()
                results['acc'] = np.asarray(results['acc'])
                results['r'] = np.asarray(results['r'])

                results['n'] = self.n_offset
                results['score'] = results['q']
                #results['acc'] = self.env.acc

                yield results
                results = {key: [] for key in results}

            if self.n_offset >= args.n_tot:
                break
