import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn as nn
import sys

from config import consts, args, lock_file, release_file
import psutil
#import socket

from model import BehavioralNet, DuelNet

from memory_gan import Memory, ReplayBatchSampler
from agent import Agent
from environment import Env
#from preprocess import release_file, lock_file, get_mc_value, get_td_value, h_torch, hinv_torch, get_expected_value, get_tde
import os
#import time
#import shutil
import itertools
mem_threshold = consts.mem_threshold


class GANAgent(Agent):

    def __init__(self, exp_name, player=False, choose=False, checkpoint=None):

        print("Learning with GANAgent")
        super(GANAgent, self).__init__(exp_name, checkpoint)

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.beta_net = BehavioralNet()
        self.value_net = DuelNet()
        self.target_net = DuelNet()

        if torch.cuda.device_count() > 1:
            self.beta_net = nn.DataParallel(self.beta_net)
            self.value_net = nn.DataParallel(self.value_net)
            self.target_net = nn.DataParallel(self.target_net)

        self.beta_net.to(self.device)
        self.value_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())

        self.pi_rand = np.ones(self.action_space) / self.action_space
        self.q_loss = nn.SmoothL1Loss(reduction='none')
        self.kl_loss = nn.KLDivLoss()

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
        self.optimizer_beta = torch.optim.Adam(self.beta_net.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.n_offset = 0

    def save_checkpoint(self, path, aux=None):

        if torch.cuda.device_count() > 1:
            state = {'beta_net': self.beta_net.module.state_dict(),
                     'value_net': self.value_net.module.state_dict(),
                     'target_net': self.target_net.module.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_beta': self.optimizer_beta.state_dict(),
                     'aux': aux}
        else:
            state = {'beta_net': self.beta_net.state_dict(),
                     'value_net': self.value_net.state_dict(),
                     'target_net': self.target_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_beta': self.optimizer_beta.state_dict(),
                     'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        if not os.path.exists(path):
        #    return {'n':0}
            assert False, "load_checkpoint"

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        if torch.cuda.device_count() > 1:
            self.beta_net.module.load_state_dict(state['beta_net'])
            self.value_net.module.load_state_dict(state['value_net'])
            self.target_net.module.load_state_dict(state['target_net'])
        else:
            self.beta_net.load_state_dict(state['beta_net'])
            self.value_net.load_state_dict(state['value_net'])
            self.target_net.load_state_dict(state['target_net'])

        self.optimizer_beta.load_state_dict(state['optimizer_beta'])
        self.optimizer_value.load_state_dict(state['optimizer_value'])
        self.n_offset = state['aux']['n']

        return state['aux']

    def learn(self, n_interval, n_tot):

        self.save_checkpoint(self.snapshot_path, {'n': 0})

        self.beta_net.train()
        self.value_net.train()
        self.target_net.eval()

        results = {'n': [], 'loss_q': [], 'loss_beta': [], 'a_player': [], 'loss_std': [],
                   'r': [], 's': [], 't': [], 'pi': [], 's_tag': [], 'pi_tag': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = sample['s'].to(self.device, non_blocking=True)
            a = sample['a'].to(self.device, non_blocking=True)
            r = sample['r'].to(self.device, non_blocking=True)
            t = sample['t'].to(self.device, non_blocking=True)
            pi = sample['pi'].to(self.device, non_blocking=True)
            s_tag = sample['s_tag'].to(self.device, non_blocking=True)
            pi_tag = sample['pi_tag'].to(self.device, non_blocking=True)


            # Behavioral nets
            beta = self.beta_net(s)
            beta_log = F.log_softmax(beta, dim=1)
            loss_beta = (-beta_log * pi).sum(dim=1).mean()

            # dqn
            q_tag = self.target_net(s_tag).detach()
            #TODO Mor: check
            q = self.value_net(s)

            ind = range(q.shape[0])
            q_a = q[ind, a]
            # index = torch.unsqueeze(a, 1)
            # one_hot = torch.LongTensor(q.shape).zero_().to(self.device)
            # one_hot = one_hot.scatter(1, index, 1)

            target_value = r + args.gamma * (pi_tag*q_tag).sum(dim=1)
            loss_q = (self.q_loss(q_a, target_value)).mean()

            self.optimizer_beta.zero_grad()
            loss_beta.backward()
            self.optimizer_beta.step()

            self.optimizer_value.zero_grad()
            loss_q.backward()
            self.optimizer_value.step()

            # collect actions statistics
            if not n % 50:
                # add results
                results['a_player'].append(a.data.cpu().numpy())
                results['r'].append(r.data.cpu().numpy())
                results['s'].append(s.data.cpu().numpy())
                results['t'].append(t.data.cpu().numpy())
                results['pi'].append(pi.data.cpu().numpy())
                results['s_tag'].append(s_tag.data.cpu().numpy())
                results['pi_tag'].append(pi_tag.data.cpu().numpy())

                # add results
                results['loss_beta'].append(loss_beta.data.cpu().numpy())
                results['loss_q'].append(loss_q.data.cpu().numpy())
                results['loss_std'].append(0)
                results['n'].append(n)

                if not n % self.update_memory_interval:
                    # save agent state
                    self.save_checkpoint(self.snapshot_path, {'n': n})

                if not n % self.update_target_interval:
                    # save agent state
                    self.target_net.load_state_dict(self.value_net.state_dict())

                if not n % n_interval:
                    results['a_player'] = np.concatenate(results['a_player'])
                    results['r'] = np.concatenate(results['r'])
                    results['s'] = np.concatenate(results['s'])
                    results['t'] = np.concatenate(results['t'])
                    results['pi'] = np.concatenate(results['pi'])
                    results['s_tag'] = np.concatenate(results['s_tag'])
                    results['pi_tag'] = np.concatenate(results['pi_tag'])

                    yield results
                    self.beta_net.train()
                    self.value_net.train()
                    results = {key: [] for key in results}

                    if n >= n_tot:
                        print("break")
                        break

        print("Learn Finish")

    def play(self, n_tot):
        # set initial episodes number
        # lock read
        fwrite = lock_file(self.episodelock)
        current_num = np.load(fwrite).item()
        episode_num = current_num
        fwrite.seek(0)
        np.save(fwrite, current_num + 1)
        release_file(fwrite)

        for i in range(n_tot):

            self.env.reset()
            episode = []
            trajectory = []
            rewards = [[]]
            states = [[]]
            ts = [[]]
            policies = [[]]

            self.beta_net.eval()
            self.value_net.eval()

            while not self.env.t:

                if not (self.frame % self.load_memory_interval):
                    try:
                        self.load_checkpoint(self.snapshot_path)
                    except:
                        pass

                    self.beta_net.eval()
                    self.value_net.eval()

                s = self.env.state.to(self.device)
                s_flat = s.view(-1, self.action_space*self.action_space)
                # get aux data

                beta = self.beta_net(s_flat)
                beta = F.softmax(beta.detach(), dim=1)
                beta = beta.data.cpu().numpy().reshape(self.action_space)

                q = self.value_net(s_flat)
                q = q.data.cpu().numpy().reshape(self.action_space)

                pi = beta.copy()
                q_temp = q.copy()

                pi_greed = np.zeros(self.action_space)
                pi_greed[np.argmax(q)] = 1
                pi_mix = (1 - self.mix) * pi + self.mix * pi_greed


                pi_mix = self.cmin * pi_mix

                delta = 1 - self.cmin
                while delta > 0:
                    a = np.argmax(q_temp)
                    delta_a = np.min((delta, (self.cmax - self.cmin) * beta[a]))
                    delta -= delta_a
                    pi_mix[a] += delta_a
                    q_temp[a] = -1e11


                pi_mix = pi_mix.clip(0, 1)
                pi_mix = pi_mix / pi_mix.sum()

                pi_explore = self.epsilon * self.pi_rand + (1 - self.epsilon) * pi_mix

                a = np.random.choice(self.action_space, 1, p=pi_explore)
                self.env.step(a)

                states[-1].append(s)
                ts[-1].append(self.env.t)
                policies[-1].append(pi_mix)
                rewards[-1].append(self.env.reward)

                episode_format = np.array((self.frame, states[-1][-1].cpu().numpy(), a, rewards[-1][-1], ts[-1][-1],
                              policies[-1][-1], -1, episode_num), dtype=consts.rec_type)

                episode.append(episode_format)

                self.frame += 1

            if (not self.env.k % self.save_to_mem) or self.env.t:

                episode_df = np.stack(episode)
                trajectory.append(episode_df)

                # write if enough space is available
                if psutil.virtual_memory().available >= mem_threshold:

                    # read
                    fwrite = lock_file(self.writelock)
                    traj_num = np.load(fwrite).item()
                    fwrite.seek(0)
                    np.save(fwrite, traj_num + 1)
                    release_file(fwrite)

                    traj_to_save = np.concatenate(trajectory)
                    traj_to_save['traj'] = traj_num

                    traj_file = os.path.join(self.trajectory_dir, "%d.npy" % traj_num)
                    np.save(traj_file, traj_to_save)

                    fread = lock_file(self.readlock)
                    traj_list = np.load(fread)
                    fread.seek(0)
                    np.save(fread, np.append(traj_list, traj_num))
                    release_file(fread)
                else:
                    assert False, "memory issue"

                if self.env.t:
                    fwrite = lock_file(self.episodelock)
                    episode_num = np.load(fwrite).item()
                    fwrite.seek(0)
                    np.save(fwrite, episode_num + 1)
                    release_file(fwrite)

                    self.env.reset()

            yield {'frames': self.frame}

    def multiplay(self):

        n_players = self.n_players

        mp_env = [Env() for _ in range(n_players)]
        self.frame = 0

        range_players = np.arange(n_players)
        rewards = [[[]] for _ in range(n_players)]
        states = [[[]] for _ in range(n_players)]
        episode = [[] for _ in range(n_players)]
        ts = [[[]] for _ in range(n_players)]
        policies = [[[]] for _ in range(n_players)]
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

        for n in tqdm(itertools.count()):

            if not (n % self.load_memory_interval):
                try:
                    aux = self.load_checkpoint(self.snapshot_path)
                except:
                    pass

                self.beta_net.eval()
                self.value_net.eval()

            s = torch.cat([env.state.unsqueeze(0) for env in mp_env]).to(self.device)
            s_flat = s.view(-1, self.action_space * self.action_space)

            beta = self.beta_net(s_flat)
            beta = F.softmax(beta.detach(), dim=1)
            beta = beta.data.cpu().numpy().reshape(-1,self.action_space)

            q = self.value_net(s_flat)
            q = q.data.cpu().numpy().reshape(-1,self.action_space)

            pi = beta.copy()
            q_temp = q.copy()

            pi_greed = np.zeros((n_players, self.action_space))
            pi_greed[range(n_players), np.argmax(q_temp, axis=1)] = 1
            pi_mix = (1 - self.mix) * pi + self.mix * pi_greed

            pi_mix = self.cmin * pi_mix

            rank = np.argsort(q_temp, axis=1)

            delta = np.ones(n_players) - self.cmin
            for i in range(self.action_space):
                a = rank[:, self.action_space - 1 - i]
                delta_a = np.minimum(delta, (self.cmax - self.cmin) * beta[range_players, a])
                delta -= delta_a
                pi[range_players, a] += delta_a


            pi_mix = pi_mix.clip(0, 1)
            pi_mix = pi_mix / np.repeat(pi_mix.sum(axis=1, keepdims=True), self.action_space, axis=1)

            pi_explore = self.epsilon * self.pi_rand + (1 - self.epsilon) * pi_mix

            pi_explore = pi_explore.astype(np.float32)

            for i in range(n_players):

                a = np.random.choice(self.action_space, 1, p=pi_explore[i])

                env = mp_env[i]

                env.step(a)

                rewards[i][-1].append(env.reward)
                states[i][-1].append(s[i])
                ts[i][-1].append(env.t)
                policies[i][-1].append(pi_mix[i])

                episode_format = np.array((self.frame, states[i][-1][-1].cpu().numpy(), a, rewards[i][-1][-1], ts[i][-1][-1],
                                           policies[i][-1][-1], -1, episode_num[i]), dtype=consts.rec_type)

                episode[i].append(episode_format)

                self.frame += 1

                if (not env.k % self.save_to_mem) or env.t:

                    print("player {} - acc {}, n-offset {}, frame {}, episode {} k {}".format(i, env.acc, self.n_offset, self.frame, episode_num[i], env.k))

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

            self.frame += 1
            if not self.frame % self.player_replay_size:
                yield True
                if self.n_offset >= self.n_tot:
                    break

    def demonstrate(self, n_tot):
        return

    def train(self, n_interval, n_tot):
        return

    def evaluate(self, n_interval, n_tot):
        return
