import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn as nn

from config import consts, args
import psutil
import socket

from model import BehavioralNet, DuelNet

from memory_gan import Memory, ReplayBatchSampler
from agent import Agent
from environment import Env
#from preprocess import release_file, lock_file, get_mc_value, get_td_value, h_torch, hinv_torch, get_expected_value, get_tde
import os
import time
import shutil

mem_threshold = consts.mem_threshold


class GANAgent(Agent):

    def __init__(self, player=False, choose=False, checkpoint=None):

        print("Learning with GANAgent")
        super(GANAgent, self).__init__(checkpoint)

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.beta_net = BehavioralNet()
        self.value_net = DuelNet()
        self.target_net = DuelNet()

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

        else:
            # datasets
            self.train_dataset = Memory()
            self.train_sampler = ReplayBatchSampler()
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
            assert (False), "save_checkpoint error"
            # state = {'beta_net': self.beta_net.module.state_dict(),
            #          'value_net': self.value_net.module.state_dict(),
            #          'target_net': self.target_net.module.state_dict(),
            #          'optimizer_value': self.optimizer_value.state_dict(),
            #          'optimizer_beta': self.optimizer_beta.state_dict(),
            #          'aux': aux}
        else:
            state = {'beta_net': self.beta_net.state_dict(),
                     'value_net': self.value_net.state_dict(),
                     'target_net': self.target_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_beta': self.optimizer_beta.state_dict(),
                     'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        if torch.cuda.device_count() > 1:
            assert (False), "load_checkpoint error"
            # self.beta_net.module.load_state_dict(state['beta_net'])
            # self.value_net.module.load_state_dict(state['value_net'])
            # self.target_net.module.load_state_dict(state['target_net'])
        else:
            self.beta_net.load_state_dict(state['beta_net'])
            self.value_net.load_state_dict(state['value_net'])
            self.target_net.load_state_dict(state['target_net'])

        self.optimizer_beta.load_state_dict(state['optimizer_beta'])
        self.optimizer_value.load_state_dict(state['optimizer_value'])
        self.n_offset = state['aux']['n']

        return state['aux']

    def play(self, n_tot):
        trajectory_dir = consts.trajectory_dir
        readlock = consts.readlock
        # set initial episodes number
        # lock read
        fwrite = open(self.episodelock, "r+b")
        current_num = np.load(fwrite).item()
        episode_num = current_num
        fwrite.seek(0)
        np.save(fwrite, current_num + 1)
        fwrite.close()

        for i in range(n_tot):

            self.env.reset()
            episode = []
            trajectory = []
            rewards = [[]]
            states = [[]]
            ts = [[]]
            policys = [[]]

            # while not fix:
            #     try:
            #         self.load_checkpoint(self.snapshot_path)
            #         break
            #     except:
            #         time.sleep(0.5)

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

                s = self.env.s.to(self.device)
                # get aux data

                beta = self.beta_net(s)
                beta = F.softmax(beta.detach(), dim=1)
                beta = beta.data.cpu().numpy()

                q = self.value_net(s)
                q = q.data.cpu().numpy()

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
                policys[-1].append(pi_mix)
                rewards[-1].append(self.env.reward)

                a = np.array((self.frame, states[-1][-1].cpu().numpy(), -1, rewards[-1][-1], ts[-1][-1],
                              policys[-1][-1].cpu().numpy(), -1, episode_num), dtype=consts.rec_type)

                episode.append(a)

                self.frame += 1

            if self.env.t:

                episode_df = np.stack(episode)

                trajectory.append(episode_df)

                print("gan | t: %d\t" % self.env.t)

                # read
                fwrite = open(self.episodelock, "r+b")
                episode_num = np.load(fwrite).item()
                fwrite.seek(0)
                np.save(fwrite, episode_num + 1)
                fwrite.close()

                if sum([len(j) for j in trajectory]) >= self.player_replay_size:

                    # write if enough space is available
                    if psutil.virtual_memory().available >= mem_threshold:
                        # read
                        fwrite = open(self.writelock, "r+b")
                        traj_num = np.load(fwrite).item()
                        fwrite.seek(0)
                        np.save(fwrite, traj_num + 1)
                        fwrite.close()

                        traj_to_save = np.concatenate(trajectory)
                        traj_to_save['traj'] = traj_num

                        traj_file = os.path.join(trajectory_dir, "%d.npy" % traj_num)
                        np.save(traj_file, traj_to_save)

                        fread = open(readlock, "r+b")
                        traj_list = np.load(fread)
                        fread.seek(0)
                        np.save(fread, np.append(traj_list, traj_num))
                        fread.close()

            print("debug only ntot: {} from {}".format(i, n_tot))

            # yield {'frames': self.frame}

    def learn(self, n_interval, n_tot):

        print("start")

        self.beta_net.train()
        self.value_net.train()
        self.target_net.eval()

        results = {'n': [], 'loss_q': [], 'loss_beta': [], 'a_player': [], 'loss_std': [],
                   'r': [], 's': [], 't': [], 'pi': [], 's_tag': [], 'pi_tag': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            print("z")
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
            q = self.value_net(s)
            target_value = r + args.gamma * (pi_tag*q_tag).sum()
            loss_q = self.q_loss(q, target_value)

            self.optimizer_beta.zero_grad()
            loss_beta.backward()
            self.optimizer_beta.step()

            self.optimizer_value.zero_grad()
            loss_q.backward()
            self.optimizer_value.step()

            # collect actions statistics
            if not n % 50:
                print("k")
                # add results
                results['a_player'].append(a[:, 0].data.cpu().numpy())
                results['r'].append(r[:, 0].data.cpu().numpy())
                results['s'].append(s.data.cpu().numpy())
                results['t'].append(t[:, 0].data.cpu().numpy())
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

                    # yield results
                    self.beta_net.train()
                    self.value_net.train()
                    results = {key: [] for key in results}

                    if n >= n_tot:
                        break

                # TODO: save result?

    def multiplay(self):
        return

    def demonstrate(self, n_tot):
        return

    def train(self, n_interval, n_tot):
        return

    def evaluate(self, n_interval, n_tot):
        return
