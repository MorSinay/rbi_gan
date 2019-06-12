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

from model import BehavioralNet as BehavioralNetRbi
from model import DuelNet as DuelNetRbi
from model_ddpg import BehavioralNet as BehavioralNetDdpg
from model_ddpg import DuelNet as DuelNetDdpg

from memory_gan import Memory, ReplayBatchSampler
from agent import Agent
from environment import Env
import os
import time

import itertools
mem_threshold = consts.mem_threshold


class GANAgent(Agent):

    def __init__(self, exp_name, player=False, choose=False, checkpoint=None):

        reward_str = args.reward.upper() + " with " + args.acc.upper()
        print("Learning POLICY method using {} with GANAgent".format(reward_str))
        super(GANAgent, self).__init__(exp_name, checkpoint)

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.beta_net = choose_behavioral_net()()
        self.beta_target_net = choose_behavioral_net()()
        self.value_net = choose_duel_net()()
        self.target_net = choose_duel_net()()

        if torch.cuda.device_count() > 1:
            self.beta_net = nn.DataParallel(self.beta_net)
            self.beta_target_net = nn.DataParallel(self.beta_target_net)
            self.value_net = nn.DataParallel(self.value_net)
            self.target_net = nn.DataParallel(self.target_net)

        self.beta_net.to(self.device)
        self.beta_target_net.to(self.device)
        self.value_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.beta_target_net.load_state_dict(self.beta_net.state_dict())

        self.pi_rand = np.ones(self.action_space, dtype=np.float32) / self.action_space
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
        self.optimizer_beta = torch.optim.Adam(self.beta_net.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.n_offset = 0

    def save_checkpoint(self, path, aux=None):

        if torch.cuda.device_count() > 1:
            state = {'beta_net': self.beta_net.module.state_dict(),
                     'beta_target_net': self.beta_target_net.module.state_dict(),
                     'value_net': self.value_net.module.state_dict(),
                     'target_net': self.target_net.module.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_beta': self.optimizer_beta.state_dict(),
                     'aux': aux}
        else:
            state = {'beta_net': self.beta_net.state_dict(),
                     'beta_target_net': self.beta_target_net.state_dict(),
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
            self.beta_target_net.module.load_state_dict(state['beta_target_net'])
            self.value_net.module.load_state_dict(state['value_net'])
            self.target_net.module.load_state_dict(state['target_net'])
        else:
            self.beta_net.load_state_dict(state['beta_net'])
            self.beta_target_net.load_state_dict(state['beta_target_net'])
            self.value_net.load_state_dict(state['value_net'])
            self.target_net.load_state_dict(state['target_net'])

        self.optimizer_beta.load_state_dict(state['optimizer_beta'])
        self.optimizer_value.load_state_dict(state['optimizer_value'])
        self.n_offset = state['aux']['n']

        return state['aux']

    def learn_rbi(self, n_interval, n_tot):
        self.save_checkpoint(self.snapshot_path, {'n': 0})

        self.beta_net.train()
        self.beta_target_net.eval()
        self.value_net.train()
        self.target_net.eval()

        results = {'n': [], 'loss_q': [], 'loss_beta': [], 'a_player': [], 'loss_std': [],
                   'r': [], 's': [], 't': [], 'pi': [], 's_tag': [], 'pi_tag': [], 'acc': [], 'beta': [],
                   'q': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = sample['s'].to(self.device, non_blocking=True)
            a = sample['a'].to(self.device, non_blocking=True)
            r = sample['r'].to(self.device, non_blocking=True)
            t = sample['t'].to(self.device, non_blocking=True)
            pi_explore = sample['pi_explore'].to(self.device, non_blocking=True)
            pi = sample['pi'].to(self.device, non_blocking=True)
            acc = sample['acc'].to(self.device, non_blocking=True)
            s_tag = sample['s_tag'].to(self.device, non_blocking=True)
            pi_tag = sample['pi_tag'].to(self.device, non_blocking=True)

            beta = self.beta_net(s)
            beta_policy = F.softmax(beta.detach(), dim=1)  # debug
            beta_log = F.log_softmax(beta, dim=1)
            loss_beta = (-beta_log * pi).sum(dim=1).mean()

            # dqn
            x_tag = self.target_net(s_tag).detach()
            x = self.value_net(s)

            if args.rl_metric == 'mc':
                target_value = r
            else:
                target_value = r + (1 - t) * self.gamma * (pi_tag * x_tag).sum(dim=1)

            # pi_explore = (self.epsilon * self.pi_rand + (1 - self.epsilon) * pi).to(self.device)
            value = (pi_explore * x).sum(dim=1)
            # value = (pi * x).sum(dim=1)
            loss_q = self.q_loss(value, target_value).mean()

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
                results['acc'].append(acc.data.cpu().numpy())
                results['s_tag'].append(s_tag.data.cpu().numpy())
                results['pi_tag'].append(pi_tag.data.cpu().numpy())
                results['beta'].append(beta_policy.cpu().numpy())
                results['q'].append(x.data.cpu().numpy())

                # add results
                results['loss_beta'].append(loss_beta.data.cpu().numpy())
                results['loss_q'].append(loss_q.data.cpu().numpy())
                results['loss_std'].append(0)

                if not n % self.update_memory_interval:
                    # save agent state
                    self.save_checkpoint(self.snapshot_path, {'n': n})

                if not n % self.update_target_interval:
                    # save agent state
                    self.target_net.load_state_dict(self.value_net.state_dict())
                    self.beta_target_net.load_state_dict(self.beta_net.state_dict())

                if not n % n_interval:
                    results['a_player'] = np.concatenate(results['a_player'])
                    results['r'] = np.concatenate(results['r'])
                    results['s'] = np.concatenate(results['s'])
                    results['t'] = np.concatenate(results['t'])
                    #results['pi'] = np.concatenate(results['pi'])
                    results['acc'] = np.average(np.concatenate(results['acc']))
                    results['s_tag'] = np.concatenate(results['s_tag'])
                   # results['pi_tag'] = np.concatenate(results['pi_tag'])

                    results['pi'] = np.average(np.concatenate(results['pi']), axis=0).flatten()
                    results['pi_tag'] = np.average(np.concatenate(results['pi_tag']), axis=0).flatten()
                    results['beta'] = np.average(np.concatenate(results['beta']), axis=0).flatten()
                    results['q'] = np.average(np.concatenate(results['q']), axis=0).flatten()
                    results['n'] = n

                    yield results
                    self.beta_net.train()
                    self.value_net.train()
                    results = {key: [] for key in results}

                    if n >= n_tot:
                        self.save_checkpoint(self.snapshot_path, {'n': n})
                        break

        print("Learn Finish")


    def learn_ddpg(self, n_interval, n_tot):
        self.save_checkpoint(self.snapshot_path, {'n': 0})

        self.beta_net.train()
        self.beta_target_net.eval()
        self.value_net.train()
        self.target_net.eval()

        results = {'n': [], 'loss_q': [], 'loss_beta': [], 'a_player': [], 'loss_std': [],
                   'r': [], 's': [], 't': [], 'pi': [], 's_tag': [], 'pi_tag': [], 'acc': [], 'beta': [],
                   'q': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = sample['s'].to(self.device, non_blocking=True)
            assert(s.shape[-1] == 100), "state error {}".format(s.shape)
            a = sample['a'].to(self.device, non_blocking=True)
            r = sample['r'].to(self.device, non_blocking=True)
            t = sample['t'].to(self.device, non_blocking=True)
            pi_explore = sample['pi_explore'].to(self.device, non_blocking=True)
            pi = sample['pi'].to(self.device, non_blocking=True)
            acc = sample['acc'].to(self.device, non_blocking=True)
            s_tag = sample['s_tag'].to(self.device, non_blocking=True)
            pi_tag = sample['pi_tag'].to(self.device, non_blocking=True)

            self.optimizer_beta.zero_grad()
            self.optimizer_value.zero_grad()

            beta = self.beta_net(s)
            beta_policy = F.softmax(beta, dim=1)
            q_value = self.value_net(s, pi_explore).view(-1)

            if args.rl_metric == 'mc':
                target_value = r
            else:
                q_value_tag = self.target_net(s_tag, F.softmax(self.beta_target_net(s), dim=1).detach()).detach()
                target_value = r + (1 - t) * self.gamma * q_value_tag.view(-1)

            loss_q = self.q_loss(q_value, target_value).mean()
            loss_q.backward()
            self.optimizer_value.step()

            loss_beta = -self.value_net(s, beta_policy).mean()
            loss_beta.backward()
            self.optimizer_beta.step()

            # Target update
            soft_update(self.target_net, self.value_net, self.tau)
            soft_update(self.beta_target_net, self.beta_net, self.tau)

            # collect actions statistics
            if not n % 50:
                # add results
                results['a_player'].append(a.data.cpu().numpy())
                results['r'].append(r.data.cpu().numpy())
                results['s'].append(s.data.cpu().numpy())
                results['t'].append(t.data.cpu().numpy())
                results['pi'].append(pi.data.cpu().numpy())
                results['acc'].append(acc.data.cpu().numpy())
                results['s_tag'].append(s_tag.data.cpu().numpy())
                results['pi_tag'].append(pi_tag.data.cpu().numpy())
                results['beta'].append(beta_policy.detach().cpu().numpy())
                results['q'].append(q_value.detach().cpu().numpy())

                # add results
                results['loss_beta'].append(loss_beta.data.cpu().numpy())
                results['loss_q'].append(loss_q.data.cpu().numpy())
                results['loss_std'].append(0)

                if not n % self.update_memory_interval:
                    # save agent state
                    self.save_checkpoint(self.snapshot_path, {'n': n})

                # if not n % self.update_target_interval:
                #     # save agent state
                #     self.target_net.load_state_dict(self.value_net.state_dict())
                #     self.beta_target_net.load_state_dict(self.beta_net.state_dict())

                if not n % n_interval:
                    results['a_player'] = np.concatenate(results['a_player'])
                    results['r'] = np.concatenate(results['r'])
                    results['s'] = np.concatenate(results['s'])
                    results['t'] = np.concatenate(results['t'])
                    #results['pi'] = np.concatenate(results['pi'])
                    results['acc'] = np.average(np.concatenate(results['acc']))
                    results['s_tag'] = np.concatenate(results['s_tag'])
                   # results['pi_tag'] = np.concatenate(results['pi_tag'])

                    results['pi'] = np.average(np.concatenate(results['pi']), axis=0).flatten()
                    results['pi_tag'] = np.average(np.concatenate(results['pi_tag']), axis=0).flatten()
                    results['beta'] = np.average(np.concatenate(results['beta']), axis=0).flatten()
                    results['q'] = np.average(np.concatenate(results['q']), axis=0).flatten()
                    results['n'] = n

                    yield results
                    self.beta_net.train()
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

                self.beta_net.eval()
                self.value_net.eval()

            s = torch.cat([env.state.unsqueeze(0) for env in mp_env]).to(self.device)
            assert (s.shape[-1] == self.action_space*self.action_space), "state error {}".format(s.shape)

            if self.n_offset <= self.n_rand:
                pi_explore = np.repeat(self.pi_rand, n_players, axis=0).reshape(n_players,self.action_space)
                pi = pi_explore
            elif self.algorithm == 'ddpg':
                beta = self.beta_net(s)
                beta = F.softmax(beta.detach(), dim=1)
                beta = beta.data.cpu().numpy().reshape(-1,self.action_space)

                pi = beta.copy()

                #to assume there is no zero policy
                pi = (1 - self.eta) * pi + self.eta / self.action_space

                explore = np.random.rand(n_players, self.action_space)
                #explore = np.exp(explore) / np.sum(np.exp(explore), axis=1).reshape(n_players, 1)
                explore = explore / np.sum(explore, axis=1).reshape(n_players, 1)

                pi_explore = self.epsilon * explore + (1 - self.epsilon) * pi
                pi_explore = pi_explore / np.repeat(pi_explore.sum(axis=1, keepdims=True), self.action_space, axis=1)

            elif self.algorithm == 'rbi':
                beta = self.beta_net(s)
                beta = F.softmax(beta.detach(), dim=1)
                beta = beta.data.cpu().numpy().reshape(-1,self.action_space)

                q = self.value_net(s)
                q = q.data.cpu().numpy().reshape(-1,self.action_space)

                pi = beta.copy()
                q_temp = q.copy()

                #to assume there is no zero policy
                pi = (1 - self.eta) * pi + self.eta / self.action_space
                pi = self.cmin * pi

                rank = np.argsort(q_temp, axis=1)

                delta = np.ones(n_players) - self.cmin
                for i in range(self.action_space):
                    a = rank[:, self.action_space - 1 - i]
                    delta_a = np.minimum(delta, (self.cmax - self.cmin) * beta[range_players, a])
                    delta -= delta_a
                    pi[range_players, a] += delta_a

                #pi_greed = np.zeros((n_players, self.action_space), dtype=np.float32)
                #pi_greed[range(n_players), np.argmax(q_temp, axis=1)] = 1
                #pi = (1 - self.mix) * pi + self.mix * pi_greed
                pi = (1 - self.mix) * pi + self.mix * self.pi_rand
                pi = pi.clip(0, 1)
                pi = pi / np.repeat(pi.sum(axis=1, keepdims=True), self.action_space, axis=1)

                explore = np.random.rand(n_players, self.action_space)
                explore = np.exp(explore) / np.sum(np.exp(explore), axis=1).reshape(n_players, 1)

                pi_explore = self.epsilon * explore + (1 - self.epsilon) * pi
                pi_explore = pi_explore / np.repeat(pi_explore.sum(axis=1, keepdims=True), self.action_space, axis=1)

            else:
                raise ImportError

            for i in range(n_players):

                env = mp_env[i]

                env.step_policy(np.expand_dims(pi_explore[i], axis=0))
                rewards[i][-1].append(env.reward)
                accs[i][-1].append(env.acc)
                states[i][-1].append(s[i])
                ts[i][-1].append(env.t)
                policies[i][-1].append(pi[i])
                explore_policies[i][-1].append(pi_explore[i])

                episode_format = np.array((self.frame, states[i][-1][-1].cpu().numpy(), 0, rewards[i][-1][-1],
                                           accs[i][-1][-1], ts[i][-1][-1], policies[i][-1][-1],
                                           explore_policies[i][-1][-1], -1, episode_num[i]), dtype=consts.rec_type)

                episode[i].append(episode_format)

                if (not env.k % self.save_to_mem) or env.t:
                #if env.t:

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

                        if args.rl_metric == 'mc':
                            for i in range (traj_to_save['r'].shape[0]-2, -1, -1):
                                traj_to_save['r'][i] += self.gamma*traj_to_save['r'][i+1]

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

    def evaluate_pi(self, eval_pi):

        results = {'n': [], 'pi': [], 'beta': [], 'q': [], 'acc': [], 'k': [], 'r': [], 'score': []}

        self.env.reset()
        self.beta_net.eval()
        self.value_net.eval()

        for _ in tqdm(itertools.count()):

            pi = np.expand_dims(eval_pi, axis=0)
            self.env.step_policy(pi)

            results['pi'].append(pi)
            results['beta'].append(np.zeros_like(pi))
            results['q'].append(np.zeros_like(pi))
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
            results['k'] = self.env.k
            results['score'] = (results['pi'] * results['q']).sum(dim=1)
            #results['acc'] = self.env.acc

            yield results
            results = {key: [] for key in results}

    def evaluate_rbi(self):

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
                pass

            self.env.reset()
            self.beta_net.eval()
            self.value_net.eval()

            for _ in tqdm(itertools.count()):

            #while not self.env.t:
                s = self.env.state.to(self.device)

                beta = self.beta_net(s)
                beta = F.softmax(beta.detach(), dim=1)
                beta = beta.data.cpu().numpy().reshape(self.action_space)

                q = self.value_net(s)
                q = q.data.cpu().numpy().reshape(self.action_space)

                if self.n_offset <= self.n_rand:
                    pi = self.pi_rand
                else:
                    pi = beta.copy()
                    q_temp = q.copy()

                    pi = (1 - self.eta) * pi + self.eta * self.pi_rand

                    pi = self.cmin * pi

                    delta = 1 - self.cmin
                    while delta > 0:
                        a = np.argmax(q_temp)
                        delta_a = np.min((delta, (self.cmax - self.cmin) * beta[a]))
                        delta -= delta_a
                        pi[a] += delta_a
                        q_temp[a] = -1e11

                    pi = (1 - self.mix) * pi + self.mix * self.pi_rand
                    pi = pi.clip(0, 1)
                    pi = pi / pi.sum()

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
                results['k'] = self.env.k
                results['score'] = (results['pi'] * results['q']).sum()
                #results['acc'] = self.env.acc

                yield results
                results = {key: [] for key in results}

            if self.n_offset >= args.n_tot:
                break

    def evaluate_ddpg(self):

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
            self.beta_net.eval()
            self.value_net.eval()

            for _ in tqdm(itertools.count()):

            #while not self.env.t:
                s = self.env.state.to(self.device)

                beta = self.beta_net(s)
                beta = F.softmax(beta.detach(), dim=1)

                q = self.value_net(s, beta)

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
                results['k'] = self.env.k
                results['score'] = results['q']
                #results['acc'] = self.env.acc

                yield results
                results = {key: [] for key in results}

            if self.n_offset >= args.n_tot:
                break


def choose_behavioral_net():
    if args.algorithm == 'ddpg':
        return BehavioralNetDdpg
    elif args.algorithm == 'rbi':
        return BehavioralNetRbi
    else:
        print(args.algorithm)
        raise ImportError


def choose_duel_net():
    if args.algorithm == 'ddpg':
        return DuelNetDdpg
    elif args.algorithm == 'rbi':
        return DuelNetRbi
    else:
        print(args.algorithm)
        raise ImportError


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
