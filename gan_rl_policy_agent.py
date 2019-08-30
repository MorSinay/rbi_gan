import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.nn as nn

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

    def __init__(self, exp_name, problem, player=False, choose=False, checkpoint_value=None, checkpoint_beta=None):

        reward_str = "BANDIT"
        print("Learning POLICY method using {} with GANAgent".format(reward_str))
        super(GANAgent, self).__init__(exp_name,problem, checkpoint_value, checkpoint_beta)

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.pi_rand = np.ones(self.action_space, dtype=np.float32) / self.action_space

        self.init = torch.tensor(self.problem.initial_solution, dtype=torch.float).to(self.device)

        self.value_net = DuelNet()
        self.value_net.to(self.device)


        self.q_loss = nn.SmoothL1Loss(reduction='none')

        if player:
            self.beta_net = nn.Parameter(self.init)
            # play variables
            self.env = Env(self.problem)
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

    def reset_beta(self):
        self.beta_net.data = self.init

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
                   'r': [], 't': [], 'pi': [], 'pi_tag': [], 'best_observed': [],
                   'q': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            r = sample['r'].to(self.device, non_blocking=True)
            t = sample['t'].to(self.device, non_blocking=True)
            pi_explore = sample['pi_explore'].to(self.device, non_blocking=True)
            pi = sample['pi'].to(self.device, non_blocking=True)
            best_observed = sample['best_observed'].to(self.device, non_blocking=True)
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
                results['best_observed'].append(best_observed.data.cpu().numpy())
                results['pi_tag'].append(pi_tag.data.cpu().numpy())
                results['q'].append(q_value.detach().cpu().numpy())

                # add results
                results['loss_q'].append(loss_q.data.cpu().numpy())

                if not n % n_interval:
                    results['r'] = np.concatenate(results['r'])
                    results['t'] = np.concatenate(results['t'])
                    results['best_observed'] = np.min(np.concatenate(results['best_observed']))

                    results['pi'] = np.average(np.concatenate(results['pi']), axis=0).flatten()
                    results['pi_tag'] = np.average(np.concatenate(results['pi_tag']), axis=0).flatten()
                    results['q'] = np.average(np.concatenate(results['q']), axis=0).flatten()
                    results['n'] = n

                    yield results
                    self.value_net.train()
                    results = {key: [] for key in results}

                    if n >= n_tot:
                        self.save_value_checkpoint(self.checkpoint_value, {'n': n})
                        break

        print("Learn Finish")

    def grad_explore(self, n_players):
        self.optimizer_beta.zero_grad()
        loss_beta = -self.value_net(self.beta_net)
        loss_beta.backward()

        grads = self.beta_net.grad.detach().cpu().numpy().copy()

        explore_factor = self.delta * grads + self.epsilon * np.random.randn(n_players, self.action_space)
        explore_factor *= 0.9 ** (2*np.array(range(n_players))).reshape(n_players,1)
        beta_explore = self.beta_net.detach().cpu().numpy() + explore_factor

        beta_explore = np.clip(beta_explore, self.problem.lower_bounds, self.problem.upper_bounds)

        return (beta_explore, grads)

    def uniform_explore(self, n_players):

        explore_factor = self.epsilon * np.random.randn(n_players, self.action_space)
        explore_factor *= 0.9 ** (2 * np.array(range(n_players))).reshape(n_players, 1)
        beta_explore = self.beta_net.detach().cpu().numpy() + explore_factor
        pi_explore = np.exp(beta_explore) / np.sum(np.exp(beta_explore), axis=1).reshape(n_players, 1)

        pi_explore = np.clip(pi_explore, self.problem.lower_bounds, self.problem.upper_bounds)
        return pi_explore

    def multiplay(self):

        n_players = self.n_players

        mp_env = [Env(self.problem) for _ in range(n_players)]
        self.frame = 0

        rewards = [[[]] for _ in range(n_players)]
        best_observed = [[[]] for _ in range(n_players)]
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

        grad = np.zeros(self.action_space)

        for self.frame in tqdm(itertools.count()):

            pi = self.beta_net.detach()
            pi = pi.data.cpu().numpy().reshape(-1, self.action_space)
            pi = np.clip(pi, self.problem.lower_bounds, self.problem.upper_bounds)

            if self.n_offset > self.n_rand and args.exploration == 'GRAD':
                (pi_explore, grad) = self.grad_explore(n_players)
            else:
                pi_explore = self.uniform_explore(n_players)

            for i in range(n_players):
                grads[i][-1].append(grad)

                env = mp_env[i]
                env.step_policy(pi_explore[i])
                rewards[i][-1].append(env.reward)
                best_observed[i][-1].append(env.best_obser)
                ts[i][-1].append(env.t)
                policies[i][-1].append(pi)
                explore_policies[i][-1].append(pi_explore[i])

                episode_format = np.array((self.frame, rewards[i][-1][-1],
                                           best_observed[i][-1][-1], ts[i][-1][-1], policies[i][-1][-1],
                                           explore_policies[i][-1][-1], -1, episode_num[i]), dtype=consts.rec_type)

                episode[i].append(episode_format)

                # TODO: BBO look here
                if not (env.k % 500):

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

                        if (i == 0) and (not self.frame % 5000):
                            print("actor {} player {} - best_observed {}, n-offset {}, frame {}, episode {}, r {}, pi_explore {}".format(
                                    self.actor_index, i, env.best_observed, self.n_offset, self.frame, episode_num[i],env.reward, pi_explore[i]))

                            if self.save_beta:  # and self.frame>30:
                                __pi = np.vstack(np.concatenate(np.vstack(policies)))
                                __pi_explore = np.vstack(np.concatenate(np.vstack(explore_policies)))
                                __grads = np.vstack(np.concatenate(np.vstack(grads)))
                                pi_file = os.path.join(self.root_dir, "pi.npy")
                                pi_explore_file = os.path.join(self.root_dir, "pi_explore.npy")
                                grad_file = os.path.join(self.root_dir, "grad.npy")
                                np.save(pi_file, __pi)
                                np.save(pi_explore_file, __pi_explore)
                                np.save(grad_file, __grads)

                        trajectory[i] = []

                    else:
                        assert False, "memory error available memory {}".format(psutil.virtual_memory().available)

                # TODO: BBO look here
                elif env.t:
                    self.save_beta_checkpoint(self.checkpoint_beta, {'n': self.frame})
                    assert False, "finished"

            if mp_env[0].k > self.budget:
                self.reset_beta()
                for j in range(n_players):
                    mp_env[j].reset()
                continue

            if not (self.frame % self.load_memory_interval):
                try:
                    self.load_value_checkpoint(self.checkpoint_value)
                except:
                    pass

                self.value_net.eval()

            if self.n_offset > self.n_rand:
                for _ in range(10):
                    self.optimizer_beta.zero_grad()
                    loss_beta = -self.value_net(self.beta_net)
                    loss_beta.backward()
                    self.optimizer_beta.step()

            if not self.frame % self.player_replay_size:
                beta_policy = self.beta_net.detach().cpu().numpy()
                reward_stack = np.hstack(rewards)
                min_player = np.argmin(reward_stack)
                policy_stack = np.hstack(policies)
                best_observed_stack = np.hstack(best_observed)
                results  = {}
                results['actor'] = self.actor_index
                results['frame'] = self.frame
                results['n_offset'] = self.n_offset
                results['beta'] = beta_policy
                results['best_observed'] = best_observed_stack[0,min_player]
                results['reward'] = reward_stack[0,min_player]
                results['policy'] = policy_stack[0,min_player]

                yield results

                if self.n_offset >= self.n_tot:
                    break

    def demonstrate(self, n_tot):
        return

    def train(self, n_interval, n_tot):
        return

    def evaluate(self):

        try:
            self.load_value_checkpoint(self.checkpoint_value)
        except:
            print("Error in load checkpoint")
            pass

        results = {'n': [], 'pi': [], 'beta': [], 'q': [], 'best_observed': [], 'r': [], 'score': []}

        for _ in tqdm(itertools.count()):
        #for _ in itertools.count():

            try:
                self.load_value_checkpoint(self.checkpoint_value)
            except:
                print("Error in load checkpoint")
                continue

            self.reset_beta()
            self.env.reset()
            self.value_net.eval()

            while self.env.k < self.budget:

                beta = self.beta_net.detach()
                q = self.value_net(beta)

                beta = beta.data.cpu().numpy().reshape(self.action_space)

                q = q.data.cpu().numpy()

                pi = np.clip(beta, self.problem.lower_bounds, self.problem.upper_bounds)
                self.env.step_policy(pi)

                results['pi'].append(pi)
                results['beta'].append(beta)
                results['q'].append(q)
                results['r'].append(self.env.reward)
                results['best_observed'].append(self.env.best_observed)

                if self.env.t:
                    break

                for _ in range(10):
                    self.optimizer_beta.zero_grad()
                    loss_beta = -self.value_net(self.beta_net)
                    loss_beta.backward()
                    self.optimizer_beta.step()

            results['pi'] = np.average(np.asarray(results['pi']), axis=0).flatten()
            results['beta'] = np.average(np.asarray(results['beta']), axis=0).flatten()
            results['q'] = np.average(np.asarray(results['q']), axis=0).flatten()
            results['best_observed'] = np.asarray(results['best_observed'])
            results['r'] = np.asarray(results['r'])

            results['n'] = self.n_offset
            results['score'] = results['q']

            yield results
            results = {key: [] for key in results}

            if self.env.t or self.n_offset >= args.n_tot:
                break

            time.sleep(4)
