import numpy as np
import os
from config import consts, args, DirsAndLocksSingleton
import torch
import time
import shutil
import itertools


class Agent(object):

    def __init__(self, exp_name, checkpoint_value=None, checkpoint_beta=None, player=False):
        # parameters
        self.update_memory_interval = args.update_memory_interval
        self.load_memory_interval = args.load_memory_interval
        self.dirs_locks = DirsAndLocksSingleton(exp_name)
        self.action_space = consts.action_space
        self.n_steps = args.n_steps
        self.player_replay_size = 5 #args.player_replay_size
        self.cmin = args.cmin
        self.cmax = args.cmax
        self.epsilon = float(args.epsilon * self.action_space / (self.action_space - 1))
        self.eta = args.eta
        self.delta = args.delta
        self.player = player
        self.cuda_id = args.cuda_default
        self.batch = args.batch
        self.replay_memory_size = args.replay_memory_size
        self.n_players = args.n_players
        self.n_tot = args.n_tot
        self.n_rand = args.n_rand
        self.save_beta = args.save_beta
        self.actor_index = args.actor_index
        self.beta_lr = args.beta_lr
        self.value_lr = args.value_lr

        self.mix = self.delta
        self.gamma = args.gamma

        self.checkpoint_value = checkpoint_value
        self.checkpoint_beta = checkpoint_beta
        self.root_dir = self.dirs_locks.root
        self.explore_dir = self.dirs_locks.explore_dir
        self.list_dir = self.dirs_locks.list_dir
        self.writelock = self.dirs_locks.writelock
        self.episodelock = self.dirs_locks.episodelock
        self.device = torch.device("cuda:%d" % self.cuda_id)

        self.trajectory_dir = self.dirs_locks.trajectory_dir
        self.readlock = self.dirs_locks.readlock

        self.algorithm = args.algorithm
        if not player:
            np.save(self.writelock, 0)
            np.save(self.episodelock, 0)
            np.save(self.readlock, [])

        # if not player:
        #     try:
        #         os.mkdir(self.best_player_dir)
        #     except FileExistsError:
        #         pass

    def save_value_checkpoint(self, path, aux=None):
        raise NotImplementedError

    def load_value_checkpoint(self, path):

        raise NotImplementedError

    def train(self, n_interval, n_tot):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def learn(self, n_interval, n_tot):
        raise NotImplementedError

    # def set_player(self, player, cmin=None, cmax=None, delta=None,
    #                epsilon=None, behavioral_avg_score=None,
    #                behavioral_avg_frame=None, explore_threshold=None):
    #
    #     self.player = player
    #
    #     if epsilon is not None:
    #         self.epsilon = epsilon * self.action_space / (self.action_space - 1)
    #
    #     if cmin is not None:
    #         self.cmin = cmin
    #
    #     if cmax is not None:
    #         self.cmax = cmax
    #
    #     if delta is not None:
    #         self.delta = delta
    #
    #     if explore_threshold is not None:
    #         self.explore_threshold = explore_threshold
    #
    #     if behavioral_avg_score is not None:
    #         self.behavioral_avg_score = behavioral_avg_score
    #
    #     if behavioral_avg_frame is not None:
    #         self.behavioral_avg_frame = behavioral_avg_frame

    def resume(self, model_path):
        aux = self.load_value_checkpoint(model_path)
        return aux

    def clean(self):

        for i in itertools.count():

            time.sleep(2)

            try:
                del_inf = np.load(os.path.join(self.list_dir, "old_explore.npy"))
            except (IOError, ValueError):
                continue
            traj_min = del_inf[0] - 32

            for traj in os.listdir(self.trajectory_dir):
                traj_num = int(traj.split(".")[0])
                if traj_num < traj_min:
                    os.remove(os.path.join(self.trajectory_dir, traj))

            if not i % 50:
                try:
                    self.load_value_checkpoint(self.checkpoint_value)
                    if self.n_offset >= args.n_tot:
                        break
                except:
                    pass
        time.sleep(260)
        shutil.rmtree(self.explore_dir)
        shutil.rmtree(self.list_dir)
        #shutil.rmtree(self.root_dir)
