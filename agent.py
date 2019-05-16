import numpy as np
import os
from config import consts, args, DirsAndLocksSingleton
import torch
import time
import shutil
import itertools


class Agent(object):

    def __init__(self, exp_name, checkpoint=None, player=False):
        # parameters
        # self.discount = args.discount
        self.update_target_interval = args.update_target_interval
        self.update_memory_interval = args.update_memory_interval
        self.load_memory_interval = args.load_memory_interval
        self.dirs_locks = DirsAndLocksSingleton(exp_name)
        self.action_space = consts.action_space
        # self.skip = args.skip
        # self.termination_reward = args.termination_reward
        self.n_steps = args.n_steps
        #self.reward_shape = args.reward_shape
        self.player_replay_size = 5 #args.player_replay_size
        self.cmin = args.cmin
        self.cmax = args.cmax
        #self.history_length= args.history_length
        #self.random_initialization = args.random_initialization
        self.epsilon = float(args.epsilon * self.action_space / (self.action_space - 1))
        self.eta = args.eta
        self.delta = args.delta
        self.player = player
        #self.priority_beta = args.priority_beta
        #self.priority_alpha = args.priority_alpha
        #self.epsilon_a = args.epsilon_a
        self.cuda_id = args.cuda_default
        #self.behavioral_avg_frame = 1
        #self.behavioral_avg_score = -1
        #self.entropy_loss = float((1 - (1 / (1 + (self.action_space - 1) * np.exp(-args.softmax_diff)))) * (self.action_space / (self.action_space - 1)))
        self.batch = args.batch
        self.replay_memory_size = args.replay_memory_size
        #self.actor_index = args.actor_index
        self.n_players = args.n_players
        #self.player = args.player
        self.n_tot = args.n_tot
        self.n_rand = args.n_rand
        #self.max_length = consts.max_length[args.game]
        #self.max_score = consts.max_score[args.game]
        #self.start_time = consts.start_time

        self.mix = self.delta
        #self.min_loop = 1. / 44
        #self.hidden_state = args.hidden_features_rnn

        #self.seq_length = args.seq_length
        #if args.target == 'tde':
        #    self.seq_length += self.n_steps

        #self.burn_in = args.burn_in
        #self.seq_overlap = args.seq_overlap

        #self.rec_type = consts.rec_type

        self.checkpoint = checkpoint
        self.root_dir = self.dirs_locks.root
        # self.best_player_dir = os.path.join(root_dir, "best")
        self.snapshot_path = self.dirs_locks.snapshot_path
        self.explore_dir = self.dirs_locks.explore_dir
        self.list_dir = self.dirs_locks.list_dir
        self.writelock = self.dirs_locks.writelock
        self.episodelock = self.dirs_locks.episodelock
        self.device = torch.device("cuda:%d" % self.cuda_id)

        self.trajectory_dir = self.dirs_locks.trajectory_dir
#        self.screen_dir = self.dirs_locks.screen_dir
        self.readlock = self.dirs_locks.readlock

        np.save(self.writelock, 0)
        np.save(self.episodelock, 0)
        np.save(self.readlock, [])

        # if not player:
        #     try:
        #         os.mkdir(self.best_player_dir)
        #     except FileExistsError:
        #         pass

    def save_checkpoint(self, path, aux=None):
        raise NotImplementedError

    def load_checkpoint(self, path):

        raise NotImplementedError

    def train(self, n_interval, n_tot):
        raise NotImplementedError

    def evaluate(self):
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
        aux = self.load_checkpoint(model_path)
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
                    self.load_checkpoint(self.snapshot_path)
                    if self.n_offset >= args.n_tot:
                        break
                except:
                    pass
        time.sleep(15)
        shutil.rmtree(self.explore_dir)
        shutil.rmtree(self.list_dir)
        #shutil.rmtree(self.root_dir)
