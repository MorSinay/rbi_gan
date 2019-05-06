#import time
import os
import sys
import numpy as np
#import pandas as pd

from tensorboardX import SummaryWriter

from tqdm import tqdm
import time

from config import consts, args, DirsAndLocksSingleton
from gan_rl_agent import GANAgent as GanActionAgent
from gan_rl_policy_agent import GANAgent as GanPolicyAgent

from logger import logger
from distutils.dir_util import copy_tree


class Experiment(object):

    def __init__(self, logger_file):

        # parameters

        dirs = os.listdir(consts.outdir)

        self.load_model = args.load_last_model or args.load_best_model
        self.load_best = args.load_best_model
        self.load_last = args.load_last_model
        self.resume = args.resume
        self.log_scores = args.log_scores

        temp_name = "%s_%s_%s_exp" % (args.game, args.algorithm, args.identifier)
        self.exp_name = ""
        if self.load_model:
            if self.resume >= 0:
                for d in dirs:
                    if "%s_%04d_" % (temp_name, self.resume) in d:
                        self.exp_name = d
                        self.exp_num = self.resume
                        break
            elif self.resume == -1:

                ds = [d for d in dirs if temp_name in d]
                ns = np.array([int(d.split("_")[-3]) for d in ds])
                self.exp_name = ds[np.argmax(ns)]
            else:
                raise Exception("Non-existing experiment")

        if not self.exp_name:
            # count similar experiments
            n = max([-1] + [int(d.split("_")[-3]) for d in dirs if temp_name in d]) + 1
            self.exp_name = "%s_%04d_%s" % (temp_name, n, consts.exptime)
            self.load_model = False
            self.exp_num = n

        # init experiment parameters
        self.dirs_locks = DirsAndLocksSingleton(self.exp_name)

        self.root = self.dirs_locks.root
        self.indir = self.dirs_locks.indir

        # set dirs
        self.tensorboard_dir = self.dirs_locks.tensorboard_dir
        self.checkpoints_dir = self.dirs_locks.checkpoints_dir
        self.results_dir = self.dirs_locks.results_dir
        self.code_dir = self.dirs_locks.code_dir
        self.analysis_dir = self.dirs_locks.analysis_dir
        self.checkpoint = self.dirs_locks.checkpoint
        self.checkpoint_best = self.dirs_locks.checkpoint_best
        self.replay_dir = self.dirs_locks.replay_dir
        self.scores_dir = self.dirs_locks.scores_dir

        if self.load_model:
            logger.info("Resuming existing experiment")
            with open("logger", "a") as fo:
                fo.write("%s resume\n" % logger_file)
        else:
            logger.info("Creating new experiment")
            # copy code to dir
            copy_tree(os.path.abspath("."), self.code_dir)

            # write args to file
            filename = os.path.join(self.root, "args.txt")
            with open(filename, 'w') as fp:
                fp.write('\n'.join(sys.argv[1:]))

            with open(os.path.join(self.root, "logger"), "a") as fo:
                fo.write("%s\n" % logger_file)

        # initialize tensorboard writer
        if args.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, comment=args.identifier)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if args.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()

    def choose_agent(self):
        if args.algorithm == 'action':
            return GanActionAgent
        elif args.algorithm == 'policy':
            return GanPolicyAgent
        else:
            print(args.algorithm)
            raise ImportError

    def learn(self):

        # init time variables

        agent = self.choose_agent()(self.exp_name, checkpoint=self.checkpoint)

        # load model
        if self.load_model:
            if self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            else:
                raise NotImplementedError
            n_offset = aux['n']
        else:
            n_offset = 0
            # save a random init checkpoint
            agent.save_checkpoint(self.checkpoint, {'n': 0})

        # define experiment generators
        learn = agent.learn(args.checkpoint_interval, args.n_tot)
        agent.save_checkpoint(agent.snapshot_path, {'n': agent.n_offset})

        batch_explore = args.batch

        hold = 1
        while hold:
            print("wait for first samples")


            if len(os.listdir(self.dirs_locks.trajectory_dir)) > 0:
                hold = 0

            time.sleep(5)


          #  if len(os.listdir(self.dirs_locks.trajectory_dir)) >= (int(500. / args.player_replay_size * batch_explore) + 1):
           #     hold = 0

            #time.sleep(5)

        logger.info("Begin Behavioral Distributional learning experiment")
        logger.info("Game: %s " % args.game)

        for n, train_results in tqdm(enumerate(learn)):

            print("print_learn_experiment")

            n = n * args.checkpoint_interval

            avg_train_loss_beta = np.mean(train_results['loss_beta'])
            avg_train_loss_v_beta = np.mean(train_results['loss_q'])
            avg_train_loss_std = np.mean(train_results['loss_std'])

            # log to tensorboard
            if args.tensorboard:
                self.writer.add_scalar('train_loss/loss_beta', float(avg_train_loss_beta), n + n_offset)
                self.writer.add_scalar('train_loss/loss_value', float(avg_train_loss_v_beta), n + n_offset)
                self.writer.add_scalar('train_loss/loss_std', float(avg_train_loss_std), n + n_offset)

                #self.writer.add_scalar('states/state', train_results['s'], n)
                self.writer.add_scalar('actions/reward', train_results['r'][-1], n)
                self.writer.add_histogram("actions/a_player", train_results['a_player'], n + n_offset, 'doane')

                if hasattr(agent, "beta_net"):
                    for name, param in agent.beta_net.named_parameters():
                        self.writer.add_histogram("beta_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset,
                                                  'fd')
                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("value_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset,
                                                  'fd')
            self.print_actions_statistics(train_results['pi'], train_results['pi_tag'] , train_results['beta'],
                                          train_results['q'], train_results['a_player'], avg_train_loss_beta,
                                          avg_train_loss_v_beta)
            agent.save_checkpoint(self.checkpoint, {'n': n + n_offset})

        print("End Learn")
        return agent

    def get_player(self, agent):
        return
        # if os.path.isdir(agent.best_player_dir) and os.listdir(agent.best_player_dir):
        #     max_n = 0
        #
        #     for stat_file in os.listdir(agent.best_player_dir):
        #
        #         while True:
        #             try:
        #                 data = np.load(os.path.join(agent.best_player_dir, stat_file)).item()
        #                 break
        #             except OSError:
        #                 time.sleep(0.1)
        #
        #         if max_n <= data['n']:
        #             max_n = data['n']
        #             player_stats = data['statistics']
        #
        #     # fix choice
        #     for pt in player_stats:
        #         if pt == "reroute":
        #             player_type = pt
        #             break
        #
        #     return player_stats[player_type]

        return None

    def multiplay(self):
        agent = self.choose_agent()(self.exp_name, player=True, checkpoint=self.checkpoint)
        multiplayer = agent.multiplay()

        for _ in multiplayer:
            pass
            #print("player X finished")
            #player = self.get_player(agent)
            #if player:
                # agent.set_player(player['player'], behavioral_avg_score=player['high'],
                #                  behavioral_avg_frame=player['frames'])

    def multiplay_random(self):
        agent = self.choose_agent()(self.exp_name, player=True, checkpoint=self.checkpoint)
        multiplay_random = agent.multiplay_random()

        for _ in multiplay_random:
            pass
            # print("player X finished")
            # player = self.get_player(agent)
            # if player:
            # agent.set_player(player['player'], behavioral_avg_score=player['high'],
            #                  behavioral_avg_frame=player['frames'])

    def play(self, params=None):

        uuid = "%012d" % np.random.randint(1e12)
        agent = self.choose_agent()(self.exp_name, player=True, checkpoint=self.checkpoint)
        aux = agent.resume(self.checkpoint)

        n = aux['n']
        results = {"n" : n, "score": [], "frame": []}

        player = agent.play(args.play_episodes_interval)

        for i, step in tqdm(enumerate(player)):
            results["frame"].append(step['frames'])
            print("experiment play - frames: %d" % (step['frames']))

        if self.log_scores:
            logger.info("Save NPY file: eval_%d_%s.npy" % (n, uuid))
            filename = os.path.join(self.scores_dir, "eval_%d_%s" % (n, uuid))
            np.save(filename, results)

    def clean(self):

        agent = self.choose_agent()(self.exp_name, player=True, checkpoint=self.checkpoint, choose=True)
        agent.clean()

    def print_actions_statistics(self, pi_player, pi_tag_player, beta_player, q_player, a_player, loss_q, loss_beta):

        # print action meanings
        logger.info("Actions statistics: \tloss_beta = %f |\t loss_q = %f |" % (loss_beta, loss_q))

        n_actions = len(a_player)
        applied_player_actions = (np.bincount(np.concatenate((a_player, np.arange(consts.action_space)))) - 1) / n_actions

        line = ''
        line += "|\tPlayer actions\t"
        for a in applied_player_actions:
            line += "|%.2f\t" % (a*100)
        line += "|"
        logger.info(line)

        pi_tag = "|\tpolicy pi tag\t"
        pi =     "|\tpolicy pi    \t"
        beta =   "|\tpolicy beta  \t"
        q =      "|\tvalue q      \t"
        for i in range(consts.action_space):
            pi_tag += "|%.2f\t" % pi_tag_player[i]
            pi += "|%.2f\t" % pi_player[i]
            beta += "|%.2f\t" % beta_player[i]
            q += "|%.2f\t" % q_player[i]
        pi += "|"
        pi_tag += "|"
        beta += "|"
        q += "|"
        logger.info(pi)
        logger.info(pi_tag)
        logger.info(beta)
        logger.info(q)

    def evaluate(self):
        agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint)
        player = agent.evaluate()

        for n, train_results in tqdm(enumerate(player)):

            frame = train_results['n']
            print("print_evaluation_experiment - |n:{}\t|frame:{}\t|acc:{}\t|k:{}\t|".format(n, frame, train_results['acc'], train_results['k']))

            # log to tensorboard
            if args.tensorboard:

                self.writer.add_scalar('evaluation/acc', train_results['acc'], frame)
                self.writer.add_scalar('evaluation/k', train_results['k'], frame)
                self.writer.add_histogram("evaluation/policy/pi", train_results['pi'], frame, 'doane')
                self.writer.add_histogram("evaluation/policy/beta", train_results['beta'], frame, 'doane')
                self.writer.add_histogram("evaluation/policy/value", train_results['q'], frame, 'doane')

                if hasattr(agent, "beta_net"):
                    for name, param in agent.beta_net.named_parameters():
                        self.writer.add_histogram("evaluation/beta_net/%s" % name, param.clone().cpu().data.numpy(), frame, 'fd')
                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("evaluation/value_net/%s" % name, param.clone().cpu().data.numpy(), frame, 'fd')

            pi = "|\tpolicy pi\t"
            beta = "|\tpolicy beta\t"
            q = "|\tvalue q\t    "
            for i in range(consts.action_space):
                pi += "|%.2f\t" % train_results['pi'][i]
                beta += "|%.2f\t" % train_results['beta'][i]
                q += "|%.2f\t" % train_results['q'][i]
            pi += "|"
            beta += "|"
            q += "|"
            logger.info(pi)
            logger.info(beta)
            logger.info(q)

        print("End evaluation")

    def evaluate_last_rl(self, params=None):
        agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint)

        # load model
        try:
            if params is not None:
                aux = agent.resume(params)
            elif self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            else:
                raise NotImplementedError
        except:  # when reading and writing collide
            time.sleep(2)
            if params is not None:
                aux = agent.resume(params)
            elif self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            else:
                raise NotImplementedError

        player = agent.evaluate_last_rl(1)

        for n, train_results in tqdm(enumerate(player)):

            print("print_evaluation_experiment")

            # log to tensorboard
            if args.tensorboard:

                res_size = train_results['s'].shape[0]
                for i in range(res_size):
                    #self.writer.add_scalar('evaluation/states/state', train_results['s'][i], i)
                    self.writer.add_scalar('evaluation/actions/reward', train_results['r'][i], i)
                    self.writer.add_scalar('evaluation/actions/acc', train_results['acc'][i], i)
                self.writer.add_histogram("evaluation/actions/a_player", train_results['a_player'], n, 'doane')

                if hasattr(agent, "beta_net"):
                    for name, param in agent.beta_net.named_parameters():
                        self.writer.add_histogram("evaluation/beta_net/%s" % name, param.clone().cpu().data.numpy(), n, 'fd')
                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("evaluation/value_net/%s" % name, param.clone().cpu().data.numpy(), n, 'fd')


    def evaluate_random_policy(self, params=None):
        agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint)

        # load model
        try:
            if params is not None:
                aux = agent.resume(params)
            elif self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            else:
                raise NotImplementedError
        except:  # when reading and writing collide
            time.sleep(2)
            if params is not None:
                aux = agent.resume(params)
            elif self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            else:
                raise NotImplementedError

        player = agent.evaluate_random_policy(1)

        for n, train_results in tqdm(enumerate(player)):

            print("print_evaluation_random_policy_experiment")

            # log to tensorboard
            if args.tensorboard:

                res_size = train_results['s'].shape[0]
                for i in range(res_size):
                    #self.writer.add_scalar('evaluation_random_policy/states/state', train_results['s'][i], i)
                    self.writer.add_scalar('evaluation_random_policy/actions/reward', train_results['r'][i], i)
                    self.writer.add_scalar('evaluation_random_policy/actions/acc', train_results['acc'][i], i)
                self.writer.add_histogram("evaluation_random_policy/actions/a_player", train_results['a_player'], n, 'doane')

                if hasattr(agent, "beta_net"):
                    for name, param in agent.beta_net.named_parameters():
                        self.writer.add_histogram("evaluation_random_policy/beta_net/%s" % name, param.clone().cpu().data.numpy(), n, 'fd')
                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("evaluation_random_policy/value_net/%s" % name, param.clone().cpu().data.numpy(), n, 'fd')
