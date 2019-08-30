import time
import os
import sys
import numpy as np
#import pandas as pd

from tensorboardX import SummaryWriter

from tqdm import tqdm
import time

from config import consts, args, DirsAndLocksSingleton
#from gan_rl_agent import GANAgent as GanActionAgent
from gan_rl_policy_agent import GANAgent as GanPolicyAgent

from logger import logger
from distutils.dir_util import copy_tree


class Experiment(object):

    def __init__(self, logger_file, problem):

        # parameters

        dirs = os.listdir(consts.outdir)

        self.load_model = args.load_last_model
        self.load_last = args.load_last_model
        self.resume = args.resume
        self.problem = problem

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
        self.checkpoint_value = self.dirs_locks.checkpoint_value
        self.checkpoint_beta = self.dirs_locks.checkpoint_beta
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
        return GanPolicyAgent

    def learn(self):

        # init time variables

        agent = self.choose_agent()(self.exp_name, self.problem, checkpoint_value=self.checkpoint_value, checkpoint_beta=self.checkpoint_beta)
        #
        #         # load model
        if self.load_model:
            try:
                aux = agent.resume(self.checkpoint_value)
                n_offset = aux['n']
            except AssertionError:
                agent.save_value_checkpoint(self.checkpoint_value, {'n': 0})
                n_offset = 0

        else:
            n_offset = 0
            # save a random init checkpoint
            agent.save_value_checkpoint(self.checkpoint_value, {'n': 0})

        # define experiment generators
        learn = agent.learn(args.checkpoint_interval, args.n_tot)

        batch_explore = args.batch

        hold = 1
        while hold:
            print("wait for first samples")


            if len(os.listdir(self.dirs_locks.trajectory_dir)) > 10:
                hold = 0

            time.sleep(5)


          #  if len(os.listdir(self.dirs_locks.trajectory_dir)) >= (int(500. / args.player_replay_size * batch_explore) + 1):
           #     hold = 0

            #time.sleep(5)

        logger.info("Begin Behavioral Distributional learning experiment")
        logger.info("Game: %s " % args.game)

        for n, train_results in (enumerate(learn)):

            print("print_learn_experiment")

            n = n * args.checkpoint_interval

            avg_train_loss_v_beta = np.mean(train_results['loss_q'])

            # log to tensorboard
            if args.tensorboard:
                self.writer.add_scalar('train_loss/loss_value', float(avg_train_loss_v_beta), n + n_offset)
                self.writer.add_scalar('actions/reward', train_results['r'].mean(), n)

                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("value_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset,
                                                  'fd')
            self.print_actions_statistics(train_results['pi'], train_results['pi_tag'],
                                          train_results['q'], avg_train_loss_v_beta)

            agent.save_value_checkpoint(self.checkpoint_value, {'n': n + n_offset})

        print("End Learn")
        return agent

    def probability_to_hist(self, prob):
        hist = [int(prob[i] * 100) * [i] for i in range(len(prob))]
        hist = [x for action in hist for x in action]
        return np.asarray(hist)

    def get_player(self, agent):
        return

    def multiplay(self):
        agent = self.choose_agent()(self.exp_name, self.problem, player=True, choose=args.exploration_only, checkpoint_value=self.checkpoint_value,
                                    checkpoint_beta=self.checkpoint_beta)
        multiplayer = agent.multiplay()

        for n, train_results in (enumerate(multiplayer)):

            with np.printoptions(precision=3, suppress=True):
                res = "------------------------------------------------------------\n"
                res += "best player - actor{} with reward {} best_observed {} -- frame {} n_offset {}" \
                       "\npolicy {}\ncurrent beta {}".format(
                        train_results['actor'], train_results['reward'], train_results['best_observed'],
                        train_results['frame'], train_results['n_offset'],
                        train_results['policy'],
                        train_results['beta'])
                res += "\n------------------------------------------------------------\n"

                logger.info(res)

    def clean(self):

        agent = self.choose_agent()(self.exp_name, self.problem, player=True, checkpoint_value=self.checkpoint_value,
                                    checkpoint_beta=self.checkpoint_beta, choose=True)
        agent.clean()

    def print_actions_statistics(self, pi_player, pi_tag_player, q_player, loss_q):
        # print action meanings
        logger.info("Actions statistics: |\t loss_q = %f |" % (loss_q))

        pi_tag =   "|\tpolicy pi tag\t"
        pi =       "|\tpolicy pi    \t"
        q =        "|\tvalue q      \t"
        for i in range(consts.action_space):
            pi_tag += "|%.2f\t" % pi_tag_player[i]
            pi += "|%.2f\t" % pi_player[i]
        for i in range(len(q_player)):
            q += "|%.2f\t" % q_player[i]
        pi += "|"
        pi_tag += "|"
        q += "|"

        logger.info(pi)
        logger.info(pi_tag)
        logger.info(q)

    def evaluate(self):
        time.sleep(15)

        agent = self.choose_agent()(self.replay_dir, self.problem, player=True, checkpoint_value=self.checkpoint_value,
                                    checkpoint_beta=self.checkpoint_beta)

        player = agent.evaluate()

        for n, train_results in (enumerate(player)):

            frame = train_results['n']
            res = ("print_evaluation_experiment - |n:{}\t|value frame:{}\t|best_observed:{}\t|".format(
                n, frame, train_results['best_observed'][-1]))

            logger.info(res)

            # log to tensorboard
            if args.tensorboard:

                self.writer.add_scalar('evaluation/best_observed', train_results['best_observed'][-1], n)
                self.writer.add_scalar('evaluation/score', train_results['score'][-1], n)
                self.writer.add_scalar('evaluation/value', train_results['q'][-1], n)
                self.writer.add_scalar('evaluation/r', train_results['r'][-1], n)


                for i in range(len(train_results['pi'])):
                    self.writer.add_scalar('evaluation/pi_' + str(i), train_results['pi'][i], n)
                    self.writer.add_scalar('evaluation/beta_' + str(i), train_results['beta'][i], n)

                # G = 0
                # rewards = []
                # first = True
                # for r in reversed(train_results['r']):
                #     # the value of the terminal state is 0 by definition
                #     # we should ignore the first state we encounter
                #     # and ignore the last G, which is meaningless since it doesn't correspond to any move
                #     rewards.append(G)
                #     G = r + args.gamma * G
                # rewards.reverse()  # we want it to be in order of state visited
                #
                # for i in range(len(train_results['best_observed'])):
                #     self.writer.add_scalar(game_str + '/best_observed', train_results['best_observed'][i], i)
                #     self.writer.add_scalar(game_str + '/r', train_results['r'][i], i)
                #     self.writer.add_scalar(game_str + '/g', rewards[i], i)



                # self.writer.add_histogram("policy/pi", self.probability_to_hist(train_results['pi']), frame, 'doane')
                # self.writer.add_histogram("policy/beta", self.probability_to_hist(train_results['beta']), frame, 'doane')
                # self.writer.add_histogram("policy/value", self.probability_to_hist(train_results['q']), frame, 'doane')

                if hasattr(agent, "beta_net"):
                    self.writer.add_histogram("evaluation/beta_net", agent.beta_net.clone().cpu().data.numpy(), frame, 'fd')
                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("evaluation/value_net/%s" % name, param.clone().cpu().data.numpy(), frame, 'fd')

            pi = "|\tpolicy pi\t"
            beta = "|\tpolicy beta\t"
            q = "|\tvalue q\t    "
            for i in range(consts.action_space):
                pi += "|%.2f\t" % train_results['pi'][i]
                beta += "|%.2f\t" % train_results['beta'][i]

            for i in range(len(train_results['q'])):
                q += "|%.2f\t" % train_results['q'][i]

            pi += "|"
            beta += "|"
            q += "|"
            logger.info(pi)
            logger.info(beta)
            logger.info(q)

        print("End evaluation")

