import time
import os
import sys
import numpy as np
from tensorboardX import SummaryWriter

from config import consts, args, DirsAndLocksSingleton
from single_agent import BBOAgent

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

        temp_name = "%s_%s_%s_bbo_%s" % (args.game, args.algorithm, args.identifier, args.problem_index)
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
        # self.indir = self.dirs_locks.indir

        # set dirs
        self.tensorboard_dir = self.dirs_locks.tensorboard_dir
        self.checkpoints_dir = self.dirs_locks.checkpoints_dir
        self.results_dir = self.dirs_locks.results_dir
        self.code_dir = self.dirs_locks.code_dir
        self.analysis_dir = self.dirs_locks.analysis_dir
        self.checkpoint = self.dirs_locks.checkpoint
        # self.replay_dir = self.dirs_locks.replay_dir

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

    def print_statistics(self, beta, beta_explore, value, reward, best_observe, beta_evaluate):
        logger.info("Actions statistics: |\t value = %f \t reward = %f|" % (value, reward))
        logger.info("Best observe      : |\t %f \t \tBeta_evaluate: = %f|" % (best_observe, beta_evaluate))

        beta_log         = "|\tbeta        \t"
        beta_explore_log = "|\tbeta_explore\t"
        for i in range(consts.action_space):
            beta_log += "|%.2f\t" % beta[i]
            beta_explore_log += "|%.2f\t" % beta_explore[i]
        beta_log += "|"
        beta_explore_log += "|"

        logger.info(beta_log)
        logger.info(beta_explore_log)

    def bbo(self):

        agent = BBOAgent(self.exp_name, self.problem, checkpoint=self.checkpoint)

        n_explore = 100
        player = agent.find_min(n_explore)

        for n, bbo_results in (enumerate(player)):
            beta = bbo_results['policies'][-1]
            beta_explore = np.average(bbo_results['explore_policies'][-1], axis=0)
            #loss_value = bbo_results['loss_value'][-1] #TODO:
            value = np.average(bbo_results['q_value'][-1])
            reward = np.average(bbo_results['rewards'][-1])
            best_observe = bbo_results['best_observed'][-1]
            beta_evaluate = self.problem(beta)

            self.print_statistics(beta, beta_explore, value, reward, best_observe, beta_evaluate)

            # log to tensorboard
            if args.tensorboard:

                #self.writer.add_scalar('evaluation/loss_value', loss_value, n)
                self.writer.add_scalar('evaluation/value', value, n)
                self.writer.add_scalar('evaluation/reward', reward, n)
                self.writer.add_scalar('evaluation/beta_evaluate', beta_evaluate, n)
                self.writer.add_scalar('evaluation/best_observe', best_observe, n)


                for i in range(len(beta)):
                    self.writer.add_scalar('evaluation/beta_' + str(i), beta[i], n)
                    self.writer.add_scalar('evaluation/beta_explore_' + str(i), beta_explore[i], n)


                if hasattr(agent, "beta_net"):
                    self.writer.add_histogram("evaluation/beta_net", agent.beta_net.clone().cpu().data.numpy(), n, 'fd')
                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("evaluation/value_net/%s" % name, param.clone().cpu().data.numpy(), n, 'fd')

        print("End evaluation")

