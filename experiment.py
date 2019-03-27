#import time
import os
import sys
import numpy as np
#import pandas as pd

#from tensorboardX import SummaryWriter

from tqdm import tqdm
import time

from config import consts, args, DirsAndLocksSingleton
from gan_rl_agent import GANAgent

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

        return GANAgent

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

            if len(os.listdir(self.dirs_locks.trajectory_dir)) >= (int(500. / args.player_replay_size * batch_explore) + 1):
                hold = 0

            time.sleep(5)

        logger.info("Begin Behavioral Distributional learning experiment")
        logger.info("Game: %s " % args.game)

        for n, train_results in enumerate(learn):

            n = n * args.checkpoint_interval

            avg_train_loss_beta = np.mean(train_results['loss_beta'])
            avg_train_loss_v_beta = np.mean(train_results['loss_q'])
            avg_train_loss_std = np.mean(train_results['loss_std'])

            # log to tensorboard
            if args.tensorboard:
                self.writer.add_scalar('train_loss/loss_beta', float(avg_train_loss_beta), n + n_offset)
                self.writer.add_scalar('train_loss/loss_value', float(avg_train_loss_v_beta), n + n_offset)
                self.writer.add_scalar('train_loss/loss_std', float(avg_train_loss_std), n + n_offset)

                self.writer.add_scalar('states/state', train_results['s'], n)
                self.writer.add_scalar('actions/reward', train_results['r'], n)
                self.writer.add_histogram("actions/a_player", train_results['a_player'], n + n_offset, 'doane')

                if hasattr(agent, "beta_net"):
                    for name, param in agent.beta_net.named_parameters():
                        self.writer.add_histogram("beta_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset,
                                                  'fd')
                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("value_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset,
                                                  'fd')

            self.print_actions_statistics(train_results['a_player'], avg_train_loss_beta, avg_train_loss_v_beta)
            agent.save_checkpoint(self.checkpoint, {'n': n + n_offset})

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
        return
        # agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint)
        # multiplayer = agent.multiplay()
        #
        # for _ in multiplayer:
        #
        #     player = self.get_player(agent)
        #     if player:
        #         agent.set_player(player['player'], behavioral_avg_score=player['high'],
        #                          behavioral_avg_frame=player['frames'])

    def play(self, params=None):

        uuid = "%012d" % np.random.randint(1e12)
        agent = self.choose_agent()(self.exp_name, player=True, checkpoint=self.checkpoint)
        aux = agent.resume(self.checkpoint)

        n = aux['n']
        results = {"n" : n, "score": [], "frame": []}

        player = agent.play(args.play_episodes_interval)

        for i, step in tqdm(enumerate(player)):
            results["frame"].append(step['frames'])
            print("frames: %d" % (step['frames']))

        if self.log_scores:
            logger.info("Save NPY file: eval_%d_%s.npy" % (n, uuid))
            filename = os.path.join(self.scores_dir, "eval_%d_%s" % (n, uuid))
            np.save(filename, results)

    def clean(self):

        agent = self.choose_agent()(self.exp_name, player=True, checkpoint=self.checkpoint, choose=True)
        agent.clean()

    def evaluate(self):
        return
        # uuid = "%012d" % np.random.randint(1e12)
        # agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint, choose=True)
        #
        # best_score = -np.inf
        #
        # tensorboard_path = os.path.join(self.results_dir, uuid)
        # os.makedirs(tensorboard_path)
        #
        # if args.tensorboard:
        #     self.writer = SummaryWriter(log_dir=tensorboard_path, comment="%s_%s" % (args.identifier, uuid))
        #
        # results_filename = os.path.join(agent.best_player_dir, "%s.npy" % uuid)
        # scores_dir = os.path.join(self.scores_dir, uuid)
        # os.makedirs(scores_dir)
        #
        # kk = 0
        #
        # if args.algorithm in ["rbi", "rbi_rnn"]:
        #     results = {'n': 0,
        #                'statistics': {
        #                    'reroute': {'player': 'reroutetv', 'cmin': args.cmin, 'cmax': args.cmax, 'delta': args.delta, 'score': 0, 'high': 0, 'frames':1},
        #                    'behavioral': {'player': 'behavioral', 'cmin': None, 'cmax': None, 'delta': 0, 'score': 0, 'high': 0, 'frames':1}
        #                }}
        # elif args.algorithm in ["ape", "r2d2"]:
        #     results = {'n': 0,
        #                'statistics': {
        #                    'reroute': {'player': 'reroutetv', 'cmin': args.cmin, 'cmax': args.cmax, 'delta': args.delta, 'score': 0, 'high': 0, 'frames':1},
        #                    'behavioral': {'player': 'behavioral', 'cmin': None, 'cmax': None, 'delta': 0, 'score': 0, 'high': 0, 'frames':1}
        #                }}
        # elif args.algorithm == "ppo":
        #     results = {'n': 0,
        #                'statistics': {
        #                    'reroute': {'player': 'reroutetv', 'cmin': args.cmin, 'cmax': args.cmax, 'delta': args.delta, 'score': 0, 'high': 0,
        #                                           'frames': 1},
        #                    'behavioral': {'player': 'behavioral', 'cmin': None, 'cmax': None, 'delta': 0, 'score': 0, 'high': 0,
        #                                   'frames': 1}
        #                }}
        # else:
        #     raise NotImplementedError
        #
        # time.sleep(args.wait)
        #
        # print("Here")
        #
        # while True:
        #
        #     # load model
        #     try:
        #         aux = agent.resume(agent.snapshot_path)
        #     except:  # when reading and writing collide
        #         time.sleep(2)
        #         aux = agent.resume(agent.snapshot_path)
        #
        #     n = aux['n']
        #
        #     if n < args.random_initialization:
        #         time.sleep(5)
        #         continue
        #
        #     results['n'] = n
        #     results['time'] = time.time() - consts.start_time
        #
        #     for player_name in results['statistics']:
        #
        #         scores = []
        #         frames = []
        #         mc = np.array([])
        #         q = np.array([])
        #
        #         player_params = results['statistics'][player_name]
        #         agent.set_player(player_params['player'], cmin=player_params['cmin'], cmax=player_params['cmax'],
        #                          delta=player_params['delta'])
        #
        #         player = agent.play(args.play_episodes_interval, save=False, load=False)
        #
        #         stats = {"score": [], "frame": [], "time": [], "n": []}
        #
        #         tic = time.time()
        #
        #         for i, step in enumerate(player):
        #
        #             print("stats | player: %s | episode: %d | time: %g" % (player_name, i, time.time() - tic))
        #             tic = time.time()
        #             scores.append(step['score'])
        #             frames.append(step['frames'])
        #             mc = np.concatenate((mc, step['mc']))
        #             q = np.concatenate((q, step['q']))
        #
        #             # add stats results
        #             stats["score"].append(step['score'])
        #             stats["frame"].append(step['frames'])
        #             stats["n"].append(step['n'])
        #             stats["time"].append(time.time() - consts.start_time)
        #
        #         # random selection
        #         set_size = 200
        #         indexes = np.random.choice(len(mc), set_size)
        #         q = np.copy(q[indexes])
        #         mc = np.copy(mc[indexes])
        #
        #         score = np.array(scores)
        #         frames = np.array(frames)
        #
        #         player_params['score'] = score.mean()
        #         # player_params['high'] = score.max()
        #         player_params['frames'] = np.percentile(frames, 90)
        #         player_params['high'] = np.percentile(scores, 90)
        #
        #         # save best player checkpoint
        #         if player_name != "behavioral" and score.mean() > best_score:
        #             best_score = score.mean()
        #             agent.save_checkpoint(self.checkpoint_best, {'n': n, 'score': score})
        #
        #         if args.tensorboard:
        #
        #             self.writer.add_scalar('score/%s' % player_name, float(score.mean()), n)
        #             self.writer.add_scalar('high/%s' % player_name, float(score.max()), n)
        #             self.writer.add_scalar('low/%s' % player_name, float(score.min()), n)
        #             self.writer.add_scalar('std/%s' % player_name, float(score.std()), n)
        #             self.writer.add_scalar('frames/%s' % player_name, float(frames.mean()), n)
        #
        #             try:
        #                 self.writer.add_histogram("mc/%s" % player_name, mc, n, 'fd')
        #                 self.writer.add_histogram("q/%s" % player_name, q, n, 'fd')
        #             except:
        #                 pass
        #
        #         np.save(results_filename, results)
        #
        #         if self.log_scores:
        #             logger.info("Save NPY file: %d_%s_%d_%s.npy" % (n, uuid, kk, player_name))
        #             stat_filename = os.path.join(scores_dir, "%d_%s_%d_%s" % (n, uuid, kk, player_name))
        #             np.save(stat_filename, stats)
        #
        #         kk += 1
        #
        #     if agent.n_offset >= args.n_tot:
        #         break
        #
        # print("End of evaluation")

    def postprocess(self):
        return
        # run_dir = os.path.join(self.root, "scores")
        # save_dir = os.path.join(self.root, "postprocess")
        #
        # if not os.path.isdir(save_dir):
        #     os.mkdir(save_dir)
        # elif os.path.isfile(os.path.join(save_dir, "df_reroute")) and os.path.isfile(
        #         os.path.join(save_dir, "df_behavioral")):
        #     return
        #
        # results_reroute = {'score': [], 'frame': [], 'n': [], 'time': []}
        # results_behavioral = {'score': [], 'frame': [], 'n': [], 'time': []}
        #
        # for d in os.listdir(run_dir):
        #     print(d)
        #     for f in os.listdir(os.path.join(run_dir, d)):
        #
        #         if "behavioral" in f:
        #             for key in results_behavioral:
        #                 item = np.load(os.path.join(run_dir, d, f)).item()
        #                 results_behavioral[key] += item[key]
        #         else:
        #             for key in results_reroute:
        #                 item = np.load(os.path.join(run_dir, d, f)).item()
        #                 results_reroute[key] += item[key]
        #
        # df_reroute = pd.DataFrame(results_reroute)
        # df_behavioral = pd.DataFrame(results_behavioral)
        #
        # df_reroute.to_pickle(os.path.join(save_dir, "df_reroute"))
        # df_behavioral.to_pickle(os.path.join(save_dir, "df_behavioral"))

    def print_actions_statistics(self, a_player, loss_q, loss_beta):

        # print action meanings
        logger.info("Actions statistics: \tloss_beta = %f |\t loss_q = %f |" % (loss_beta, loss_q))

        n_actions = len(a_player)
        applied_player_actions = (np.bincount(np.concatenate((a_player, np.arange(consts.action_space)))) - 1) / n_actions

        line = ''
        line += "|\tPlayer actions\t"
        for a in applied_player_actions:
            line += "|%.2f\t    " % (a*100)
        line += "|"
        logger.info(line)

    def demonstrate(self, params=None):
        return
        # agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint)
        #
        # # load model
        # try:
        #     if params is not None:
        #         aux = agent.resume(params)
        #     elif self.load_last:
        #         aux = agent.resume(self.checkpoint)
        #     elif self.load_best:
        #         aux = agent.resume(self.checkpoint_best)
        #     else:
        #         raise NotImplementedError
        # except:  # when reading and writing collide
        #     time.sleep(2)
        #     if params is not None:
        #         aux = agent.resume(params)
        #     elif self.load_last:
        #         aux = agent.resume(self.checkpoint)
        #     elif self.load_best:
        #         aux = agent.resume(self.checkpoint_best)
        #     else:
        #         raise NotImplementedError
        #
        # player = agent.demonstrate(128)
        #
        # for i, step in enumerate(player):
        #     yield step
