import argparse
import time
import numpy as np
import socket
import os
import pwd
import fcntl

parser = argparse.ArgumentParser(description='gan_rl')
username = pwd.getpwuid(os.geteuid()).pw_name
server = socket.gethostname()

if "gpu" in server:
    base_dir = os.path.join('/home/mlspeech/', username, 'data/gan_rl', server)
elif "root" == username:
    base_dir = r'/workspace/data/gan_rl/'
else:
    base_dir = os.path.join('/data/', username, 'gan_rl', server)

def boolean_feature(feature, default, help):

    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})


# General Arguments
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--batch', type=int, default=64, help='Mini-Batch Size')


# strings
parser.add_argument('--game', type=str, default='bbo', help='bbo | net')
parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')
parser.add_argument('--algorithm', type=str, default='reinforce', help='[reinforce]')

# # booleans
boolean_feature("load-last-model", False, 'Load the last saved model')
boolean_feature("learn", False, 'Learn from the observations')
boolean_feature("save-beta", False, 'Save beta')
boolean_feature("exploration-only", False, 'Exploration Agent')
boolean_feature("postprocess", False, 'Postprocess evaluation results')
boolean_feature("multiplay", False, 'Send samples to memory from multiple parallel players')
boolean_feature("evaluate", False, 'evaluate player')
boolean_feature("clean", False, 'Clean old trajectories')
boolean_feature("tensorboard", False, "Log results to tensorboard")
parser.add_argument('--n-steps', type=int, default=1, metavar='STEPS', help='Number of steps for multi-step learning')
parser.add_argument('--budget', type=int, default=10000, help='Number of steps')
# parameters
parser.add_argument('--resume', type=int, default=-1, help='Resume experiment number, set -1 for last experiment')

# #exploration parameters
parser.add_argument('--epsilon', type=float, default=0.1, metavar='ε', help='exploration parameter before behavioral period')
#
# #dataloader
parser.add_argument('--cpu-workers', type=int, default=24, help='How many CPUs will be used for the data loading')
parser.add_argument('--cuda-default', type=int, default=0, help='Default GPU')
#
# #train parameters
parser.add_argument('--n-tot', type=int, default=1500000, metavar='STEPS', help='Total number of training steps')
parser.add_argument('--checkpoint-interval', type=int, default=1000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--player-replay-size', type=int, default=2000, help='Player\'s replay memory size')
parser.add_argument('--update-memory-interval', type=int, default=10, metavar='STEPS', help='Number of steps between memory updates')
parser.add_argument('--load-memory-interval', type=int, default=10, metavar='STEPS', help='Number of steps between memory loads')
parser.add_argument('--replay-updates-interval', type=int, default=50, metavar='STEPS', help='Number of training iterations between q-target updates')
parser.add_argument('--replay-memory-size', type=int, default=20000, help='Total replay exploit memory size')
parser.add_argument('--gamma', type=float, default=0.97, metavar='LR', help='gamma (default: 0.97)')
parser.add_argument('--delta', type=float, default=0.1, metavar='delta', help='Total variation constraint')
parser.add_argument('--save-to-mem', type=int, default=200, metavar='stm', help='Save to memory')
parser.add_argument('--n-rand', type=int, default=2000, metavar='rnand', help='random play')
parser.add_argument('--rl-metric', type=str, default='td', metavar='rl', help='td|mc')
#
# #actors parameters
parser.add_argument('--n-players', type=int, default=30, help='Number of parallel players for current actor')
parser.add_argument('--actor-index', type=int, default=0, help='Index of current actor')
parser.add_argument('--beta-lr', type=float, default=0.0001, metavar='LR', help='beta learning rate')
parser.add_argument('--value-lr', type=float, default=0.0001, metavar='LR', help='value learning rate')


parser.add_argument('--metric', type=str, default='SMOOTH', metavar='N', help='L1|MSE|SMOOTH')
parser.add_argument('--architecture', type=str, default='SAME', metavar='N', help='SAME|BIGGER')
parser.add_argument('--exploration', type=str, default='GRAD', metavar='N', help='GRAD|UNIFORM')

# distributional learner

args = parser.parse_args()


# consts
class Consts(object):

    server = socket.gethostname()
    start_time = time.time()
    exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    action_space = 10
    nop = 0

    #TODO Mor: what is this?
    mem_threshold = int(2e9)

    rec_type = np.dtype([('fr', np.int64),
                         ('r', np.float32), ('best_observed', np.float32), ('t', np.float32), ('pi', np.float32, action_space),
                         ('pi_explore', np.float32, action_space), ('traj', np.int64), ('ep', np.int64)])

    outdir = os.path.join(base_dir, 'results')
    indir = os.path.join('/dev/shm/', username, 'gan_rl')
    logdir = os.path.join(base_dir, 'logs')

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

consts = Consts()



class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DirsAndLocksSingleton(metaclass=Singleton):
    def __init__(self, exp_name):
        self.outdir = consts.outdir
        self.exp_name = exp_name
        self.root = os.path.join(self.outdir, self.exp_name)

        self.indir = consts.indir
        self.explore_dir = os.path.join(self.indir, "explore")
        self.list_dir = os.path.join(self.indir, "list")

        self.trajectory_dir = os.path.join(self.explore_dir, "trajectory")
        self.list_old_path = os.path.join(self.list_dir, "old_explore")
        #self.snapshot_path = os.path.join(self.root, "snapshot")

        self.readlock = os.path.join(self.list_dir, "readlock_explore.npy")
        self.writelock = os.path.join(self.list_dir, "writelock.npy")
        self.episodelock = os.path.join(self.list_dir, "episodelock.npy")

        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
        self.results_dir = os.path.join(self.root, 'results')
        self.code_dir = os.path.join(self.root, 'code')
        self.analysis_dir = os.path.join(self.root, 'analysis')
        self.checkpoint_value = os.path.join(self.checkpoints_dir, 'checkpoint_value')
        self.checkpoint_beta = os.path.join(self.checkpoints_dir, 'checkpoint_beta')
        self.replay_dir = os.path.join(self.indir, self.exp_name)
        self.scores_dir = os.path.join(self.root, 'scores')

        if not os.path.exists(self.trajectory_dir):
            os.makedirs(self.trajectory_dir)
        if not os.path.exists(self.list_old_path):
            os.makedirs(self.list_old_path)
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(self.code_dir):
            os.makedirs(self.code_dir)
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
        if not os.path.exists(self.scores_dir):
            os.makedirs(self.scores_dir)


def lock_file(file):

    fo = open(file, "r+b")
    while True:
        try:
            fcntl.lockf(fo, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except IOError:
            pass

    return fo


def release_file(fo):
    fcntl.lockf(fo, fcntl.LOCK_UN)
    fo.close()

