#from environment import train_primarily_model
#import torch
#train_primarily_model(250, 128, 5)
import sys,os

# env = Env()
# action_space = 10
# device = "cuda"
# for _ in range(10):
#     pi_rand = torch.ones(action_space, dtype=torch.float).to(device) / action_space
#     env.step(pi_rand)
from gan_rl_agent import GANAgent
from config import consts, args, DirsAndLocksSingleton
import numpy as np
from logger import logger
import torch
from experiment import Experiment
from distutils.dir_util import copy_tree

with Experiment(logger.filename) as exp:

    # if args.learn:
    #     logger.info("Enter RBI Learning Session, it might take a while")
    #     exp.learn()
    #
    # elif args.play:
    logger.info("Evaluate final performance")
    exp.play()
    logger.info("Learn")
    exp.learn()
    logger.info("Clean old trajectories")
    exp.clean()

#agent_p = GANAgent(consts.exp_name, player=True, choose=False, checkpoint=None)

# agent_l = GANAgent(consts.exp_name, player=False, choose=False, checkpoint=None)
#
# #player = agent_p.play(4)
#
# #for i, step in enumerate(player):
#  #   print("frames: %d" % (step['frames']))
#
# learner = agent_l.learn(1,1)
#
# for n, train_results in enumerate(learner):
#     avg_train_loss_beta = np.mean(train_results['loss_beta'])
#     avg_train_loss_v_beta = np.mean(train_results['loss_q'])
#     avg_train_loss_std = np.mean(train_results['loss_std'])
#     print("avg loss b: {} v: {} std: {}".format(avg_train_loss_beta,avg_train_loss_v_beta,avg_train_loss_std))
#

print("Done")
#except Exception as e:
 #   exc_type, exc_obj, exc_tb = sys.exc_info()
  #  fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
   # print(exc_type, fname, exc_tb.tb_lineno)
    #print(e)

#agent.play(16)
#agent.play(args.play_episodes_interval)
