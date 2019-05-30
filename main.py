from config import consts, args
from logger import logger
from experiment import Experiment
import torch
import numpy as np
import time


def main():

    torch.set_num_threads(1000)
    print("Torch %d" % torch.get_num_threads())
    # print args of current run
    logger.info("Welcome to Gan simulation")
    logger.info(' ' * 26 + 'Simulation Hyperparameters')
    for k, v in vars(args).items():
        logger.info(' ' * 26 + k + ': ' + str(v))

    with Experiment(logger.filename) as exp:

        if args.learn:
            logger.info("GanRL Learning Session, it might take a while")
            exp.learn()

        elif args.play:
            logger.info("Start a player Session")
            exp.play()

        elif args.evaluate:
            logger.info("Evaluate performance")
            exp.evaluate()

        elif args.evaluate_random_policy:
            logger.info("Evaluate random policy performance")
            pi = np.ones(consts.action_space, dtype=np.float32) / consts.action_space
            #pi = np.ones(consts.action_space, dtype=np.float32)*0.01
            #pi[9] += 0.65
            #pi[3] += 0.25
            exp.evaluate(pi=pi)

        elif args.multiplay:
            logger.info("Start a multiplay Session")
            exp.multiplay()

        elif args.multiplay_random:
            logger.info("Start a random multiplay Session")
            exp.multiplay_random()

        elif args.clean:
            logger.info("Clean old trajectories")
            exp.clean()

        else:
            raise NotImplementedError

    logger.info("End of simulation")


if __name__ == '__main__':
    main()

