from config import consts, args
from logger import logger
from experiment import Experiment
import torch
import numpy as np
import time
import cocoex


def main():

    torch.set_num_threads(1000)
    print("Torch %d" % torch.get_num_threads())
    # print args of current run
    logger.info("Welcome to Gan simulation")
    logger.info(' ' * 26 + 'Simulation Hyperparameters')
    for k, v in vars(args).items():
        logger.info(' ' * 26 + k + ': ' + str(v))

    #################################
    suite_name = "bbob"
    suite_filter_options = ("dimensions: 10 " #"year:2019 " +  "instance_indices: 1-5 "
                        )
    suite = cocoex.Suite(suite_name, "", suite_filter_options)
    #observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    for problem in suite:  # this loop will take several minutes or longer
        my_problem = problem
        break

    with Experiment(logger.filename, my_problem) as exp:

        if args.learn:
            logger.info("GanRL Learning Session, it might take a while")
            exp.learn()

        elif args.evaluate:
            logger.info("Evaluate performance")
            exp.evaluate()

        elif args.multiplay:
            logger.info("Start a multiplay Session")
            exp.multiplay()

        elif args.clean:
            logger.info("Clean old trajectories")
            exp.clean()

        else:
            raise NotImplementedError

    logger.info("End of simulation")


if __name__ == '__main__':
    main()

