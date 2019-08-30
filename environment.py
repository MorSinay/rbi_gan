import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
import numpy as np

from config import consts, args

class Env(object):

    def __init__(self, problem):
        self.best_observed = None
        self.reward = None
        self.t = 0
        self.k = 0
        self.problem = problem
        self.output_size = self.problem.dimension

        self.reset()
        self.upper_bounds = self.problem.upper_bounds
        self.lower_bounds = self.problem.lower_bounds

    def reset(self):
        self.best_observed = None
        self.reward = None
        self.k = 0
        self.t = 0

    def step_policy(self, policy):

        assert(policy.size == self.output_size), "action error"
        assert ((np.clip(policy, self.lower_bounds, self.upper_bounds) - policy).sum()< 0.000001), "clipping error"
        self.reward = -self.problem(policy)
        self.best_observed = self.problem.best_observed_fvalue1
        self.k += 1
        self.t = self.problem.final_target_hit
