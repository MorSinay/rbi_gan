try: import cocoex
except: pass

try: import cma
except: pass

import scipy.optimize
import pandas as pd
import numpy as np
import pwd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

username = pwd.getpwuid(os.geteuid()).pw_name

if username == 'morsi':
    base_dir = os.path.join('/Users', username, 'Desktop', 'baseline')
else:
    base_dir = os.path.join('/data/', username, 'gan_rl', 'baseline')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def reset_data_dic():
    return {
        'index': [], 'hit': [], 'id': [], 'dimension': [], 'best_observed': [], 'initial_solution': [],
        'upper_bound': [], 'lower_bound': [], 'number_of_evaluations': []
        }

def random_search(f, lbounds, ubounds, evals):
    """Won't work (well or at all) for `evals` much larger than 1e5"""
    [f(x) for x in np.asarray(lbounds) + (np.asarray(ubounds) - lbounds)
                               * np.random.rand(int(evals), len(ubounds))]


def run_baseline(budget=100000):
    suite_name = "bbob"
    suite_filter_options = ("dimensions: 2,3,5,10,20,40 ")  # "year:2019 " +  "instance_indices: 1-5 ")
    suite = cocoex.Suite(suite_name, "", suite_filter_options)

    optimization_function = [scipy.optimize.fmin_slsqp, scipy.optimize.fmin, scipy.optimize.fmin_cobyla,
                             cocoex.solvers.random_search, cma.fmin2]

    for fmin in optimization_function:
        data = reset_data_dic()
        suite = cocoex.Suite(suite_name, "", suite_filter_options)
        fmin_name = fmin.__name__

        for i, problem in tqdm(enumerate(suite)):

            if fmin_name is 'fmin_slsqp':
                output = fmin(problem, problem.initial_solution, iter=budget,  # very approximate way to respect budget
                              full_output=True, iprint=-1)

            elif fmin_name is 'fmin':
                output = fmin(problem, problem.initial_solution,
                              maxfun=budget * problem.dimension, disp=False, full_output=True)

            elif fmin_name is 'random_search':
                fmin(problem, problem.dimension * [-5], problem.dimension * [5], budget * problem.dimension)

            elif fmin_name is 'fmin2':
                xopt, es = fmin(problem, problem.initial_solution, 2,
                                {'maxfevals':budget * problem.dimension, 'verbose':-9}, restarts=9)

            elif fmin_name is 'fmin_cobyla':
                fmin(problem, problem.initial_solution, cons=lambda x: problem.constraint(x), maxfun=budget * problem.dimension,
                     disp=0, rhoend=1e-9)


            else:
                raise NotImplementedError

            data['index'].append(problem.index)
            data['hit'].append(problem.final_target_hit)
            data['id'].append(problem.id)
            data['dimension'].append(problem.dimension)
            data['best_observed'].append(problem.best_observed_fvalue1)
            data['initial_solution'].append(problem.initial_solution)
            data['upper_bound'].append(problem.upper_bounds)
            data['lower_bound'].append(problem.lower_bounds)
            data['number_of_evaluations'].append(problem.evaluations)

        df = pd.DataFrame(data)
        fmin_file = os.path.join(base_dir, fmin_name+'.csv')
        df.to_csv(fmin_file)


def merge_baseline(optimizers=['fmin', 'fmin_slsqp', 'random_search', 'fmin2', 'fmin_cobyla']):

    data_fmin = pd.read_csv(os.path.join(base_dir, optimizers[0]+".csv"))
    data = data_fmin[['index', 'id', 'dimension', 'initial_solution', 'upper_bound', 'lower_bound']]

    hit_col = []
    for op in optimizers:
        data_fmin = pd.read_csv(os.path.join(base_dir, op+".csv"))
        data_fmin = data_fmin.rename(columns={"hit": op+"_hit", "best_observed": op+"_best_observed",
                                               "number_of_evaluations": op+"_number_of_evaluations"})
        hit_col.append(op+"_hit")

        data_fmin = data_fmin[['id', op+'_hit', op+'_best_observed', op+'_number_of_evaluations']]
        data = data.merge(data_fmin, on='id')

    data['baseline_hit'] = data[hit_col].max(axis=1)
    fmin_file = os.path.join(base_dir, 'baselines.csv')
    data.to_csv(fmin_file)


def merge_bbo(dim=['2', '3', '5', '10', '20', '40'], optimizers=['fmin', 'fmin_slsqp', 'random_search', 'fmin2', 'fmin_cobyla']):

    df = pd.read_csv(os.path.join(base_dir, "bbo_"+dim[0]+".csv"))

    for d in dim[1:]:
        df_d = pd.read_csv(os.path.join(base_dir, "bbo_"+d+".csv"))
        df = df.append(df_d, ignore_index=True)


    baseline_df = pd.read_csv(os.path.join(base_dir,'baselines.csv'))
    df = df.rename(columns={"hit": "bbo_hit", "best_observed": "bbo_best_observed",
                            "number_of_evaluations": "bbo_number_of_evaluations"})
    df = df[['id', 'bbo_hit', 'bbo_best_observed', 'bbo_number_of_evaluations']]
    baseline_df = baseline_df.merge(df, on='id')
    best_observed = [op+'_best_observed' for op in optimizers]
    best_observed.append('bbo_best_observed')
    baseline_df['dist_from_min'] = np.abs(baseline_df[best_observed].min(axis=1) - baseline_df['bbo_best_observed'])
    baseline_df['bbo__dist_hit'] = baseline_df['dist_from_min'] < 2

    file = os.path.join(base_dir, 'compare.csv')
    baseline_df.to_csv(file)

def plot_res(optimizers=['fmin', 'fmin_slsqp', 'random_search', 'fmin2', 'fmin_cobyla', 'bbo', 'bbo__dist']):
    dimension = [2, 3, 5, 10, 20, 40]
    color = ['b', 'g', 'r', 'y', 'c', 'm', 'k', '0.75']
    df = pd.read_csv(os.path.join(base_dir, "compare.csv"))

    res = [[] for _ in range (len(optimizers))]
    for _, dim in enumerate(dimension):
        df_temp = df[df['dimension'] == dim]
        dim_size = len(df_temp)
        for n, op in enumerate(optimizers):
            res[n].append(df_temp[op+'_hit'].count()/dim_size)

    x = [5*i for i in range(len(dimension))]
    ax = plt.subplot(111)
    w = 0.3

    for i in range(len(dimension)):
        ax.bar(x + i*w, res[i], width=w, color=color[i], align='center', label=optimizers[i])

    ax.set_xticks([i + int(len(dimension)/2) for i in x])
    ax.set_xticklabels(dimension)
    ax.autoscale(tight=True)

    plt.show()



if __name__ == '__main__':
    merge_baseline(optimizers=['fmin', 'fmin_slsqp', 'random_search', 'fmin_cobyla'])
    merge_bbo(dim=['2', '3', '5', '10', '20', '40'],
              optimizers=['fmin', 'fmin_slsqp', 'random_search', 'fmin_cobyla'])

    plot_res(optimizers=['fmin', 'fmin_slsqp', 'random_search', 'fmin_cobyla'])