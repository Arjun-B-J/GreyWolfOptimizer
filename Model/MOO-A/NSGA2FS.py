# NSGA2 Feature Selector

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

import numpy as np

class NSGA2FS:
  def __init__(self, dataset, ANN, pop_n=8, n_gen=100):
    self.dataset = dataset
    self.ANN = ANN
    self.pop_n = pop_n
    self.n_gen = n_gen

  def optimize(self):
    algorithm = NSGA2(pop_size= self.pop_n,
                  sampling=get_sampling("bin_random"),
                  crossover=get_crossover("bin_two_point"),
                  mutation=get_mutation("bin_bitflip"),
                  eliminate_duplicates=True)
    
    problem = FeatureSelection(self.dataset, self.ANN)

    res = minimize(problem, algorithm, ('n_gen', self.n_gen), seed=42, verbose=True)

    return res

class FeatureSelection(Problem):

  def __init__(self, dataset, ANN):

    self.dataset = dataset
    self.ANN = ANN

    n_var = len(dataset.columns) - 1

    super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl = np.zeros(n_var) , xu = np.ones(n_var))

  def _evaluate(self, x, out, *args, **kwargs):

    res = []
    for feature_subset in x:

      no_of_features = np.count_nonzero(feature_subset)

      if no_of_features != 0:
        ann = self.ANN(feature_subset, self.dataset)
        ann.train()
        err = ann.test_error()

        res.append((no_of_features, err))
        
      else:
        res.append((len(feature_subset), float('inf')))

    out["F"] = np.array(res)