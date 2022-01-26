import numpy as np
import datetime
import time
from ANN_Classifier_v2 import *

np.random.seed(42)

# sigmoid fun
def sigmoid(x):
  return 1/(1 + np.exp( (-10) * (x - 0.5) ))


class Wolf:

  def __init__(self, dim):
    pos = sigmoid(np.random.rand(dim)) >= 0.5

    # To avoid all zero condition
    while(np.count_nonzero(pos)==0):
      pos = sigmoid(np.random.rand(dim)) >= 0.5

    self.position = pos
    self.obj_score = float('inf')
    self.obj_fun_time = None

class BinaryGWO:
  
  def __init__(self, dataset, wolves_n, iter_n):
    self.dataset = dataset
    self.dim = dataset.shape[1] - 1
    self.wolves_n = wolves_n
    self.iter_n = iter_n

    self.history = {
        'alpha' : [],
        'beta' : [],
        'delta' : []
    }

    self.wolves = []
    self.alpha = None
    self.beta = None
    self.delta = None


  def initialize_wolves(self):
    self.wolves = []

    for i in range(self.wolves_n):
      w = Wolf(self.dim)
      #print(w.position)
      self.wolves.append(w)
    
    self.alpha, self.beta, self.delta = self.wolves[:3]
  

  def update_leaders(self):
    for w in self.wolves:

      if w==self.alpha or w==self.beta or w==self.delta:
        continue

      print(f'Position : {w.position}')
      a = datetime.datetime.now()

      ann_model = ANN(w.position, self.dataset)
      ann_model.train()
      w.obj_score = ann_model.test_error()
#     w.obj_score = self.obj_fun(w.position)
#     time.sleep(.2)
      b = datetime.datetime.now()
      w.obj_fun_time = b-a
        
      if w.obj_score <= self.alpha.obj_score:
          self.delta = self.beta
          self.beta = self.alpha
          self.alpha = w
        
      elif w.obj_score <= self.beta.obj_score:
          self.delta = self.beta
          self.beta = w
            
      elif w.obj_score <= self.delta.obj_score:
          self.delta = w
          
          
    self.history['alpha'].append({ 'score': self.alpha.obj_score, 'pos': self.alpha.position})
    self.history['beta'].append({ 'score': self.beta.obj_score, 'pos': self.beta.position})
    self.history['delta'].append({ 'score': self.delta.obj_score, 'pos': self.delta.position})

    print(f'alpha pos: {self.alpha.position}\tscore : {self.alpha.obj_score}')
    print(f'beta pos: {self.beta.position}\tscore : {self.beta.obj_score}')
    print(f'delta pos: {self.delta.position}\tscore : {self.delta.obj_score}')


  def print_wolf_time(self) :
    for w in self.wolves:
        print('Time => ', w.obj_fun_time)

  def update_positions(self, i):

    a = 2 * (1 - (i/self.iter_n))
        
    for w in self.wolves:

      if w==self.alpha or w==self.beta or w==self.delta:
        continue
            
      # r1 & r2 are random vectors in [0, 1]
      r1 = np.random.rand(self.dim)
      r2 = np.random.rand(self.dim)
            
      A1 = a * ((2 * r1) - 1)
      C1 = 2 * r2
          
      D_alpha = abs((C1 * self.alpha.position) - w.position)

      c_step_alpha = sigmoid(A1 * D_alpha)
      b_step_alpha = c_step_alpha >= np.random.rand(self.dim)
      X1 = ((self.alpha.position + b_step_alpha) >= 1)
      #X1 = self.alpha.position - (A1 * D_alpha)
            
            
      r1 = np.random.rand(self.dim)
      r2 = np.random.rand(self.dim)
            
      A2 = a * ((2 * r1) - 1)
      C2 = 2 * r2

      D_beta = abs((C2 * self.beta.position) - w.position) 

      c_step_beta = sigmoid(A2 * D_beta)
      b_step_beta = c_step_beta >= np.random.rand(self.dim)
      X2 = ((self.beta.position + b_step_beta) >= 1)
      #X2 = self.beta.position - (A2 * D_beta)
            
            
      r1 = np.random.rand(self.dim)
      r2 = np.random.rand(self.dim)
            
      A3 = a * ((2 * r1) - 1)
      C3 = 2 * r2

      D_delta = abs((C3 * self.delta.position) - w.position)

      c_step_delta = sigmoid(A3 * D_delta)
      b_step_delta = c_step_delta >= np.random.rand(self.dim)
      X3 = ((self.delta.position + b_step_delta) >= 1)
      #X3 = self.delta.position - (A3 * D_delta)
      
      updated_position = (sigmoid( (X1 + X2 + X3)/3 ) >= np.random.rand(self.dim))

      if np.count_nonzero(updated_position) != 0:
        w.position = updated_position

    

  def optimize(self):

    self.initialize_wolves()

    self.score_history = {
        'alpha' : [],
        'beta' : [],
        'delta' : []
    }

    for i in range(self.iter_n):
      print('\n=> Iteration : ', i)
      self.update_leaders()
      self.update_positions(i)
