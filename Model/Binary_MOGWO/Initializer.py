from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import sys
from google.colab import drive

def Initializer(data) :

  initial_wolves = {}


  # Mutual Info

  data1 = data
  X = data1.drop("target", axis = 1)
  y = data1["target"]

  mutual_mat = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)

  sum(mutual_mat > np.mean(mutual_mat))
  mutual_mean = np.mean(mutual_mat)
  max(mutual_mat)
  min(mutual_mat)
  point_1 = (min(mutual_mat) + mutual_mean)/2
  point_2 = (max(mutual_mat) + mutual_mean)/2

  init_1 = [0 for i in range(len(mutual_mat))] 
  for i in range(len(mutual_mat)) :
    if mutual_mat[i] >= mutual_mean :
      init_1[i] = 1

  init_2 = [0 for i in range(len(mutual_mat))] 
  for i in range(len(mutual_mat)) :
    if mutual_mat[i] >= point_1 :
      init_2[i] = 1 

  init_3 = [0 for i in range(len(mutual_mat))] 
  for i in range(len(mutual_mat)) :
    if mutual_mat[i] >= point_2 :
      init_3[i] = 1


  # Correlation 

  correlation_mat = data.corr()
  correlation_required_array = abs(correlation_mat['target'][:-1])


  correlation_mean = np.mean(correlation_required_array)
  point_3 = (min(correlation_required_array) + correlation_mean)/2
  point_4 = (max(correlation_required_array) + correlation_mean)/2

  init_4 = [0 for i in range(len(correlation_required_array))] 
  for i in range(len(correlation_required_array)) :
    if correlation_required_array[i] >= correlation_mean :
      init_4[i] = 1

  init_5 = [0 for i in range(len(correlation_required_array))] 
  for i in range(len(correlation_required_array)) :
    if correlation_required_array[i] >= point_3 :
      init_5[i] = 1

  init_6 = [0 for i in range(len(correlation_required_array))] 
  for i in range(len(correlation_required_array)) :
    if correlation_required_array[i] >= point_4 :
      init_6[i] = 1



# Random Allotment

  init_7 = [0 for i in range(len(correlation_required_array))] 
  while(np.count_nonzero(init_7)==0):
    init_7 = (np.random.uniform(size = len(mutual_mat)) >= 0.5) * 1


  init_8 = [0 for i in range(len(correlation_required_array))] 
  while(np.count_nonzero(init_8)==0):
    init_8 = (np.random.uniform(size = len(mutual_mat)) >= 0.5) * 1 





  initial_wolves[0] = init_1
  initial_wolves[1] = init_2
  initial_wolves[2] = init_3  
  initial_wolves[3] = init_4
  initial_wolves[4] = init_5
  initial_wolves[5] = init_6  
  initial_wolves[6] = init_7
  initial_wolves[7] = init_8  


  return initial_wolves

