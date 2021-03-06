#instructions
#Assuming Naming would be <DatasetName>_<Iteration_count>.txt
#Datasets = ["BreastCancer",
#            "BreastEW",
#            "CongressEW",
#            "Exactly",
#            "Exactly2",
#            "HeartEW",
#            "Ionosphere",
#            "KrVsKpEW",
#            "Lymphography",
#            "M-of-n",
#            "PenglungEW",
#            "Sonar",
#            "SpectEW",
#            "Tic-tac-toe",
#            "Vote",
#            "WaveformEW",
#            "Wine"
#            ]
##Change the Iteration count according to your need
#iteration_count = 100
##give proper location of where the results are stored
#res_loc = '/content/gdrive/My Drive/GWO/Results_ArchiveMod/'
#"""other assumptions are key value of dictionary would be 'explored','archiveCosts' """

from cProfile import label
import pickle
import numpy as np
import matplotlib.pyplot as plt

#Assuming Naming would be <DatasetName>_<Iteration_count>.txt
def get_filenames(datasets,i):
  temp = []
  for dataset in datasets:
    name = dataset + "_"+str(i)+".txt"
    temp.append(name)
  return temp

def exploration_progress(exp_res, maxIt, nGrey):
  unique_pos = set()
  unique_it = np.zeros(maxIt)
  for i in range(maxIt):
    for j in range(nGrey):
      pos = str(exp_res[i][j])
      if pos not in unique_pos:
        unique_pos.add(pos)
        unique_it[i] += 1
  return unique_it

def load_results(it,datasets,res_loc):
  results = []
  dataset_file_names = get_filenames(datasets,it)
  for name in dataset_file_names:
    f = open(res_loc+name,'rb')
    res = pickle.load(f)
    results.append(res)
  return results

def plot_explored(it,datasets,res_loc):
  results = load_results(it,datasets,res_loc)
  len_results = len(results)
  fig = plt.figure(figsize = (30,45))
  x = 3
  y = len_results/x
  n = 1 
  for res in results:
    exploration = exploration_progress(res['explored'],it,8)
    no_of_features = len(res['explored'][0][0])
    fig.add_subplot(y+1,x,n)
    plt.plot(exploration)
    plt.xlabel('iterations')
    plt.ylabel('unique solutions explored')
    plt.title(f'{datasets[n-1]} - [No of Features: {no_of_features}]')
    n+=1
    
def plot_archivecost(it,datasets,res_loc, scatter=False):
  results = load_results(it,datasets,res_loc)
  len_results = len(results)
  fig = plt.figure(figsize=(30,45))
  x = 3
  y = len_results/x
  n=1
  for res in results:
    archiveCost = res['archiveCosts']
    no_of_features = len(res['explored'][0][0])
    fig.add_subplot(y+1,x,n)
    if scatter:
      cost = np.array(archiveCost)
      plt.plot(cost[:,0], cost[:,1], '*')
      plt.ylabel('Classification Error')
      plt.xlabel('Number of Features')
    else:
      plt.plot(archiveCost)
      plt.ylabel('Number of Features')
      plt.xlabel('Classification Error')
    plt.title(f'{datasets[n-1]} - [No of Features: {no_of_features}]')
    n+=1

def compare_archive(it, datasets, res_loc_list, labels):
  if type(it)==list:
    it1 = it[0]
    it2 = it[1]
  else:
    it1 = it2 = it
  results_1 = load_results(it1,datasets,res_loc_list[0])
  results_2 = load_results(it2,datasets,res_loc_list[1])
  len_results = len(results_1)

  fig = plt.figure(figsize=(30,45))

  x = 3
  y = len_results/x
  n=1

  for i in range(len_results):
    res1 = results_1[i]
    res2 = results_2[i]

    archiveCost1 = res1['archiveCosts']
    archiveCost2 = res2['archiveCosts']

    archiveCost1.sort( key = lambda x: x[0] )
    archiveCost2.sort( key = lambda x: x[0] )

    no_of_features = res1['dim']
    fig.add_subplot(y+1,x,n)
    
    cost1 = np.array(archiveCost1)
    cost2 = np.array(archiveCost2)

    plt.plot(cost1[:,0], cost1[:,1], '*-', label=labels[0])
    plt.plot(cost2[:,0], cost2[:,1], '*-', label=labels[1])
    plt.ylabel('Classification Error')
    plt.xlabel('Number of Features')
    
    plt.legend()
    plt.title(f'{datasets[n-1]} - [No of Features: {no_of_features}]')
    n+=1