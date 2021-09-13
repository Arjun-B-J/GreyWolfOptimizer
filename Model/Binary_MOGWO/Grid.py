import numpy as np

class EmptyGrid:
    def __init__(self):
        self.lower = []
        self.upper = []
    
def createHypercubes(costs, ngrid, alpha):
    nobj = costs.shape[0]
    G = []
    for i in range(nobj):
        grid = EmptyGrid()
        G.append(grid) 

    for j in range(nobj):
        min_cj=min(costs[j])
        max_cj=max(costs[j])
        dcj=alpha*(max_cj-min_cj)
        min_cj=min_cj-dcj
        max_cj=max_cj+dcj
        gx=np.linspace(min_cj, max_cj, ngrid-1)
        G[j].lower = np.insert(gx, 0, float('-inf'))
        G[j].upper = np.append(gx, float('inf'))
        
    return G


#GetGridIndex
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    if ind<0 or ind >= array_shape[0]*array_shape[1]:
        return -1
    return ind

def find_first_bigger(a, U, U_size):
    for i in range(U_size):
        if(a<U[i]):
            return i

def getGridIndex(particle, G):
    #num_of_obj=len(particle.cost)
    c=particle.cost

    nobj=len(c)  # nobj = 2
    ngrid=len(G[0].upper)

    
    ones_array = np.ones(nobj)*ngrid
    SubIndex=np.zeros(nobj)
    
    for j in range(nobj):
        U=G[j].upper
        U_size = len(U)
        i=find_first_bigger(c[j], U, U_size)
        SubIndex[j]=i
    
    Index=sub2ind(ones_array, SubIndex[0], SubIndex[1])
    
    SubIndex = np.array(SubIndex)

    #print(f'particel_cost: {c} Index: {Index}')
    
    return Index, SubIndex