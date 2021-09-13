from BMOGWO import *
import numpy as np

def fun(b):
    #print(sum(b))
    return np.array([ 1/sum(b), sum(b) ])

a = BMOGWO(dim=10, fobj = fun , greyWolvesNum=10, archiveSize=5, maxIt= 100, nGrid=4)
print(a)

a.optimize()

for obj in a.archive:
    print(obj.position, obj.cost)

