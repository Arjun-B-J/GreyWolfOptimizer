import numpy as np

def rouletteWheelSelection(p):
    
    return np.random.choice( np.arange(len(p)), p=p)