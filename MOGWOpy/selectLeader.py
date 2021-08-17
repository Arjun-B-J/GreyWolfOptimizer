from getOccupiedCells import *
from rouletteWheelSelection import *
import numpy as np


def selectLeader(repo, beta):

    #Array of gridIndices of objects in archive
    #gridIndices = np.array([ obj.gridIndex for obj in repo ])

    #Unique occupied grids and no.of objects in each
    occ_cell_index, occ_cell_member_count = getOccupiedCells(repo)

    # Leader has to be selected from least crowded area.
    # Probability inversely related to count

    p = np.power(occ_cell_member_count, -beta)
    p = p/sum(p)

    # Selecting the grid by rouletteWheelSelection
    selected_cell_index = occ_cell_index[rouletteWheelSelection(p)]

    # Selecting objects from selected grid
    selected_cell_members = np.array([ obj for obj in repo if obj.gridIndex == selected_cell_index ])

    # Selecting random object from selected grid
    return np.random.choice(selected_cell_members)