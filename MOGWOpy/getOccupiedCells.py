import numpy as np

def getOccupiedCells(repo):
    
    gridIndices = np.array([ obj.gridIndex for obj in repo ])

    occ_cell_ind = np.unique(gridIndices)

    occ_cell_member_count = np.zeros(len(occ_cell_ind))

    for i in range(len(occ_cell_ind)):
        occ_cell_member_count[i] = np.count_nonzero(gridIndices == occ_cell_ind[i])

    return occ_cell_ind, occ_cell_member_count