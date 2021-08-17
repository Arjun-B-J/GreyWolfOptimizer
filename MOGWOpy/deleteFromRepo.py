from getOccupiedCells import *
from rouletteWheelSelection import *

def deleteFromRepo(repo, extra, gamma):

    for k in range(extra):

        #Unique occupied grids and no.of objects in each
        occ_cell_index, occ_cell_member_count = getOccupiedCells(repo)

        # Deletion has to be done from most crowded area
        p = np.power(occ_cell_member_count, gamma)
        p = p/sum(p)

        # Selecting the grid by rouletteWheelSelection
        selected_cell_index = occ_cell_index[rouletteWheelSelection(p)]

        # Selecting objects from selected grid
        selected_cell_members_index = np.array([ i for i in range(len(repo)) if repo[i].gridIndex == selected_cell_index ])

        # Selecting object to be deleted from selected grid
        del_obj_index = np.random.choice(selected_cell_members_index)

        repo = np.delete(repo, del_obj_index)

    return repo