from GreyWolf import *
from Grid import *
import numpy as np
from copy import deepcopy

# Fixing archive addition
# Adding unique solution set

class BMOGWO:

    def __init__(self, dataset, classifier, greyWolvesNum=8, maxIt=100, archiveSize=50,
        nGrid=10, alpha=0.1, beta=4, gamma=2):
        
        self.dim = dataset.shape[1] - 1
        self.dataset = dataset
        self.classifier = classifier
        self.greyWolvesNum = greyWolvesNum
        self.maxIt = maxIt
        self.archiveSize = archiveSize
        self.nGrid = nGrid

        self.alpha = alpha  # Grid inflation parameter
        self.beta = beta    # Leader selection pressure parameter
        self.gamma = gamma  # Repository Member Selection Pressure

        self.greyWolves = self.createWolves()
        self.archive = []
        self.unique_sols = set()

        self.explored = {}  # Keeps track of explored position in each iteration
    
    def fobj(self, selected_features):
        ann = self.classifier(selected_features, self.dataset)
        ann.train()
        err = ann.test_error()
        no_of_features = np.count_nonzero(selected_features)
        return np.array([no_of_features, err])
    
    def optimize(self):
        maxIt = self.maxIt
        greyWolvesNum = self.greyWolvesNum
        beta = self.beta
        gamma = self.gamma
        archiveSize = self.archiveSize
        dim = self.dim


        self.initialize()
        
        for w in self.greyWolves:
            pos_tuple = tuple(w.position)
            self.unique_sols.add(pos_tuple)
        
        self.determineDomination()
        
        # ===================================================
        # initialize archive with non dominated solutions
        nonDominatedSolutions = self.getNonDominatedWolves() 
        archive = []
        for sol in nonDominatedSolutions:
            archive.append(deepcopy(sol))

        self.archive = np.array(archive)

        archiveCosts = self.getCosts()
        grid = createHypercubes(archiveCosts, self.nGrid, self.alpha)

        archive = self.archive
        for i in range(len(self.archive)):
            archive[i].gridIndex, archive[i].gridSubIndex = getGridIndex(archive[i], grid)

        # MOGWO main loop
        for it in range(maxIt):

            self.explored[it] = [ obj.position for obj in self.greyWolves ]
            
            a = 2 - it*((2)/maxIt)

            Alpha = self.selectLeader(beta)
            Beta = self.selectLeader(beta)
            Delta = self.selectLeader(beta)

            print('Archive',self.archive)

            addBack = 0
            if(len(self.archive)>1):
                self.archive = np.delete(self.archive, np.where(self.archive==Alpha))
                Beta = self.selectLeader(beta)
                addBack += 1

            if(len(self.archive)>1):
                self.archive = np.delete(self.archive, np.where(self.archive==Beta))
                Delta = self.selectLeader(beta)
                addBack += 1

            if addBack == 2:    
                self.archive = np.append(self.archive, [ Alpha, Beta ])
            elif addBack == 1:    
                self.archive = np.append(self.archive, [ Alpha ])

            print(f'\nIteration: {it}')
            print(f'alpha: {Alpha.cost}')
            print(f'beta: {Beta.cost}')
            print(f'delta: {Delta.cost}')

            # Update positions
            for w in self.greyWolves:

                #Equations

                # r1 & r2 are random vectors in [0, 1]
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                A1 = a * ((2 * r1) - 1)     
                C1 = 2 * r2

                D_alpha = abs((C1 * Alpha.position) - w.position)
                c_step_alpha = sigmoid(A1 * D_alpha)
                b_step_alpha = (c_step_alpha >= np.random.rand(dim))*1
                X1 = (((Alpha.position + b_step_alpha) >= 1))*1

                # r1 & r2 are random vectors in [0, 1]
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                A1 = a * ((2 * r1) - 1)     
                C1 = 2 * r2

                D_beta = abs((C1 * Beta.position) - w.position)
                c_step_beta = sigmoid(A1 * D_beta)
                b_step_beta = (c_step_beta >= np.random.rand(dim)) * 1
                X2 = (((Beta.position + b_step_beta) >= 1)) * 1

                # r1 & r2 are random vectors in [0, 1]
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                A1 = a * ((2 * r1) - 1)     
                C1 = 2 * r2

                D_delta = abs((C1 * Delta.position) - w.position)
                c_step_delta = sigmoid(A1 * D_delta)
                b_step_delta = (c_step_delta >= np.random.rand(self.dim))*1
                X3 = (((Delta.position + b_step_delta) >= 1))*1

                updated_position = ((sigmoid( (X1 + X2 + X3)/3 ) >= np.random.rand(dim))) * 1
                updated_position_tuple = tuple(updated_position)

                w.updated = False
                if np.count_nonzero(updated_position) != 0:
                    if updated_position_tuple not in self.unique_sols:
                        w.position = updated_position
                        w.cost = self.fobj(w.position)
                        w.updated = True
                        self.unique_sols.add(updated_position_tuple)

            # for w in self.greyWolves:
                

            self.determineDomination()

            nonDominatedSolutions = self.getNonDominatedWolves() 
            
            for sol in nonDominatedSolutions:

                if not sol.updated:
                    continue
                
                case3 = True    # Neither of new solution and archive members dominate each other
                case2 = False   # New solution dominates one or more archive members
                
                for archMem in self.archive:
                    
                    # new soln dominated by any one in archive
                    if archMem.dominates(sol):
                        case3 = False
                        break

                    # new soln dominates any one in archive
                    if sol.dominates(archMem):

                        sol.gridIndex, sol.gridSubIndex = getGridIndex(sol,grid)
                        self.archive = np.delete(self.archive, np.where(self.archive==archMem))
                        case2 = True
                        case3 = False

                if case2:
                    self.archive = np.append(self.archive, deepcopy(sol))
                
                elif case3:
                    if(len(self.archive) < self.archiveSize):
                        self.archive = np.append(self.archive, deepcopy(sol))
                    else:
                        self.deleteFromRepo(1,gamma)
                        sol.gridIndex, sol.gridSubIndex = getGridIndex(sol,grid)
                        self.archive = np.append(self.archive, deepcopy(sol))
            
            archiveCosts = self.getCosts()
            grid = createHypercubes(archiveCosts, self.nGrid, self.alpha) 

            for i in range(len(self.archive)):
                self.archive[i].gridIndex, self.archive[i].gridSubIndex = getGridIndex(self.archive[i],grid)      
                


            # archive = list(self.archive)
            # 
            # for sol in nonDominatedSolutions:
                # archive.append(deepcopy(sol))
            # 
            # self.archive = np.array(archive)


            # [Modification Possible] 

            # archiveCosts = self.getCosts()
            # grid = createHypercubes(archiveCosts, self.nGrid, self.alpha)

            # for i in range(len(self.archive)):
            #     self.archive[i].gridIndex, self.archive[i].gridSubIndex = getGridIndex(self.archive[i],grid)
            
            # if ( len(self.archive) > archiveSize ):
            #     EXTRA = len(self.archive) - archiveSize
            #     self.deleteFromRepo(EXTRA,gamma)

            #     archiveCosts = self.getCosts()
            #     grid = createHypercubes(archiveCosts, self.nGrid, self.alpha)

            

    def createWolves(self):
        wolves = []
        for i in range(self.greyWolvesNum):
            wolves.append(GreyWolf())
        wolves = np.array(wolves)
        return wolves

    def initialize(self):

        greyWolves = self.greyWolves

        for i in range(self.greyWolvesNum):
            #greyWolves[i].velocity = 0
            greyWolves[i].position = (np.random.uniform(size = self.dim) >= 0.5) * 1
            while(np.count_nonzero(greyWolves[i].position)==0):
                greyWolves[i].position = (np.random.uniform(size = self.dim) >= 0.5) * 1
            greyWolves[i].cost = np.asarray(self.fobj(greyWolves[i].position))
            greyWolves[i].best['Position'] = np.copy(greyWolves[i].position)
            greyWolves[i].best['Cost'] = np.copy(greyWolves[i].cost)

    def determineDomination(self):
        #Note here we are taking particles as 1D-array
        greyWolvesNum = self.greyWolvesNum
        greyWolves = self.greyWolves

        #Loop
        for i in range(greyWolvesNum): #(0 .. greyWolvesNum -1)
            greyWolves[i].dominated = False

            for j in range(i):
                if (greyWolves[j].dominated == False):
                    if (greyWolves[i].dominates(greyWolves[j])):
                        greyWolves[j].dominated = True
                    elif (greyWolves[j].dominates(greyWolves[i])):
                        greyWolves[i].dominated = True
                        break
    
    def getNonDominatedWolves(self):
        num_of_obj=self.greyWolvesNum
        pop = self.greyWolves
        nd_pop=[]
        for i in range(num_of_obj):
            if not pop[i].dominated:
                nd_pop.append(pop[i])
        return np.array(nd_pop)

    def getCosts(self):
        pop = self.archive
        #nobj=len(pop[0].cost)
        num_of_obj=len(pop)
        pop_cost = []
        for i in range(num_of_obj):
            pop_cost.append(pop[i].cost)
        pop_cost=np.array(pop_cost)
        costs=np.transpose(pop_cost)
        return costs
    
    def selectLeader(self, beta):

        repo = self.archive

        #Unique occupied grids and no.of objects in each
        occ_cell_index, occ_cell_member_count = self.getOccupiedCells()
        print(occ_cell_member_count)

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

    def deleteFromRepo(self, extra, gamma):
        
        for k in range(extra):
            repo = self.archive

            #Unique occupied grids and no.of objects in each
            occ_cell_index, occ_cell_member_count = self.getOccupiedCells()

            #print(occ_cell_member_count)

            # Deletion has to be done from most crowded area
            p = np.power(occ_cell_member_count, gamma)
            p = p/sum(p)

            # Selecting the grid by rouletteWheelSelection
            selected_cell_index = occ_cell_index[rouletteWheelSelection(p)]

            # Selecting objects from selected grid
            selected_cell_members_index = np.array([ i for i in range(len(repo)) if repo[i].gridIndex == selected_cell_index ])

            # Selecting object to be deleted from selected grid
            del_obj_index = np.random.choice(selected_cell_members_index)

            self.archive = np.delete(repo, del_obj_index)

    def getOccupiedCells(self):

        repo = self.archive

        gridIndices = np.array([ obj.gridIndex for obj in repo ], dtype=object)

        occ_cell_ind = np.unique(gridIndices)

        occ_cell_member_count = np.zeros(len(occ_cell_ind))

        for i in range(len(occ_cell_ind)):
            occ_cell_member_count[i] = np.count_nonzero(gridIndices == occ_cell_ind[i])

        return occ_cell_ind, occ_cell_member_count


def rouletteWheelSelection(p):
    return np.random.choice( np.arange(len(p)), p=p)

# sigmoid fun
def sigmoid(x):
  return 1/(1 + np.exp( (-10) * (x - 0.5) ))