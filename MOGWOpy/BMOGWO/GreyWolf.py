
class GreyWolf:

    def __init__(self):
        self.position=[]
        #self.velocity=[]
        self.cost=[]
        self.dominated=False
        self.best = {}
        self.best['Position']=[] 
        self.best['Cost']=[]
        self.gridIndex=[]
        self.gridSubIndex=[]


    def dominates(self,y):      # returns True if self dominates y
        x= self.cost
        y= y.cost
        #dom = all(x<=y) and any(x<y)
        flag = 1
        dom = 0
        for i in range(len(x)):
            if(x[i]>y[i]):
                flag = 0

        if(flag==1):
            for i in range(len(x)):
                if(x[i]<y[i]):
                    dom = 1
                    break
        return dom