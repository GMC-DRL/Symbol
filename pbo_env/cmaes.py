import numpy as np
from cmaes import CMA, SepCMA
from population.population import Population
import gym


class sep_CMA_ES(gym.Env):
    def __init__(self,dim=30,ps=100,problem=None,
                 max_x=100,min_x=-100,max_fes=100000,sigma=2.0): # boundary is a dim * 2 array
        self.dim = dim
        self.sigma = sigma
        self.population_size = ps
        self.boundary = np.zeros((dim,2))
        self.boundary[:,0]=min_x
        self.boundary[:,1]=max_x
        self.min_x=min_x
        self.max_x=max_x
        self.max_fes=max_fes

    
    def reset(self):
        self.optimizer = SepCMA(mean = np.zeros(self.dim), 
                             sigma = self.sigma, 
                             population_size=self.population_size, 
                             bounds=self.boundary)
        samples = []
        for _ in range(self.population_size):
            samples.append(self.optimizer.ask())
        X = np.vstack(samples)


        self.population=Population(self.dim,self.population_size,self.min_x,self.max_x,self.max_fes,self.problem)
        Y = self.population.get_costs(X)
        self.optimizer.tell(list(zip(X,Y)))
        self.population.reset(init_pop=X,init_y=Y,need_his=False)

        return self.population

    def step(self,action):
        if action.get('problem') is not None:
            self.problem=action['problem']
            return None,None,None,{}
        
        if self.population.cur_fes>=self.max_fes:
            return self.population,0,self.population.cur_fes>=self.max_fes,{}
        
        if action.get('skip_step') is not None:
            skip_step=action['skip_step']
        elif action.get('fes') is not None:
            step_fes=action['fes']
            next_fes=self.population.cur_fes+step_fes
        else:
            assert True, 'action error!!'

        step_end=False
        step=0
        
        while not step_end:
            # sample x at time step t
            samples = []
            for _ in range(self.population_size):
                samples.append(self.optimizer.ask())
            X = np.vstack(samples)
            Y = self.population.get_costs(X)
            self.optimizer.tell(list(zip(X,Y)))
            self.population.update_cmaes(X,Y)
            step+=1
            if action.get('fes') is not None:
                if self.population.cur_fes>=next_fes or self.population.cur_fes>=self.population.max_fes:
                    step_end=True
                    break
            elif action.get('skip_step') is not None:
                if step>=skip_step:
                    step_end=True
                    break
            # print('best solution with costs {}'.format(np.min(Y)))
        return self.population,0,self.population.cur_fes>=self.max_fes,{}