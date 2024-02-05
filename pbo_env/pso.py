import numpy as np
import gym
from population.population import Population

class PSO(gym.Env):
    def __init__(self,ps,dim,max_fes,min_x,max_x,pho) -> None:
        super().__init__() 
        self.ps=ps
        self.dim=dim
        self.max_fes=max_fes
        self.min_x=min_x
        self.max_x=max_x
        self.max_v=pho*(self.max_x-self.min_x)
        self.w=0.7298
        self.c1=2.
        self.c2=2.

    def reset(self):
        self.population=Population(self.dim,self.ps,self.min_x,self.max_x,self.max_fes,self.problem)
        self.population.reset()
        self.velocity=np.random.uniform(-self.max_v,self.max_v,(self.ps,self.dim))
        return self.population
    
    def step(self,action):
        if action.get('problem') is not None:
            self.problem=action['problem']
            return None,None,None,{}
        
        # skip_step=action['skip_step']


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
            x=self.population.current_position
            gbest=self.population.gbest_position
            pbest=self.population.pbest_position
            self.velocity=self.w*self.velocity+self.c1*(pbest-x)+self.c2*(gbest-x)
            self.velocity=np.clip(self.velocity,-self.max_v,self.max_v)
            new_x=np.clip(x+self.velocity,self.min_x,self.max_x)
            self.population.update(new_x)

            step+=1
            if action.get('fes') is not None:
                if self.population.cur_fes>=next_fes or self.population.cur_fes>=self.max_fes:
                    step_end=True
                    break
            elif action.get('skip_step') is not None:
                if step>=skip_step:
                    step_end=True
                    break
            
        return self.population,0,self.population.cur_fes>=self.max_fes,{}