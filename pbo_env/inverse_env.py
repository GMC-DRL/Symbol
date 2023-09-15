from .env import L2E_env
import numpy as np
from expr.expression import *
import random

class inverse_agent_random(L2E_env):
    def __init__(self, dim=30, ps=100, problem=None, max_x=100, min_x=-100, max_fes=100000, boarder_method='clipping',exprs=None):
        super().__init__(dim, ps, problem, max_x, min_x, max_fes, boarder_method)
        self.exprs=exprs
        self.n_exprs=len(self.exprs)
    
    def step(self, action):
        if action.get('problem') is not None:
            # print('change problem!!')
            self.problem=action['problem']
            # self.population.problem=action['problem']
            return None,None,None,{}
        
        base_population=action['base_population']
        skip_step=action['skip_step']
        expr=random.choice(self.exprs)
        update_function=expr_to_func(expr,self.tokenizer.variables)
        for sub_step in range(skip_step):
            # self.__sort(base_population)
            x=base_population.current_position
            
            gb=base_population.gbest_position[None,:].repeat(self.NP,0)
            gw=base_population.gworst_position[None,:].repeat(self.NP,0)
            dx=base_population.dx
            randx=x[np.random.randint(self.NP, size=self.NP)]
            pbest=base_population.pbest_position
            # print(f'x.shape:{x.shape},gb.shape:{gb.shape},gw.shape:{gw.shape},dx.shape:{dx.shape}')
            assert x.shape==gb.shape==gw.shape==dx.shape==randx.shape, 'not same shape'
            
            # change to x+\delta_x
            next_position=x+update_function(x,gb,gw,dx,randx,pbest)
            
            # update hit_wall
            clip_filters=np.abs(next_position)>self.max_x
            self.hit_wall+=np.sum(np.any(clip_filters,-1))
            # boarder clip or what
            next_position=np.clip(next_position,self.min_x,self.max_x)
            # print(f'next_position:{next_position.shape}')
            # update population
            # self.update_population(next_position)
            base_population.update(next_position,filter_survive=False)

        
        return base_population,0,base_population.cur_fes>=self.max_fes,{'hit_wall':self.hit_wall}

