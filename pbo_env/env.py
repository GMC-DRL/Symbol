import numpy as np
from population.population import Population
from population.madde_population import MadDE_Population
from expr.expression import *
from expr.tokenizer import MyTokenizer
import re
import copy
from utils.utils import set_seed
import numpy as np
import gym
from gym import spaces
from scipy import integrate

'''Working environment for SYMBOL'''

# preprocess the randx operand
class Myreplace(object):
    def __init__(self):
        self.count = 0 

    def replace(self, match):
        self.count += 1
        if self.count > 1:
            return self.pattern + str(self.count-1)
        else:
            return match.group()

    def process_string(self, string, pattern):
        self.pattern=pattern
        self.count = 0
        new_string = re.sub(pattern, self.replace, string)
        return new_string


# inherit gym.Env for the convenience of parallelism
class L2E_env(gym.Env):
    def __init__(self,dim=30,ps=100,problem=None,
                 max_x=100,min_x=-100,max_fes=100000,
                 boarder_method='periodic'):
        super(L2E_env,self).__init__()
        
        self.tokenizer=MyTokenizer()

        self.dim = dim
        
        
        self.max_x=max_x
        self.min_x=min_x
        
        self.NP=ps
        # instance
        self.problem=problem
        
        self.no_improve=0
        self.per_no_improve=np.zeros((self.NP,))
        self.fes=0
        self.evaling=False
        self.max_fes=max_fes
        self.max_dist=np.sqrt((2*max_x)**2*dim)
        
        self.boarder_method=boarder_method
        

        self.action_space=spaces.Box(low=0,high=1,shape=(self.NP,))
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.NP,9))
        self.replace=Myreplace()
        self.name='L2E_env'
        

    # the interface for environment reseting
    def reset(self):
        # self.NP=self.__Nmax
        self.population=Population(self.dim,self.NP,self.min_x,self.max_x,self.max_fes,self.problem)
        self.population.reset()
        return self.population

    def eval(self):
        self.evaling=True

    def train(self):
        # set_seed()
        self.evaling=False

    
    # feature encoding
    def observe(self):
        return self.population.feature_encoding()
 

    def seed(self, seed=None):
        np.random.seed(seed)

    def update_population(self,next_position,pre_population):
        new_cost=self.get_costs(next_position)

    def get_random_pbest(self,population):
        p_rate= ( 0.4  -   1  ) * population.cur_fes/self.max_fes + 1
        p_random_index = np.random.randint(0, int(np.ceil(self.NP*p_rate)), size=(self.NP))

        sorted_index=np.argsort(population.pbest_cost)
        sorted_pbest_pos=population.pbest_position[sorted_index]
        return sorted_pbest_pos[p_random_index]

    # input the base_population and expr function, return the population after applying expr function
    def step(self, action):
        # change working problem instance
        if action.get('problem') is not None:
            self.problem=action['problem']
            return None,None,None,{}
        
        base_population=action['base_population']
        expr=action['expr']
        skip_step=action['skip_step']

        # record the previous gbest
        base_population.pre_gbest=base_population.gbest_cost

        cnt_randx=expr.count('randx')
        pattern = 'randx'
        expr=self.replace.process_string(expr, pattern)
        count = self.replace.count

        assert count==cnt_randx,'randx count is wrong!!'
        variables=copy.deepcopy(self.tokenizer.variables)
        for i in range(1,count):
            variables.append(f'randx{i}')
        update_function=expr_to_func(expr,variables)

        for sub_step in range(skip_step):
            x=base_population.current_position
            
            gb=base_population.gbest_position[None,:].repeat(self.NP,0)
            gw=base_population.gworst_position[None,:].repeat(self.NP,0)

            dx=base_population.delta_x
            randx=x[np.random.randint(self.NP, size=self.NP)]
            
            pbest=base_population.pbest_position
            
            names = locals()
            inputs=[x,gb,gw,dx,randx,pbest]
            for i in range(1,count):
                names[f'randx{i}']=x[np.random.randint(self.NP, size=self.NP)]
                inputs.append(eval(f'randx{i}'))
            assert x.shape==gb.shape==gw.shape==dx.shape==randx.shape, 'not same shape'
            
            next_position=x+update_function(*inputs)
            
            # boarder clip or what
            if self.boarder_method=="clipping":
                next_position=np.clip(next_position,self.min_x,self.max_x)
            elif self.boarder_method=="periodic":
                next_position=self.min_x+(next_position-self.max_x)%(self.max_x-self.min_x)
            else:
                raise AssertionError('this board method is not surported!')
            
            # update population
            base_population.update(next_position)
        
        return base_population,0,base_population.cur_fes>=self.max_fes,{}
    