import numpy as np
from population.population import Population
import gym

class SRPSO(gym.Env):

    def __init__(self,dim=30,ps=100,max_velocity = 10, max_x=100,min_x=-100, max_fes=100000):
        super().__init__() 
        self.dim = dim
        self.c1=1.49445
        self.c2=1.49445
        self.max_velocity = max_velocity
        self.max_x=max_x
        self.min_x=min_x
        self.ps=ps
        self.elite_num=int(0.25*ps)
        self.max_fes=max_fes

        # print(f'SRPSO with {self.dim} dims .')


    # modified: add gbest_position & gbest_val
    def initialize_particles(self, batch):
        self.batch_size = len(batch)
        # rand_pos=torch.empty(size=(self.batch_size, self.ps, self.dim),dtype=torch.float32).uniform_(-self.max_x,self.max_x).to(self.cuda)
        # rand_vel = torch.empty(size=(self.batch_size, self.ps, self.dim),dtype=torch.float32).uniform_(-self.max_velocity,self.max_velocity).to(self.cuda)
        self.population=Population(self.dim,self.ps,self.min_x,self.max_x,self.max_fes,self.problem)
        self.population.reset()
        self.velocity=np.random.uniform(-self.max_v,self.max_v,(self.ps,self.dim))
        # rand_vel = torch.zeros(size=(self.batch_size, self.ps, self.dim),dtype=torch.float32).to(self.cuda)
        # c_cost = self.get_costs(batch, rand_pos)
        # gbest_val,gbest_index=torch.min(c_cost,dim=1)
        # gbest_position=rand_pos[range(self.batch_size),gbest_index,:]
        # self.max_cost=torch.max(c_cost,dim=1)[0].to(self.cuda)
        # self.particles={'current_position': rand_pos.clone(), # bs, ps, dim
        #                 'c_cost': c_cost.clone(), # bs, ps
        #                 'pbest_position': rand_pos.clone(), # bs, ps, dim
        #                 'pbest': c_cost.clone(), # bs, ps
        #                 'gbest_position':gbest_position.clone(), # bs, dim
        #                 'gbest_val':gbest_val.clone(),  # bs
        #                 'velocity': rand_vel .clone(),
        #                 'gbest_index':gbest_index,
        #                 'c_best':gbest_val.clone(),
        #                 'c_best_index':gbest_index.clone()
        #                 }


    

    # 重置时调用
    def reset(self):
        self.w=np.ones((self.ps,))*0.9
        self.fes=0
        self.population=Population(self.dim,self.ps,self.min_x,self.max_x,self.max_fes,self.problem)
        self.population.reset()
        self.velocity=np.random.uniform(-self.max_velocity,self.max_velocity,(self.ps,self.dim))

        return self.population

    def update_w(self):
        delta_w=0.5/(self.max_fes//self.ps)
        self.w-=delta_w
        self.w[self.population.cbest_index]+=2*delta_w

    def get_p_so(self):
        rnd=np.random.rand(self.ps,self.dim)
        filters=rnd>0.5
        p_so=np.where(filters,np.ones_like(rnd),np.zeros_like(rnd))
        p_so[self.population.cbest_index]=0
        return p_so
    
    def get_p_self(self):
        p_self=np.ones_like(self.population.pbest_cost)
        p_self[self.population.cbest_index]=0
        return p_self[:,None]

    def step(self,action):
        if action.get('problem') is not None:
            # print('change problem!!')
            self.problem=action['problem']
            # self.absolute_min_cost=action['sgbest']
            # self.population.problem=action['problem']
            return None,None,None,{}
        
        step_fes=action['fes']
        next_fes=self.population.cur_fes+step_fes
        
        while self.population.cur_fes<next_fes and self.population.cur_fes<self.max_fes:
            self.update_w()
            p_so=self.get_p_so()
            p_self=self.get_p_self()

            # input action_dim should be : bs, ps
            # action in (0,1) the ratio to learn from pbest & gbest
            rand1=np.random.rand(self.ps,1)
            rand2=np.random.rand(self.ps,1)
            pbest=self.population.pbest_position
            gbest=self.population.gbest_position
            cur_pos=self.population.current_position
            v_pbest=self.c1*rand1*p_self*(pbest-cur_pos)
            v_gbest=self.c2*rand2*p_so*(gbest[None,:]-cur_pos)
            new_velocity=self.w[:,None]*self.velocity+v_pbest+v_gbest
            
            new_velocity=np.clip(new_velocity,-self.max_velocity,self.max_velocity)


            # update position
            new_position = np.clip(cur_pos+ new_velocity,self.min_x,self.max_x)

            self.population.update(new_position)
            self.velocity=new_velocity
        
        return self.population,0,self.population.cur_fes>=self.max_fes,{}