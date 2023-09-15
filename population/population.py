import numpy as np
from utils.utils import set_seed
import torch
from scipy.spatial.distance import cdist
import queue
import copy

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2,axis=-1))
    

class Population(object):
    def __init__(self,dim,pop_size,min_x,max_x,max_fes,problem):
        self.dim=dim
        self.pop_size=pop_size
        self.min_x=min_x
        self.max_x=max_x
        self.max_fes=max_fes
        self.problem=problem
        self.fea_dim=8
        self.max_dist=((max_x-min_x)**2*dim)**0.5
        # self.g=g
        # self.b=b
        # self.absolute_min_cost=sgbest
        self.cur_fes=0

    # calculate costs of solutions
    def get_costs(self,position):
        ps=position.shape[0]
        self.cur_fes+=ps

        # ! change
        cost=self.problem.func(position)
        
        return cost
    
    def reset(self,init_pop=None,init_y=None,need_his=True):
        np.random.seed()
        # set_seed()
        # print('reset!!')
        # init fes and stag_count
        if init_y is not None:
            self.cur_fes+=init_y.shape[0]
        else:
            self.cur_fes=0
        self.stag_count=0
        
        # init population
        if init_pop is None:
        # randomly generate the position and velocity
            rand_pos=np.random.uniform(low=-self.max_x,high=self.max_x,size=(self.pop_size,self.dim))
        else:
            rand_pos=init_pop

        self.current_position=rand_pos.copy()
        self.dx=np.zeros_like(rand_pos)
        self.delta_x=np.zeros_like(rand_pos)
        # print(f'dx.shape:{dx.shape}')
        # get the initial cost
        if init_y is None:
            self.c_cost = self.get_costs(self.current_position) # ps
        else:
            self.c_cost = init_y

        # init pbest related
        self.pbest_position=rand_pos.copy()
        self.pbest_cost=self.c_cost.copy()

        # find out the gbest_val
        self.gbest_cost = np.min(self.c_cost)
        gbest_index = np.argmin(self.c_cost)
        self.gbest_position=rand_pos[gbest_index]
        

        # init cbest related
        self.cbest_cost=self.gbest_cost
        self.cbest_position=self.gbest_position
        self.cbest_index=gbest_index
        
        # init gworst related
        self.gworst_cost=np.max(self.c_cost)
        gworst_index=np.argmax(self.c_cost)
        self.gworst_position=rand_pos[gworst_index]

        # record
        self.init_cost=np.min(self.c_cost)
        self.pre_position=self.current_position
        self.pre_cost=self.c_cost
        self.pre_gbest=self.gbest_cost
        
        # init historical cost
        # if need_his:
        #     self.historical_relative_h=[]
        #     self.historical_absolute_h=[]
        #     self.update_h()

        # print(f'self.dx:{self.dx}')
    
    def update_h(self):
        self.historical_relative_h.append(get_h(self.c_cost,self.b,np.min(self.c_cost),np.max(self.c_cost)))
        # ? in the absolute h calculation, what should the two min and max be?
        self.historical_absolute_h.append(get_h(self.c_cost,self.b,self.absolute_min_cost,self.gworst_cost))
        self.historical_absolute_h=self.historical_absolute_h[-self.g:]
        self.historical_relative_h=self.historical_relative_h[-self.g:]
        


    def update(self,next_position,filter_survive=False):
        self.pre_cost=self.c_cost
        self.pre_position=copy.deepcopy(self.current_position)
        # self.pre_gbest=self.gbest_cost

        self.before_select_pos=next_position

        new_cost=self.get_costs(next_position)
        if filter_survive:
            surv_filter=new_cost<=self.c_cost
            next_position=np.where(surv_filter[:,None],next_position,self.current_position)
            new_cost=np.where(surv_filter,new_cost,self.c_cost)

       
        # update particles
        filters = new_cost < self.pbest_cost
        
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)
        
        # 重新生成rand_parent
        # rand_index=np.random.randint(0,self.ps,size=(self.ps))

        self.current_position=next_position
        self.c_cost=new_cost
        self.pbest_position=np.where(np.expand_dims(filters,axis=-1),
                                                        next_position,
                                                        self.pbest_position)
        self.pbest_cost=np.where(filters,
                                new_cost,
                                self.pbest_cost)
        if new_cbest_val<self.gbest_cost:
            self.gbest_cost=new_cbest_val
            self.gbest_position=self.current_position[new_cbest_index]
            # gbest_index=new_cbest_index
            self.stag_count=0
        else:
            self.stag_count+=1

        self.cbest_cost=new_cbest_val
        self.cbest_position=next_position[new_cbest_index]
        self.cbest_index=new_cbest_index

        new_cworst_val=np.max(new_cost)
        if new_cworst_val>self.gworst_cost:
            self.gworst_cost=new_cworst_val
            gworst_index=np.argmax(new_cost)
            self.gworst_position=next_position[gworst_index]
        
        # dx 的计算不知道对不对
        self.dx=(self.c_cost-self.pre_cost)[:,None]/(self.current_position-self.pre_position+1e-5)
        self.dx=np.where(np.isnan(self.dx),np.zeros_like(self.current_position),self.dx)

        self.delta_x=self.current_position-self.pre_position
        # # update historical h
        # self.update_h()
    
    def update_cmaes(self,next_position,next_y):
        self.pre_cost=self.c_cost
        self.pre_position=self.current_position
        # self.pre_gbest=self.c_cost

        new_cost=next_y

       
        # update particles
        filters = new_cost < self.pbest_cost
        
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)
        
        # 重新生成rand_parent
        # rand_index=np.random.randint(0,self.ps,size=(self.ps))

        self.current_position=next_position
        self.c_cost=new_cost
        self.pbest_position=np.where(np.expand_dims(filters,axis=-1),
                                                        next_position,
                                                        self.pbest_position)
        self.pbest_cost=np.where(filters,
                                new_cost,
                                self.pbest_cost)
        if new_cbest_val<self.gbest_cost:
            self.gbest_cost=new_cbest_val
            self.gbest_position=self.current_position[new_cbest_index]
            # gbest_index=new_cbest_index
            self.stag_count=0
        else:
            self.stag_count+=1

        self.cbest_cost=new_cbest_val
        self.cbest_position=next_position[new_cbest_index]
        self.cbest_index=new_cbest_index

        new_cworst_val=np.max(new_cost)
        if new_cworst_val>self.gworst_cost:
            self.gworst_cost=new_cworst_val
            gworst_index=np.argmax(new_cost)
            self.gworst_position=next_position[gworst_index]
        

    # 问题：模型针对的是整个种群，但是编码却是对于一个个体而言的？
    # def feature_encoding(self):
    #     fea_1=(self.c_cost-self.gbest_cost)/(self.gworst_cost-self.gbest_cost)
        
    #     fea_2=(np.sum(self.c_cost/self.pop_size,axis=-1)-self.gbest_cost)/(self.gworst_cost-self.gbest_cost)
    #     fea_2=np.full((self.pop_size,),fea_2)
        
    #     fit=np.zeros_like(self.c_cost)
    #     fit[:self.pop_size//2]=self.gworst_cost
    #     fit[self.pop_size//2:]=self.gbest_cost
    #     maxstd=np.std(fit)
    #     fea_3=np.std(self.c_cost)/maxstd
    #     fea_3=np.full((self.pop_size,),fea_3)

    #     fea_4=(self.max_fes-self.cur_fes)/self.max_fes
    #     fea_4=np.full((self.pop_size,),fea_4)
    #     # dim related no 
    #     # fea_5=?

    #     # todo stag_count
    #     fea_6=self.stag_count/self.max_fes
    #     fea_6=np.full((self.pop_size,),fea_6)

    #     # problem : random index is the same in the whole population?
    #     # r_index=random_index(5,0,self.pop_size)
    #     # fea_7=dist(self.current_position,self.current_position[:,r_index[0]])/self.max_dist
    #     # fea_8=dist(self.current_position,self.current_position[:,r_index[1]])/self.max_dist
    #     # fea_9=dist(self.current_position,self.current_position[:,r_index[2]])/self.max_dist
    #     # fea_10=dist(self.current_position,self.current_position[:,r_index[3]])/self.max_dist
    #     # fea_11=dist(self.current_position,self.current_position[:,r_index[4]])/self.max_dist
        

    #     fea_12=dist(self.current_position,self.cbest_position[None,:])/self.max_dist

    #     # r_index=random_index(5,0,self.pop_size)
    #     # fea_13=(self.c_cost-self.c_cost[:,r_index[0]])/(self.gworst_cost-self.gbest_cost)
    #     # fea_14=(self.c_cost-self.c_cost[:,r_index[1]])/(self.gworst_cost-self.gbest_cost)
    #     # fea_15=(self.c_cost-self.c_cost[:,r_index[2]])/(self.gworst_cost-self.gbest_cost)
    #     # fea_16=(self.c_cost-self.c_cost[:,r_index[3]])/(self.gworst_cost-self.gbest_cost)
    #     # fea_17=(self.c_cost-self.c_cost[:,r_index[4]])/(self.gworst_cost-self.gbest_cost)

    #     fea_18=(self.c_cost-self.gbest_cost)/(self.gworst_cost-self.gbest_cost)

    #     fea_19=dist(self.current_position,self.gbest_position[None,:])/self.max_dist

    #     return np.concatenate((fea_1[:,None],fea_2[:,None],fea_3[:,None],fea_4[:,None],fea_6[:,None],fea_12[:,None],fea_18[:,None],fea_19[:,None]),axis=-1)

    # 针对种群的版本
    def feature_encoding(self,fea_mode):
        if fea_mode=='xy':
            return np.concatenate((self.current_position,self.c_cost[:,None]),-1)
        assert self.gbest_cost!=self.gworst_cost,f'gbest == gworst!!,{self.gbest_cost}'
        fea_1=(self.c_cost-self.gbest_cost)/(self.gworst_cost-self.gbest_cost+1e-8)
        fea_1=np.mean(fea_1)
        
        fea_2=calculate_mean_distance(self.current_position)/self.max_dist

        fit=np.zeros_like(self.c_cost)
        fit[:self.pop_size//2]=self.gworst_cost
        fit[self.pop_size//2:]=self.gbest_cost
        maxstd=np.std(fit)
        fea_3=np.std(self.c_cost)/(maxstd+1e-8)

        fea_4=(self.max_fes-self.cur_fes)/self.max_fes

        fea_6=self.stag_count/(self.max_fes//self.pop_size)
        
        fea_12=dist(self.current_position,self.cbest_position[None,:])/self.max_dist
        fea_12=np.mean(fea_12)

        fea_18=(self.c_cost-self.cbest_cost)/(self.gworst_cost-self.gbest_cost+1e-8)
        fea_18=np.mean(fea_18)

        fea_19=dist(self.current_position,self.gbest_position[None,:])/self.max_dist
        fea_19=np.mean(fea_19)

        fea_20=0
        if self.gbest_cost<self.pre_gbest:
            fea_20=1
        if fea_mode=='full':
            feature=np.array([fea_1,fea_2,fea_3,fea_4,fea_6,fea_12,fea_18,fea_19,fea_20])
        elif fea_mode=='no_dis':
            # feature=np.array([fea_1,fea_2,fea_3,fea_4,fea_6,fea_19,fea_20])
            feature=np.array([fea_1,fea_3,fea_4,fea_6,fea_18,fea_20])
        elif fea_mode=='no_fit':
            feature=np.array([fea_2,fea_4,fea_6,fea_12,fea_19,fea_20])
        elif fea_mode=='no_opt':
            feature=np.array([fea_1,fea_2,fea_3,fea_12,fea_18,fea_19])
        elif fea_mode=='only_dis':
            feature=np.array([fea_2,fea_12,fea_19])
        elif fea_mode=='only_fit':
            feature=np.array([fea_1,fea_3,fea_18])
        elif fea_mode=='only_opt':
            feature=np.array([fea_4,fea_6,fea_20])

        assert not np.any(np.isnan(feature)),f'feature has nan!!,{feature}'
        return feature
    
def calculate_mean_distance(population):
    # 计算个体之间的距离矩阵
    distances = cdist(population, population, metric='euclidean')
    
    # 排除对角线上的距离（个体与自身的距离为0）
    np.fill_diagonal(distances, 0)
    
    # 计算平均距离
    mean_distance = np.mean(distances)
    
    return mean_distance

# todo: debug, 以及这个normalize真的有必要吗？
# can be used to calculate relative h or absolute h
def get_h(cur_cost,b,min_cost,max_cost):
    ps=cur_cost.shape[0]
    cost_norm=(cur_cost-min_cost)/(max_cost-min_cost)
    cost_norm=np.clip(cost_norm,0,1)
    separators=np.linspace(0,1,b+1)
    h=[]
    for i in range(b):
        filters=np.logical_and(cost_norm>=separators[i],cost_norm<separators[i+1])
        h.append(np.sum(filters)/ps)
    # print(f'h_sum:{np.sum(h)}')
    # print(f'h:{h}')

    return h