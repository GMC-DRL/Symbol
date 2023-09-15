
# from population import Population
import numpy as np
from expr.expression import get_prefix_with_consts,prefix_to_infix
import copy
import scipy
from population.population import Population

class TaskForTrain(object):
    def __init__(self,learning_env,teacher_env,baseline_env,bs,opts) -> None:
        self.learning_env=learning_env
        self.teacher_env=teacher_env
        self.baseline_env=baseline_env
        self.bs=bs
        self.opts=opts
        self.train_mode=opts.train_mode
        
    
    def reset(self,problem,testing=False):
        action=[{'problem':copy.deepcopy(problem[i])} for i in range(self.bs)]
        # for p in problem:
        #     print(f'offset:{p.shift}, rotate:{p.rotate}')
        self.teacher_env.step(action)
        self.baseline_env.step(action)
        self.learning_env.step(action)
        
        tea_population=self.teacher_env.reset()
        l_pop=self.learning_env.reset()
        baseline_pop=self.baseline_env.reset()

        # when training stu_pop is fromed from teacher's pop
        if not testing:
            get_init_pop(tea_population,l_pop,self.opts.init_pop)
        
        return tea_population,l_pop
    
    
    def state(self,population):
        pop_feature=[]
        for p in population:
            pop_feature.append(p.feature_encoding(self.opts.fea_mode))
        return pop_feature
    
    
    # modified: support different training mode
    def reward(self,learning_population,target_population,reward_method,base_reward_method,max_step,epoch,s_init_cost,s_gbest,pre_learning_population,ids):
        reward_func={
            'gap_id':cal_gap_id,
            'gap_div':cal_gap_nearest,
            'gap_div5':cal_gap_nearest,
            'gap_near':cal_gap_nearest,
            'w':cal_wasserstein,
            'gap_rank':cal_gap_rank,
        }

        base_reward_func={
            '1':base_reward1,
            '2':base_reward2,
            '2div2':base_reward2,
            '3':base_reward3,
            '4':base_reward4,
            '5':base_reward5,
            '6':base_reward6,
            '7':base_reward7,
            '8':base_reward8,
            '9':base_reward9,
            '10':base_reward10,
        }
        # max_x=learning_population[0].max_x
        
        # 对position归一
        gap_rewards=[]
        gaps=[]
        base_rewards=[]
        total_rewards=[]
        for i,(p1,p2,p3) in enumerate(zip(learning_population,target_population,pre_learning_population)):
            # pure gap
            if self.train_mode == '1' or (self.train_mode == '4' and epoch<self.opts.epoch_end//2):
                base_rewards.append(0)
                dist=reward_func.get(reward_method)(p1,p2,self.opts.gap_mode)
                if reward_method == 'gap_div':
                    r=1/dist/max_step
                    gap_rewards.append(r)
                    # assert (1/dist-1)/max_step>=0,f'gap_div have error, reward:{(1/dist-1)/max_step},gap:{dist}'
                elif reward_method == 'gap_div5':
                    r=1/dist/max_step/5
                    gap_rewards.append(r)
                else:
                    r=-dist/max_step
                    gap_rewards.append(r)

                total_rewards.append(r)
                gaps.append(dist)
            # pure opt reward
            elif self.train_mode == '2' or (self.train_mode == '4' and epoch>=self.opts.epoch_end//2):
                gaps.append(0)
                gap_rewards.append(0)
                base_rewards.append(pure_opt_reward(p3,p1) if self.train_mode=='2' else neg_opt_reward(p1)/max_step)
                total_rewards.append(base_rewards[-1])
            # imitation + breward
            elif self.train_mode == '3' or self.train_mode == '7' or self.train_mode == '10':
                if s_init_cost is not None:
                    b_reward=base_reward_func.get(base_reward_method)(p1,p2,p3,s_gbest[ids[i]],s_init_cost[i])/max_step
                    base_rewards.append(b_reward)
                # if reward_method=='base' and (s_init_cost is not None):
                #     base_rewards.append(base_reward(p1,s_gbest[ids[i]],s_init_cost)/max_step)
                    
                # elif reward_method=='gap_base' and (s_init_cost is not None):
                #     base_rewards.append(base_reward(p1,s_gbest[ids[i]],s_init_cost)/max_step)
                    
                dist=reward_func.get(reward_method)(p1,p2,self.opts.gap_mode)
                if reward_method == 'gap_div':
                    r=1/dist/max_step
                    gap_rewards.append(r)
                    # assert (1/dist-1)/max_step>=0,f'gap_div have error, reward:{(1/dist-1)/max_step},gap:{dist}'
                elif reward_method == 'gap_div5':
                    r=1/dist/max_step/5
                    gap_rewards.append(r)
                else:
                    r=-dist/max_step
                    gap_rewards.append(r)
                if self.train_mode == '3':
                    total_rewards.append(r+b_reward)
                elif self.train_mode == '7':
                    total_rewards.append(r*(1-p1.cur_fes/p1.max_fes)+p1.cur_fes/p1.max_fes*b_reward)
                elif self.train_mode == '10':
                    ratio=(epoch-self.opts.epoch_start)/(self.opts.epoch_end-self.opts.epoch_start)
                    total_rewards.append(r*(1-ratio)+b_reward*(1+ratio))
                gaps.append(dist)
            # only breward
            elif self.train_mode == '5':
                if s_init_cost is not None:
                    b_reward=base_reward_func.get(base_reward_method)(p1,p2,p3,s_gbest[ids[i]],s_init_cost[i])/max_step
                    base_rewards.append(b_reward)
                total_rewards.append(b_reward)
                gap_rewards.append(0)
                gaps.append(0)
            # if stu.gbest<tea.gbest breward else gap_reward
            elif self.train_mode == '6':
                b_reward=base_reward_func.get(base_reward_method)(p1,p2,p3,s_gbest[ids[i]],s_init_cost[i])/max_step
                if base_reward_func == '2div2':
                    b_reward/=2
                base_rewards.append(b_reward)
                dist=reward_func.get(reward_method)(p1,p2,self.opts.gap_mode)
                if reward_method == 'gap_div':
                    r=1/dist/max_step
                    gap_rewards.append(r)
                    # assert (1/dist-1)/max_step>=0,f'gap_div have error, reward:{(1/dist-1)/max_step},gap:{dist}'
                elif reward_method == 'gap_div5':
                    r=1/dist/max_step/5
                    gap_rewards.append(r)
                else:
                    r=-dist/max_step
                    gap_rewards.append(r)
                
                gaps.append(dist)
                if p1.gbest_cost<p2.gbest_cost:
                    total_rewards.append(b_reward)
                else:
                    total_rewards.append(r)
            elif self.train_mode == '8' or self.train_mode == '9':
                # b_reward=pure_opt_reward(p3,p1)-1
                b_reward=neg_opt_reward(p1)/max_step
                base_rewards.append(b_reward)
                # if reward_method=='base' and (s_init_cost is not None):
                #     base_rewards.append(base_reward(p1,s_gbest[ids[i]],s_init_cost)/max_step)
                    
                # elif reward_method=='gap_base' and (s_init_cost is not None):
                #     base_rewards.append(base_reward(p1,s_gbest[ids[i]],s_init_cost)/max_step)
                    
                dist=reward_func.get(reward_method)(p1,p2,self.opts.gap_mode)
                if reward_method == 'gap_div':
                    r=1/dist/max_step
                    gap_rewards.append(r)
                    # assert (1/dist-1)/max_step>=0,f'gap_div have error, reward:{(1/dist-1)/max_step},gap:{dist}'
                elif reward_method == 'gap_div5':
                    r=1/dist/max_step/5
                    gap_rewards.append(r)
                else:
                    r=-dist/max_step
                    gap_rewards.append(r)
                if self.train_mode == '8':
                    total_rewards.append(r+b_reward)
                else:
                    if p1.gbest_cost<p2.gbest_cost:
                        total_rewards.append(b_reward)
                    else:
                        total_rewards.append(r)
                gaps.append(dist)
            else:
                assert True, 'no matched train mode!!'
            
        assert not np.any(np.isnan(gap_rewards)),'reward has nan!!'
        
        return total_rewards,gap_rewards,base_rewards,gaps
        

    # return learning population after applying the expr and the target_env following teacher's instruction
    def step(self,base_population,skip_step,seq,const_seq,tokenizer,rand_seq=None,rand_c_seq=None,baseline_pop=None,only_stu=False,no_tea=False,testing=False):
        
        is_done=False
        expr=[]
        rand_expr=[]
        
        for i in range(seq.shape[0]):
            pre,c_pre=get_prefix_with_consts(seq[i],const_seq[i],0)
            str_expr=[tokenizer.decode(pre[i]) for i in range(len(pre))]
            success,infix=prefix_to_infix(str_expr,c_pre,tokenizer)
            assert success, 'fail to construct the update function'
            expr.append(infix)
            if not only_stu:
                rand_pre,rand_c_pre=get_prefix_with_consts(rand_seq[i],rand_c_seq[i],0)
                rand_str_expr=[tokenizer.decode(rand_pre[i]) for i in range(len(rand_pre))]
                success,rand_infix=prefix_to_infix(rand_str_expr,rand_c_pre,tokenizer)
                assert success, 'fail to construct the update function'
                rand_expr.append(rand_infix)
        
        action=[{'base_population':copy.deepcopy(base_population[i]),'expr':expr[i],'skip_step':skip_step,'select':self.opts.stu_select} for i in range(len(base_population))]
        # learning_env step
        learning_population,_,is_done,_=self.learning_env.step(action)
        # print(learning_population[0].cur_fes)
        if not only_stu:
            # baseline_env step
            action=[{'base_population':copy.deepcopy(baseline_pop[i]),'expr':rand_expr[i],'skip_step':skip_step,'select':self.opts.stu_select} for i in range(len(base_population))]
            baseline_population,_,_,_=self.baseline_env.step(action)

            # teacher_env step
            tea_select=self.opts.tea_select
            if self.opts.tea_step == 'step':
                if testing:
                    skip_step=int((self.opts.max_fes//self.opts.tea_fes)*skip_step)
                    action=[{'skip_step':skip_step,'select':tea_select} for i in range(len(base_population))]
                else:
                    if self.opts.teacher=='glpso' and skip_step!=1:
                        skip_step=skip_step//2
                    action=[{'skip_step':skip_step,'select':tea_select} for i in range(len(base_population))]
            elif self.opts.tea_step == 'fes':
                action=[{'fes':skip_step*self.opts.population_size,'select':tea_select} for i in range(self.bs)]
            teacher_population,_,_,_=self.teacher_env.step(action)
        else:
            teacher_population=None
            baseline_population=None
        # print(teacher_population[0].current_position==learning_population[0].current_position)
        # print(f'stepteacher:{teacher_population[0].current_position.shape}')
        # print(f'stepstu:{learning_population[0].current_position.shape}')
        return teacher_population,learning_population,baseline_population,expr,is_done.all()

# ! change to max
# def cal_gap(position1,position2,max_x):
#     norm_p1=position1/max_x
#     norm_p2=position2/max_x
#     dim=position1.shape[1]
#     max_dist=2*np.sqrt(dim)
#     gap=np.max(np.sqrt(np.sum((norm_p1-norm_p2)**2,axis=-1)))
#     return gap/max_dist

def cal_gap_nearest(stu_pop,tea_pop,mode):
    max_x=stu_pop.max_x
    if mode=='after':
        stu_position=stu_pop.current_position
        tea_position=tea_pop.current_position
    elif mode=='before':
        stu_position=stu_pop.before_select_pos
        tea_position=tea_pop.before_select_pos
    else:
        assert True, 'gap mode is not supported!!'
    norm_p1=stu_position/max_x
    norm_p1=norm_p1[None,:,:]
    norm_p2=tea_position/max_x
    norm_p2=norm_p2[:,None,:]
    dist=np.sqrt(np.sum((norm_p2-norm_p1)**2,-1))
    min_dist=np.min(dist,-1)

    # ? max or mean
    gap=np.max(min_dist)
    dim=stu_position.shape[1]
    max_dist=2*np.sqrt(dim)
    return gap/max_dist


'''base reward functions'''
def base_reward(stu_pop,s_gbest,s_init_cost):
    factor=10
    # ! error
    if stu_pop.gbest_cost<s_gbest:
        r=(s_gbest-stu_pop.gbest_cost)/(s_init_cost-s_gbest)*factor
        # r=(stu_pop.gbest_cost-s_gbest)/(s_init_cost-s_gbest)/factor
    else:
        r=-(stu_pop.gbest_cost-s_gbest)/(s_init_cost-s_gbest)
    # ! change
    # if r<-1:
    #     print(r)
    return r

# -tanh((sg-g)/(tg-g))
def base_reward1(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=-np.tanh((stu_pop.gbest_cost-s_gbest)/(tea_pop.gbest_cost-s_gbest))
    return r

# -tanh((w-tg)/(w-sg))
def base_reward2(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=-np.tanh((s_init_cost-tea_pop.gbest_cost)/(s_init_cost-stu_pop.gbest_cost))
    return r

def base_reward2p(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=-np.tanh((s_init_cost-tea_pop.gbest_cost)/(s_init_cost-stu_pop.gbest_cost))
    return r

# (g-sg)/(w-g)
def base_reward3(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=(s_gbest-stu_pop.gbest_cost)/(s_init_cost-s_gbest)
    return r

# (w-sg)/(w-g)
def base_reward4(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=(s_init_cost-stu_pop.gbest_cost)/(s_init_cost-s_gbest)
    return r


def base_reward5(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=-(stu_pop.gbest_cost-s_gbest)/(stu_pop.init_cost-s_gbest)
    return r

def base_reward6(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=-np.tanh(max(stu_pop.gbest_cost-s_gbest,1e-8)/max(tea_pop.gbest_cost-s_gbest,1e-8))
    return r

def base_reward7(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=(-np.tanh((stu_pop.gbest_cost-tea_pop.gbest_cost)/max(tea_pop.gbest_cost-s_gbest,1e-8))-1)/2
    return r

def base_reward8(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=-np.tanh((stu_pop.gbest_cost-s_gbest)/max(tea_pop.gbest_cost-s_gbest,1e-8))
    return r

# -tanh((t.init_cost-tg)/(t.init_cost-g)/(s.init_cost-sg)/(s.init_cost-g))
def base_reward9(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=-np.tanh(((tea_pop.init_cost-tea_pop.gbest_cost+1e-8)/(tea_pop.init_cost-s_gbest+1e-8))/((stu_pop.init_cost-stu_pop.gbest_cost+1e-8)/(stu_pop.init_cost-s_gbest+1e-8)))
    return r

# -tanh((t.init_cost-tg)/(t.init_cost-g)/(s.init_cost-sg)/(s.init_cost-g)-1)
def base_reward10(stu_pop,tea_pop,pre_stu_pop,s_gbest,s_init_cost):
    r=-np.tanh(((tea_pop.init_cost-tea_pop.gbest_cost+1e-8)/(tea_pop.init_cost-s_gbest+1e-8))/((stu_pop.init_cost-stu_pop.gbest_cost+1e-8)/(stu_pop.init_cost-s_gbest+1e-8))-1)
    return r

'''pure opt reward function'''
def pure_opt_reward(pre_stu_pop,next_stu_pop):
    assert pre_stu_pop.init_cost>pre_stu_pop.problem.optimum, 'error: init cost == problem.optimum!!'
    r=(pre_stu_pop.gbest_cost-next_stu_pop.gbest_cost)/(pre_stu_pop.init_cost-pre_stu_pop.problem.optimum)
    return r

# neg optimization reward, used in train mode 4
def neg_opt_reward(stu_pop):
    assert stu_pop.init_cost>stu_pop.problem.optimum, 'error: init cost == problem.optimum!!'
    r=-(stu_pop.gbest_cost-stu_pop.problem.optimum)/(stu_pop.init_cost-stu_pop.problem.optimum)
    return r

'''gap calculating function'''
def cal_gap_id(stu_pop,tea_pop,mode):
    if mode=='after':
        stu_position=stu_pop.current_position
        tea_position=tea_pop.current_position
    elif mode=='before':
        stu_position=stu_pop.before_select_pos
        tea_position=tea_pop.before_select_pos
    else:
        assert True, 'gap mode is not supported!!'
    max_x=stu_pop.max_x
    stu_position=stu_pop.current_position
    tea_position=tea_pop.current_position
    index=tea_pop.index
    
    norm_p1=stu_position[index]/max_x
    norm_p2=tea_position/max_x
    dim=stu_position.shape[1]
    max_dist=2*np.sqrt(dim)
    gap=np.max(np.sqrt(np.sum((norm_p1-norm_p2)**2,axis=-1)))
    return gap/max_dist


def cal_gap_rank(stu_pop,tea_pop,mode):
    if mode=='after':
        stu_position=stu_pop.current_position
        tea_position=tea_pop.current_position
    elif mode=='before':
        stu_position=stu_pop.before_select_pos
        tea_position=tea_pop.before_select_pos
    else:
        assert True, 'gap mode is not supported!!'


    dim=stu_position.shape[1]
    max_dist=2*np.sqrt(dim)
    max_x=stu_pop.max_x
    norm_p1=stu_position/max_x
    norm_p1=norm_p1[None,:,:]
    norm_p2=tea_position/max_x
    norm_p2=norm_p2[:,None,:]
    dist=np.sqrt(np.sum((norm_p2-norm_p1)**2,-1))
    min_dist=np.min(dist,-1)/max_dist
    tea_cur_cost=tea_pop.c_cost
    cost_rank=np.argsort(np.argsort(-tea_cur_cost))+1
    gap=(min_dist*cost_rank)/np.sum(cost_rank)
    # print(f'gap_rank:{np.sum(gap)}')
    return np.sum(gap)

def cal_wasserstein(stu_pop,tea_pop,mode):
    if mode=='after':
        stu_position=stu_pop.current_position
        tea_position=tea_pop.current_position
    elif mode=='before':
        stu_position=stu_pop.before_select_pos
        tea_position=tea_pop.before_select_pos
    else:
        assert True, 'gap mode is not supported!!'

        
    max_x=stu_pop.max_x
    norm_p1=stu_pop.current_position/max_x
    norm_p2=tea_pop.current_position/max_x
    dim=norm_p1.shape[1]
    ws=[]
    for i in range(dim):
        a=norm_p1[:,i]
        b=norm_p2[:,i]
        w_dist=scipy.stats.wasserstein_distance(a,b)
        ws.append(w_dist)
        
    return np.max(ws)


'''forming init pop'''
def get_init_pop(tea_pop_list,stu_pop_list,method):
    if method=='best':
        for tea_pop,stu_pop in zip(tea_pop_list,stu_pop_list):
            sort_index=np.argsort(tea_pop.c_cost)
            init_pos=tea_pop.current_position[sort_index[:stu_pop.pop_size]]
            stu_pop.reset(init_pop=init_pos)
    elif method == 'harf':
        for tea_pop,stu_pop in zip(tea_pop_list,stu_pop_list):
            sort_index=np.argsort(tea_pop.c_cost)
            init_pos=np.concatenate((tea_pop.current_position[sort_index[:int(stu_pop.pop_size*0.5)]],tea_pop.current_position[sort_index[:stu_pop.pop_size-int(stu_pop.pop_size*0.5)]]),axis=0)
            stu_pop.reset(init_pop=init_pos)
    elif method == 'random':
        for tea_pop,stu_pop in zip(tea_pop_list,stu_pop_list):
            rand_index=np.random.randint(0,tea_pop.pop_size,size=(stu_pop.pop_size,))
            init_pos=tea_pop.current_position[rand_index]
            stu_pop.reset(init_pop=init_pos)
    elif method == 'uniform':
        for tea_pop,stu_pop in zip(tea_pop_list,stu_pop_list):
            sort_index=np.argsort(tea_pop.c_cost)
            init_pos=tea_pop.current_position[sort_index[::tea_pop.pop_size//stu_pop.pop_size]]
            stu_pop.reset(init_pop=init_pos)
    else:
        raise ValueError('init pop method is currently not supported!!')