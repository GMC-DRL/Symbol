from model.rnn import rnn
from model.critic import Critic
from utils.logger import log_to_tb_train,gen_overall_tab,log_weight
from utils.utils import clip_grad_norms
from .rollout import rollout
from utils.utils import set_seed
from dataset.generate_dataset import generate_dataset,make_dataset,construct_problem_set,sample_batch_task,get_test_set,get_train_set,sample_batch_task_cec21,get_train_set_cec21
from dataset.cec_test_func import *
import numpy as np
from .task import TaskForTrain
from env import SubprocVectorEnv,DummyVectorEnv
# from pbo_env.env import L2E_env
# from pbo_env.glpso import GL_PSO
# from pbo_env.madde import MadDE
# from pbo_env.cmaes import sep_CMA_ES
from pbo_env import L2E_env,GL_PSO,MadDE,sep_CMA_ES,PSO,DE,SRPSO
from expr.tokenizer import MyTokenizer
import torch
import platform
from tqdm import tqdm
from utils.utils import torch_load_cpu, get_inner_model, get_init_cost, get_surrogate_gbest
import os

class Data_Memory():
    def __init__(self) -> None:
        self.teacher_cost=[]
        self.stu_cost=[]
        self.baseline_cost=[]
        self.gap=[]
        self.baseline_gap=[]
        self.expr=[]
    
    def clear(self):
        del self.teacher_cost[:]
        del self.stu_cost[:]
        del self.baseline_cost[:]
        del self.gap[:]
        del self.baseline_gap[:]
        del self.expr[:]

# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.gap_rewards=[]
        self.b_rewards=[]

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.gap_rewards[:]
        del self.b_rewards[:]

class trainer(object):
    def __init__(self,model,opts) -> None:
        self.actor=model
        self.critic=Critic(opts)
        self.opts=opts
        self.optimizer=torch.optim.Adam([{'params':self.actor.parameters(),'lr':opts.lr}] + 
                                        [{'params':self.critic.parameters(),'lr':opts.lr_critic}])
        # figure out the lr schedule
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        if opts.use_cuda:
            # move to cuda
            self.actor.to(opts.device)
            self.critic.to(opts.device)
        # load model to cuda
        # move optimizer's data onto chosen device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    def set_training(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        self.critic.train()
    
    def set_evaling(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        self.critic.eval()

    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.test:
            # load data for critic
            model_critic=get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic':get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    # inference for training
    def start_training(self,tb_logger):
        opts=self.opts
        
        self.vector_env=SubprocVectorEnv if opts.is_linux else DummyVectorEnv
        
        # construct the dataset
        if self.opts.dataset=='cec':
            set_seed(42)
            # train_set,train_pro_id,test_set,test_pro_id=generate_dataset(10,20,8)
            train_set,train_pro_id=get_train_set_cec21(self.opts)
        elif self.opts.dataset=='coco':
            train_set,train_pro_id,test_set,test_pro_id=construct_problem_set(self.opts)
        elif self.opts.dataset=='metaes':
            train_set,train_pro_id=get_train_set(self.opts)
            # test_set,test_pro_id=get_test_set(self.opts)
        
        # construct parallel environment
        # learning_env
        learning_env_list=[lambda e=train_set[0]: L2E_env(dim=opts.dim,ps=opts.population_size,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,boarder_method=opts.boarder_method) for i in range(opts.batch_size)]
        learning_env=self.vector_env(learning_env_list)
        # teacher_env
        if self.opts.teacher=='madde':
            teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): MadDE(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes) for i in range(opts.batch_size)]
            
        elif self.opts.teacher=='cmaes':
            teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): sep_CMA_ES(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,sigma=opts.cmaes_sigma) for i in range(opts.batch_size)]
        elif self.opts.teacher=='glpso':
            teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): GL_PSO(dim=opts.dim,max_fes=opts.tea_fes,problem=e,min_x=opts.min_x,max_x=opts.max_x,NP=opts.population_size) for i in range(opts.batch_size)]
        elif self.opts.teacher=='pso':
            teacher_env_list=[lambda e=None: PSO(ps=opts.population_size,dim=opts.dim,max_fes=opts.max_fes,min_x=opts.min_x,max_x=opts.max_x,pho=0.2) for i in range(opts.batch_size)]
        elif self.opts.teacher=='de':
            teacher_env_list=[lambda e=None: DE(dim=opts.dim,ps=opts.population_size,min_x=opts.min_x,max_x=opts.max_x,max_fes=opts.max_fes) for i in range(opts.batch_size)]
        elif self.opts.teacher=='srpso':
            teacher_env_list=[lambda e=None: SRPSO(dim=opts.dim,ps=opts.population_size,min_x=opts.min_x,max_x=opts.max_x,max_fes=opts.max_fes,max_velocity=0.1*(opts.max_x-opts.min_x)) for i in range(opts.batch_size)]
        else:
            assert True, 'this teacher is currently not supported!!'
        teacher_env=self.vector_env(teacher_env_list)
        # random_env
        random_env_list=[lambda e=train_set[0]: L2E_env(dim=opts.dim,ps=opts.population_size,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,boarder_method=opts.boarder_method) for i in range(opts.batch_size)]
        random_env=self.vector_env(random_env_list)
        
        # get surrogate gbest and init cost
        # no matter which train mode, sgbest need to be calculated
        self.surrogate_gbest=get_surrogate_gbest(teacher_env,train_set,train_pro_id,opts.batch_size,seed=999,fes=opts.max_fes)
        # print(f's_gbest:{self.surrogate_gbest}')
        # surrogate_init_cost=get_init_cost(learning_env,train_set,opts.batch_size,seed=999)

        # self.surrogate_gbest=np.random.rand(len(train_set))
        # surrogate_init_cost=np.random.rand(len(train_set))

        # test_set sgbest
        # test_set_s_gbest=get_surrogate_gbest(teacher_env,test_set,opts.batch_size,seed=999,skip_step=opts.skip_step)
        # test_set_s_gbest=np.random.rand(len(test_set))

        # if opts.train_mode in ['3','5','6','7']:
        #     surrogate_gbest=get_surrogate_gbest(teacher_env,train_set,opts.batch_size,seed=999,skip_step=opts.skip_step)
        #     surrogate_init_cost=get_init_cost(learning_env,train_set,opts.batch_size,seed=999)
        #     # surrogate_gbest=np.random.rand(len(train_set))
        #     # surrogate_init_cost=np.random.rand(len(train_set))
        # else:
            # surrogate_gbest=np.random.rand(len(train_set))
            # surrogate_init_cost=np.random.rand(len(train_set))

        # teacher_env.close()

        # if self.opts.teacher=='madde':
        #     teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): MadDE(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.tea_fes) for i in range(opts.batch_size)]
            
        # elif self.opts.teacher=='cmaes':
        #     teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): sep_CMA_ES(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.tea_fes,sigma=opts.cmaes_sigma) for i in range(opts.batch_size)]
        # elif self.opts.teacher=='glpso':
        #     teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): GL_PSO(dim=opts.dim,max_fes=2*opts.tea_fes,problem=e) for i in range(opts.batch_size)]
        
        # teacher_env=self.vector_env(teacher_env_list)

        task=TaskForTrain(learning_env,teacher_env,random_env,opts.batch_size,opts)
        tokenizer=MyTokenizer()

        # for epoch in range(100):
        #     test_ratio=rollout(opts,self,epoch,tb_logger,tokenizer,testing=True)
        update_step=0

        outperform_ratio_list=[]
        test_ratio_list=[]

        epoch_len=18
        # epoch_len=1

        # begin training 
        for epoch in range(opts.epoch_start,opts.epoch_end):
            self.lr_scheduler.step(epoch)
            # shuffle
            # set_seed()
            # if self.opts.dataset=='cec':
            #     rand_index=np.random.choice(len(train_set),size=(len(train_set),),replace=False)
            #     train_set=train_set[rand_index]
            #     train_pro_id=train_pro_id[rand_index]
            #     surrogate_gbest=surrogate_gbest[rand_index]
            #     surrogate_init_cost=surrogate_init_cost[rand_index]
            # elif self.opts.dataset=='coco':
            #     train_set.shuffle()
            #     train_pro_id=train_pro_id[train_set.index]
            #     surrogate_gbest=surrogate_gbest[train_set.index]
            #     surrogate_init_cost=surrogate_init_cost[train_set.index]

            self.set_training()
            # logging
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with RNN lr={:.3e} for run {}".format(self.optimizer.param_groups[0]['lr'], opts.run_name) , flush=True)
            # start training
            epoch_step=epoch_len * (opts.max_fes // opts.population_size // opts.skip_step // opts.n_step) * opts.k_epoch
            pbar = tqdm(total = epoch_step,
                        disable = opts.no_progress_bar, desc = 'training',
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            
            epoch_outperform_time=0

            total_gap=0
            
            # collect_dict={}

            
            # for one batch
            # for bat_id,batch_pro in enumerate(train_set):
            for b in range(epoch_len):
                batch_step,bat_outperform_time,bat_gap,data_memory=self.train_batch(task,update_step,tokenizer,pbar,epoch,tb_logger)
                
                # update surrogate gbest
                # min_cost=min(np.min(data_memory.teacher_cost[-1]),np.min(data_memory.stu_cost[-1]))
                # if min_cost<surrogate_gbest[bat_id]:
                #     surrogate_gbest[bat_id]=min_cost
                
                # for recording table
                # collect_dict[batch_pro.__class__.__name__]={}
                # collect_dict[batch_pro.__class__.__name__]['teacher']={}
                # collect_dict[batch_pro.__class__.__name__]['random_model']={}
                # collect_dict[batch_pro.__class__.__name__]['student']={}
                # collect_dict[batch_pro.__class__.__name__]['teacher']['mean']=np.mean(data_memory.teacher_cost[-1])
                # collect_dict[batch_pro.__class__.__name__]['teacher']['std']=np.std(data_memory.teacher_cost[-1])
                # collect_dict[batch_pro.__class__.__name__]['random_model']['mean']=np.mean(data_memory.baseline_cost[-1])
                # collect_dict[batch_pro.__class__.__name__]['random_model']['std']=np.std(data_memory.baseline_cost[-1])
                # collect_dict[batch_pro.__class__.__name__]['student']['mean']=np.mean(data_memory.stu_cost[-1])   
                # collect_dict[batch_pro.__class__.__name__]['student']['std']=np.std(data_memory.stu_cost[-1])   

                data_memory.clear()

                update_step+=batch_step
                total_gap+=bat_gap
                epoch_outperform_time+=bat_outperform_time

            # gen table
            # path=os.path.join(self.opts.data_saving_dir,f'epoch_{epoch}','train')
            # gen_overall_tab(collect_dict,path)
            
            avg_gap=total_gap/epoch_len
            pbar.close()
            epoch_outperform_ratio=epoch_outperform_time/(opts.train_set_num * (opts.max_fes // opts.population_size // opts.skip_step)*opts.batch_size )
            outperform_ratio_list.append(epoch_outperform_ratio)
            
            # logging
            tb_logger.add_scalar('performance/train_outperform_ratio',epoch_outperform_ratio,epoch)

            # save model
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                                       epoch == opts.epoch_end - 1): self.save(epoch)
            
            # rollout
            # rollout
            if epoch%2==0:
                test_ratio=rollout(opts,self,epoch,tb_logger,tokenizer)
                test_ratio_list.append(test_ratio)
                if epoch == opts.epoch_start:
                    best_test_ratio=test_ratio
                else:
                    if best_test_ratio<test_ratio:
                        best_test_ratio=test_ratio
                        best_test_epoch=epoch
                        
            if epoch == opts.epoch_start:
                best_ratio=epoch_outperform_ratio
                
                best_train_epoch=opts.epoch_start
                best_test_epoch=opts.epoch_start
                best_avg_gap=avg_gap
                best_gap_epoch=opts.epoch_start
            else:
                if best_ratio<epoch_outperform_ratio:
                    best_ratio=epoch_outperform_ratio
                    best_train_epoch=epoch
                if best_avg_gap>avg_gap:
                    best_avg_gap=avg_gap
                    best_gap_epoch=epoch
            
            # log to screen
            # print(f'train_ratio_list:{outperform_ratio_list}')
            print(f'best_train_epoch:{best_train_epoch}')
            print(f'best_train_ratio:{best_ratio}')
            # print(f'test_ratio_list:{test_ratio_list}')
            print(f'best_test_epoch:{best_test_epoch}')
            print(f'best_test_ratio:{best_test_ratio}')
            print(f'current_avg_gap:{avg_gap}')
            print(f'best_avg_gap:{best_avg_gap}')
            print(f'best_gap_epoch:{best_gap_epoch}')
            log_weight(tb_logger,net=self.actor,epoch=epoch)

        # close the parallel vector_env
        learning_env.close()
        teacher_env.close()
        random_env.close()


    # for one batch training
    def train_batch(self,task:TaskForTrain,pre_step,tokenizer,pbar,epoch,tb_logger=None):
        
        max_step=self.opts.max_fes//(self.opts.population_size*self.opts.skip_step)
        data_memory=Data_Memory()
        memory=Memory()
        outperform_time=0

        # reset
        set_seed()
        

        # sample task
        if self.opts.dataset=='coco':
            instances,ids=sample_batch_task(self.opts)
            
        elif self.opts.dataset=='cec':
            instances,ids=sample_batch_task_cec21(self.opts)
            # print(f'ids:{ids}')
            
        tea_pop,stu_population=task.reset(instances)

        baseline_pop=copy.deepcopy(stu_population)

        pop_feature=task.state(stu_population)
        pop_feature=torch.FloatTensor(pop_feature).to(self.opts.device)

        # record the pre_population
        pre_stu_pop=stu_population
        pre_baseline_pop=stu_population

        # record infomation
        data_memory.teacher_cost.append([p.gbest_cost for p in tea_pop])
        data_memory.stu_cost.append([p.gbest_cost for p in stu_population])
        data_memory.baseline_cost.append([p.gbest_cost for p in stu_population])

        gamma = self.opts.gamma
        eps_clip = self.opts.eps_clip
        n_step = self.opts.n_step
        k_epoch=self.opts.k_epoch

        t=0
        total_gap=0
        # for logging
        current_step=pre_step
        
        # record init cost
        init_cost=[p.gworst_cost for p in tea_pop]

        # bat_step=self.opts.max_fes//self.opts.population_size//self.opts.skip_step
        is_done=False
        while not is_done:
            t_s=t

            bl_val_detached = []
            bl_val = []

            while t-t_s < n_step:
                # get feature
                memory.states.append(pop_feature)

                # using model to generate expr
                if self.opts.require_baseline:
                    seq,const_seq,log_prob,rand_seq,rand_c_seq,action_dict=self.actor(pop_feature,save_data=True)
                else:
                    seq,const_seq,log_prob,action_dict=self.actor(pop_feature,save_data=True)
                
                
                # next_pop为使用expr更新的population，target_pop为teacher
                target_pop,next_pop,baseline_pop,expr,is_done=task.step(stu_population,self.opts.skip_step,seq,const_seq,tokenizer,rand_seq,rand_c_seq,baseline_pop)
                
                # get reward
                # total_reward,gap_reward,base_reward,gap=task.reward(next_pop,target_pop,self.opts.reward_func,self.opts.b_reward_func,max_step,s_init_cost,s_gbest)
                # ! change: s_init_cost 变成了teacher init pop的gworst
                total_reward,gap_reward,base_reward,gap=task.reward(learning_population=next_pop,target_population=target_pop,reward_method=self.opts.reward_func,
                                                                    base_reward_method=self.opts.b_reward_func,max_step=max_step,epoch=epoch,s_init_cost=init_cost,
                                                                    s_gbest=self.surrogate_gbest,pre_learning_population=pre_stu_pop,ids=ids)
                gap_reward=torch.FloatTensor(gap_reward).to(self.opts.device)
                base_reward=torch.FloatTensor(base_reward).to(self.opts.device)
                total_reward=torch.FloatTensor(total_reward).to(self.opts.device)

                if torch.any(torch.isnan(total_reward)):
                    print(f'gap_reward:{gap_reward},base_reward:{base_reward},total_reward:{total_reward}')
                    assert True, 'nan in reward!!'

                memory.gap_rewards.append(gap_reward)
                memory.b_rewards.append(base_reward)

                
                total_gap+=np.mean(gap)

                # critic network
                baseline_val_detached,baseline_val=self.critic(pop_feature)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # store reward for ppo
                memory.actions.append(action_dict)
                memory.logprobs.append(log_prob)
                memory.rewards.append(total_reward)

                # store data
                data_memory.teacher_cost.append([p.gbest_cost for p in target_pop])
                data_memory.stu_cost.append([p.gbest_cost for p in next_pop])
                data_memory.gap.append(gap)
                data_memory.baseline_cost.append([p.gbest_cost for p in baseline_pop])
                data_memory.expr.append(expr)

                # total_baseline_reward,baseline_reward,baseline_base_reward,baseline_gap=task.reward(baseline_pop,target_pop,self.opts.reward_func,self.opts.b_reward_func,max_step,s_init_cost,s_gbest)
                total_baseline_reward,baseline_reward,baseline_base_reward,baseline_gap=task.reward(learning_population=baseline_pop,target_population=target_pop,reward_method=self.opts.reward_func,
                                                                    base_reward_method=self.opts.b_reward_func,max_step=max_step,epoch=epoch,s_init_cost=init_cost,
                                                                    s_gbest=self.surrogate_gbest,pre_learning_population=pre_baseline_pop,ids=ids)
                # record outperform ratio from aspect of reward
                outperform_time+=np.sum(total_reward.cpu().numpy()>total_baseline_reward)

               

                # next step 
                stu_population=next_pop
                pre_stu_pop=next_pop
                pre_baseline_pop=baseline_pop

                t=t+1

                # 定位到下一个state
                pop_feature = task.state(stu_population)
                pop_feature=torch.FloatTensor(pop_feature).to(self.opts.device)

                
                if is_done:
                    # update surrogate gbest
                    for i,id in enumerate(ids):
                        min_cost=min(data_memory.teacher_cost[-1][i],data_memory.stu_cost[-1][i])
                        self.surrogate_gbest[id]=min(self.surrogate_gbest[id],min_cost)
                    break

            t_time=t-t_s
            

            # begin update
            old_actions = memory.actions
            old_states = torch.stack(memory.states).detach() #.view(t_time, bs, ps, dim_f)
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            old_value = None
            for _k in range(k_epoch):
                
                if _k == 0:
                    logprobs = memory.logprobs
                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):

                        # get new action_prob
                        log_p = self.actor(old_states[tt],fix_action = old_actions[tt])

                        logprobs.append(log_p)
                        
                        baseline_val_detached, baseline_val = self.critic(old_states[tt])

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)
                logprobs = torch.stack(logprobs).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
               
                R = self.critic(pop_feature)[0]
                critic_output=R.clone()
                for r in range(len(reward_reversed)):
                    R = R * gamma + reward_reversed[r]
                    Reward.append(R)
                # clip the target:
                Reward = torch.stack(Reward[::-1], 0)
                Reward = Reward.view(-1)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()
                # calculate loss
                loss = baseline_loss + reinforce_loss
                
                # see if loss is nan
                if torch.isnan(loss):
                    print(f'baseline_loss:{baseline_loss}')
                    print(f'reinforce_loss:{reinforce_loss}')
                    assert True, 'nan found in loss!!'

                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradient norm and get (clipped) gradient norms for logging if needed
                grad_norms = clip_grad_norms(self.optimizer.param_groups)[0]

                # perform gradient descent
                self.optimizer.step()

                current_step+=1
                

                # logging to tensorboard
                if (not self.opts.no_tb) and (tb_logger is not None):
                    if current_step % self.opts.log_step == 0:
                        log_to_tb_train(tb_logger,self,Reward,grad_norms, memory.rewards,memory.gap_rewards,memory.b_rewards,gap,reinforce_loss,baseline_loss,log_prob,self.opts.show_figs,current_step)
                pbar.update(1)
            memory.clear_memory()

            
        # save data
        # path=os.path.join(self.opts.data_saving_dir,f'epoch_{epoch}','train')
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # np.save(os.path.join(path,f'batch_{bat_id}_pro{batch_pro_id}'),data_memory)

        
        # data_memory.clear()

        # return batch step
        return current_step-pre_step,outperform_time,total_gap/t,data_memory

