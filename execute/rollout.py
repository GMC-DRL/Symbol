import numpy as np
import torch
from utils.utils import set_seed
from utils.logger import log_to_test_with_teacher,log_to_test_without_teacher,gen_overall_tab
from .task import TaskForTrain
from pbo_env import L2E_env,MadDE,sep_CMA_ES,PSO,DE
import os
from tqdm import tqdm
import copy
from dataset.generate_dataset import sample_batch_task_id_cec21

# inference for rollout
# todo: fix

class Memory():
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


def rollout(opts,agent,epoch,tb_logger,tokenizer,testing=False):
    # switch model's state to eval
    agent.set_evaling()
    
    need_log=False
    if epoch % 5 == 0 or testing:
        need_log=True
    test_outperform_time=0
    
    # for saving table
    collect_dict={}
    
    test_id=range(1,11)
    with tqdm(range(len(test_id)),desc='rollout') as pbar:
        # learning_env
        learning_env_list=[lambda e=None: L2E_env(dim=opts.dim,ps=opts.population_size,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,boarder_method=opts.boarder_method) for i in range(opts.batch_size)]
        learning_env=agent.vector_env(learning_env_list)
        # teacher_env
        if opts.teacher=='madde':
            teacher_env_list=[lambda e=None: MadDE(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes) for i in range(opts.batch_size)]
        elif opts.teacher=='cmaes':
            teacher_env_list=[lambda e=None: sep_CMA_ES(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,sigma=opts.cmaes_sigma) for i in range(opts.batch_size)]
        elif opts.teacher=='pso':
            teacher_env_list=[lambda e=None: PSO(ps=opts.population_size,dim=opts.dim,max_fes=opts.max_fes,min_x=opts.min_x,max_x=opts.max_x,pho=0.2) for i in range(opts.batch_size)]
        elif opts.teacher=='de':
            teacher_env_list=[lambda e=None: DE(dim=opts.dim,ps=opts.population_size,min_x=opts.min_x,max_x=opts.max_x,max_fes=opts.max_fes) for i in range(opts.batch_size)]
        else:
            assert True, 'this teacher is currently not supported!!'
        teacher_env=agent.vector_env(teacher_env_list)

        # random_env
        random_env_list=[lambda e=None: L2E_env(dim=opts.dim,ps=opts.population_size,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,boarder_method=opts.boarder_method) for i in range(opts.batch_size)]
        random_env=agent.vector_env(random_env_list)

        task=TaskForTrain(learning_env,teacher_env,random_env,opts.batch_size,opts)
        
        
        for bat_id,id in enumerate(test_id):
            # generate batch instances for testing
            instances,p_name=sample_batch_task_id_cec21(dim=opts.dim,batch_size=opts.batch_size,problem_id=id,seed=999)

                
            
            memory_imitate,bat_outperform_time,cost_stu,cost_base,cost_tea=rollout_imitate(opts,task,agent,tokenizer,instances,testing)
            
            test_outperform_time+=bat_outperform_time

            # for recording table
            collect_dict[f'F{id}']={}
            collect_dict[f'F{id}']['teacher']={}
            collect_dict[f'F{id}']['random_model']={}
            collect_dict[f'F{id}']['student']={}
            collect_dict[f'F{id}']['teacher']['mean']=cost_tea
            collect_dict[f'F{id}']['teacher']['std']=np.std(memory_imitate.teacher_cost[-1])
            collect_dict[f'F{id}']['random_model']['mean']=cost_base
            collect_dict[f'F{id}']['random_model']['std']=np.std(memory_imitate.baseline_cost[-1])
            collect_dict[f'F{id}']['student']['mean']=cost_stu
            collect_dict[f'F{id}']['student']['std']=np.std(memory_imitate.stu_cost[-1])

            # log to tensorboard about the cost
            if not opts.no_tb:
                tb_logger.add_scalars(f'performance/cost/{p_name}',{'student':cost_stu,'baseline':cost_base,'teacher':cost_tea},epoch)

            # save data
            path=os.path.join(opts.data_saving_dir,f'epoch_{epoch}','test')
            if not os.path.exists(path):
                os.makedirs(path)

            # only one memory to store
            np.save(os.path.join(path,f'batch_{bat_id}_{id}'),memory_imitate)

            
            # log opt figures
            if need_log:
                log_to_test_with_teacher(memory_imitate.teacher_cost,memory_imitate.baseline_cost,memory_imitate.stu_cost,epoch,bat_id,id,tb_logger.file_writer.get_logdir(),logged=True)
                # log_to_test_without_teacher(tb_logger,memory_imitate.teacher_cost,memory_no_teacher.stu_cost,epoch,bat_id,problem_id[bat_id])
            memory_imitate.clear()
            pbar_info={'batch_id':bat_id,'stu_gbest':cost_stu,'tea_gbest':cost_tea}


            # print(pbar_info)
            pbar.set_postfix(pbar_info)
            pbar.update(1)
            if testing:
                print(f'problem:{p_name},teacher:{cost_tea},student:{cost_stu},baseline:{cost_base}')
            # memory_no_teacher.clear()
        pbar.close()
        # close the parallel vector_env
        learning_env.close()
        teacher_env.close()
        random_env.close()
    test_outperform_ratio=test_outperform_time/(len(test_id)*opts.batch_size)
    # logging 
    if not opts.no_tb:
        tb_logger.add_scalar('performance/test_outperform_ratio',test_outperform_ratio,epoch)
    
    gen_overall_tab(collect_dict,path)

    return test_outperform_ratio
    

# with teacher
def rollout_imitate(opts,task,agent,tokenizer,instances,testing):
    # print(f'cur_problem:{bat_pro.__str__()}')
    
    max_step=opts.max_fes//(opts.population_size*opts.skip_step)

    memory=Memory()
    set_seed(999)
    # reset environment

    if opts.teacher!='cmaes':
        tea_pop,stu_population=task.reset(instances,True)
    else:
        tea_pop,stu_population=task.reset(instances,False)
    
    baseline_pop=copy.deepcopy(stu_population)

    memory.teacher_cost.append([p.gbest_cost for p in tea_pop])
    memory.stu_cost.append([p.gbest_cost for p in stu_population])
    memory.baseline_cost.append([p.gbest_cost for p in baseline_pop])

    outperform_time=0
    is_done=False
    while not is_done:
        
        # get feature
        
        pop_feature = task.state(stu_population)
        pop_feature=torch.FloatTensor(pop_feature).to(opts.device)

        
        # using lstm to generate expr
        if opts.require_baseline:
            seq,const_seq,log_prob,rand_seq,rand_c_seq=agent.actor(pop_feature)
        else:
            seq,const_seq,log_prob=agent.actor(pop_feature)
        
        
        target_pop,next_pop,baseline_pop,expr,is_done=task.step(stu_population,opts.skip_step,seq,const_seq,tokenizer,rand_seq,rand_c_seq,baseline_pop,testing=True)
        
        memory.teacher_cost.append([p.gbest_cost for p in target_pop])
        memory.stu_cost.append([p.gbest_cost for p in next_pop])
        # memory.gap.append(gap)
        memory.baseline_cost.append([p.gbest_cost for p in baseline_pop])
        memory.expr.append(expr)

        if is_done:
            break
        # next pop
        stu_population=next_pop
    outperform_time+=np.sum(memory.stu_cost[-1]<memory.baseline_cost[-1])
    # print(f'test outperform time:{outperform_time}')
    
    return memory,outperform_time,np.mean(memory.stu_cost[-1]),np.mean(memory.baseline_cost[-1]),np.mean(memory.teacher_cost[-1])

