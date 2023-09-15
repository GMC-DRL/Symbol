import torch
import math
import numpy as np
from .plots import plot_grad_flow, plot_improve_pg
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd

def log_to_screen(init_value, best_value, reward, costs_history, search_history,
                  batch_size, dataset_size, T):
    # reward
    print('\n', '-'*70)
    print('Avg total reward:'.center(35), '{:<10f} +- {:<10f}'.format(
        reward.sum(1).mean(), torch.std(reward.sum(1)) / math.sqrt(batch_size)))
    print('Avg step reward:'.center(35), '{:<10f} +- {:<10f}'.format(
        reward.mean(), torch.std(reward) / math.sqrt(batch_size)))

    # cost
    print('-'*70)
    print('Avg init cost:'.center(35), '{:<10f} +- {:<10f}'.format(
        init_value.mean(), torch.std(init_value) / math.sqrt(batch_size)))
    for per in range(20,100,20):
        cost_ = costs_history[:,round(T*per/100)]
        print(f'Avg cost after {per}% steps:'.center(35), '{:<10f} +- {:<10f}'.format(
            cost_.mean(),
            torch.std(cost_) / math.sqrt(batch_size)))
    # best cost
    print('-'*70)

    for per in range(20,100,20):
        cost_ = search_history[:,round(T*per/100)]
        print(f'Avg best cost after {per}% steps:'.center(35), '{:<10f} +- {:<10f}'.format(
            cost_.mean(),
            torch.std(cost_) / math.sqrt(batch_size)))
    print('Avg final best cost:'.center(35), '{:<10f} +- {:<10f}'.format(
        best_value.mean(), torch.std(best_value) / math.sqrt(batch_size)))

    print('-'*70, '\n')

def log_to_tb_val(tb_logger, time_used, init_value, best_value, reward, costs_history, search_history,
                  batch_size, dataset_size, T, show_figs, epoch):
    if show_figs:
        tb_logger.log_images('validation/improve_pg',[plot_improve_pg(costs_history)], epoch)
        tb_logger.log_images('validation/search_pg',[plot_improve_pg(search_history)], epoch)

    #    tb_logger.add_scalar('validation/avg_time',  time_used.mean() / dataset_size, epoch)
    tb_logger.add_scalar('validation/avg_total_reward', reward.sum(1).mean(), epoch)
    tb_logger.add_scalar('validation/avg_step_reward', reward.mean(), epoch)

    tb_logger.add_scalar('validation/avg_init_cost', init_value.mean(), epoch)
    tb_logger.add_scalar('validation/avg_best_cost', best_value.mean(), epoch)

def log_to_val(tb_logger,cost_rollout,epoch_id):
    len1=cost_rollout.shape[0]
    
    # todo
    for i in range(len1):
        tb_logger.add_scalars('validation/Optimization_curve_epoch_{}'.format(epoch_id),{'RL-PSO':cost_rollout[i].item()},i)
    
    tb_logger.add_scalars('performance_overtime',{'RL-PSO':cost_rollout[-1].item()},epoch_id)

def log_to_tb_train(tb_logger, agent, Reward,  grad_norms, reward,gap_reward,b_reward,gap, reinforce_loss,baseline_loss, log_likelihood,  show_figs, mini_step):

    tb_logger.add_scalar('learnrate_pg', agent.optimizer.param_groups[0]['lr'], mini_step)
    # avg_cost = (total_cost).mean().item()
    # tb_logger.add_scalar('train/avg_cost', avg_cost, mini_step)
    
    # avg_reward = torch.stack(reward, 0).sum(0).mean().item()
    # avg_reward = torch.mean(reward).item()

    # # max_reward = torch.stack(reward, 0).max(0)[0].mean().item()
    # max_reward = torch.max(reward).item()
    # # reward
    # tb_logger.add_scalar('train/avg_reward', avg_reward, mini_step)
    # # tb_logger.add_scalar('train/init_cost', initial_cost.mean(), mini_step)
    # tb_logger.add_scalar('train/max_reward', max_reward, mini_step)
    avg_reward = torch.stack(reward, 0).sum(0).mean().item()
    
    max_reward = torch.stack(reward, 0).max(0)[0].mean().item()
    tb_logger.add_scalar('train/avg_reward', avg_reward, mini_step)
    # tb_logger.add_scalar('train/init_cost', initial_cost.mean(), mini_step)
    tb_logger.add_scalar('train/max_reward', max_reward, mini_step)
    avg_gap_reward=torch.stack(gap_reward,0).sum(0).mean().item()
    tb_logger.add_scalar('train/avg_gap_reward', avg_gap_reward, mini_step)
    avg_b_reward=torch.stack(b_reward,0).sum(0).mean().item()
    tb_logger.add_scalar('train/avg_b_reward', avg_b_reward, mini_step)

    tb_logger.add_scalar('train/target_return',Reward.mean().item(),mini_step)

    # grad_norms, grad_norms_clipped = grad_norms
    # tb_logger.add_scalar('loss/actor_loss', reinforce_loss.item(), mini_step)

    mean_gap=np.mean(gap)
    max_gap=np.max(gap)
    # gap
    tb_logger.add_scalar('train/avg_gap',mean_gap,mini_step)
    tb_logger.add_scalar('train/max_gap',max_gap,mini_step)

    # loss
    tb_logger.add_scalar('loss/critic_loss', baseline_loss.item(), mini_step)
    tb_logger.add_scalar('loss/actor_loss', reinforce_loss.item(), mini_step)
    tb_logger.add_scalar('loss/nll', -log_likelihood.mean().item(), mini_step)
    tb_logger.add_scalar('loss/total_loss', (reinforce_loss+baseline_loss).item(), mini_step)

    # gradient
    tb_logger.add_scalar('grad/actor',grad_norms[0],mini_step)
    tb_logger.add_scalar('grad/critic',grad_norms[1],mini_step)
    # tb_logger.add_scalar('train/entropy', entropy.mean().item(), mini_step)
    

    # tb_logger.add_scalar('grad/actor', grad_norms[0], mini_step)
    # tb_logger.add_scalar('grad_clipped/actor', grad_norms_clipped[0], mini_step)
    # tb_logger.add_scalar('loss/critic_loss', baseline_loss.item(), mini_step)

    # tb_logger.add_scalar('loss/total_loss', (reinforce_loss+baseline_loss).item(), mini_step)

    # tb_logger.add_scalar('grad/critic', grad_norms[1], mini_step)
    # tb_logger.add_scalar('grad_clipped/critic', grad_norms_clipped[1], mini_step)

    if show_figs and mini_step % 1000 == 0:
        # tb_logger.log_images('grad/actor',[plot_grad_flow(agent.actor)], mini_step)
        # tb_logger.log_images('grad/critic',[plot_grad_flow(agent.critic)], mini_step)
        tb_logger.log_images('grad',[plot_grad_flow(agent.model)],mini_step)

def log_weight(tb_logger,net,epoch):
    for name, layer in net.named_parameters():
        tb_logger.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)
    
# def log_to_tb_train(tb_logger, agent,  entropy, loss, mini_step):

#     tb_logger.add_scalar('learnrate_pg', agent.optimizer.param_groups[0]['lr'], mini_step)
#     tb_logger.add_scalar('loss/actor_loss', loss.item(), mini_step)
#     tb_logger.add_scalar('train/entropy', entropy.mean().item(), mini_step)


# def log_to_test_with_teacher(tb_logger,teacher_cost,baseline_cost,stu_cost,epoch,bat_id,bat_pro):
#     teacher_cost=np.stack(teacher_cost,0).mean(-1)
#     baseline_cost=np.stack(baseline_cost,0).mean(-1)
#     stu_cost=np.stack(stu_cost,0).mean(-1)
#     for i in range(len(teacher_cost)):
#         tb_logger.add_scalars(f'test_with_teacher/epoch_{epoch}_batch_{bat_id}',{'teacher':teacher_cost[i],'baseline':baseline_cost[i],'student':stu_cost[i]},i)
#     logdir = tb_logger.file_writer.get_logdir()
#     # print(logdir)
#     logdir_delete = logdir + f'/test_with_teacher/epoch_{epoch}_batch_{bat_id}'
#     writer_delete=SummaryWriter(logdir_delete)
#     # print(writer_delete.file_writer.get_logdir())
#     writer_delete.close()


# 改成存图片的形式
def log_to_test_with_teacher(teacher_cost,baseline_cost,student_cost,epoch,bat_id,bat_pro,output_dir,logged):
    
    output_dir=output_dir+'/cost_curves/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    markers = ['o', '^', '*', 'O', 'v', 'x', 'X', 'd', 'D', '.', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H']
    colors = ['b', 'g', 'orange', 'r', 'purple', 'brown', 'grey', 'limegreen', 'turquoise', 'olivedrab', 'royalblue', 'darkviolet', 
            'chocolate', 'crimson', 'teal','seagreen', 'navy', 'deeppink', 'maroon', 'goldnrod', 
            ]
    matplotlib.use('Agg')
    

    # print(len(memory.teacher_cost))
    teacher_cost=np.stack(teacher_cost,0)
    student_cost=np.stack(student_cost,0)
    baseline_cost=np.stack(baseline_cost,0)

    logged=True

    if logged:
        teacher_cost = np.log(np.maximum(teacher_cost, 1e-8))
        student_cost = np.log(np.maximum(student_cost, 1e-8))
        baseline_cost = np.log(np.maximum(baseline_cost, 1e-8))


    tea_mean=np.mean(teacher_cost,-1)
    stu_mean=np.mean(student_cost,-1)
    baseline_mean=np.mean(baseline_cost,-1)

    tea_std=np.std(teacher_cost,-1)
    stu_std=np.std(student_cost,-1)
    baseline_std=np.std(baseline_cost,-1)

    x = np.arange(teacher_cost.shape[0])

    lables=['teacher','student','baseline']
    mean=[tea_mean,stu_mean,baseline_mean]
    std=[tea_std,stu_std,baseline_std]

    index=np.linspace(0,teacher_cost.shape[0]-1,50,dtype=int)
    # print(index)

    plt.figure()
    for i in range(3):
        plt.plot(x[index], mean[i][index], label=lables[i], marker='*', markevery=8, markersize=10, c=colors[i])
        plt.fill_between(x[index], mean[i][index] - std[i][index], mean[i][index] + std[i][index], alpha=0.2, facecolor=colors[i])

    plt.grid()
    plt.xlabel('step')
    plt.legend()

    if logged:
        plt.ylabel('log Costs')
        plt.savefig(output_dir+f'epoch_{epoch}_batch_{bat_id}_pro_{bat_pro}__log_cost_curve.png',bbox_inches='tight')
        # plt.savefig(output_dir + f'learnable_{name}_log_cost_curve.png', bbox_inches='tight')
    else:
        plt.ylabel('Costs')
        plt.savefig(output_dir+f'epoch_{epoch}_batch_{bat_id}_pro_{bat_pro}_cost_curve.png',bbox_inches='tight')
        # plt.savefig(output_dir + f'learnable_{name}_cost_curve.png', bbox_inches='tight')

    # plt.show()
    plt.close()
    

def log_to_test_without_teacher(tb_logger,teacher_cost,stu_cost,epoch,bat_id,bat_pro):
    teacher_cost=np.stack(teacher_cost,0).mean(-1)
    stu_cost=np.stack(stu_cost,0).mean(-1)
    for i in range(len(teacher_cost)):
        tb_logger.add_scalars(f'test_without_teacher/epoch_{epoch}_batch_{bat_id}',{'teacher':teacher_cost[i],'student':stu_cost[i]},i)
    logdir = tb_logger.file_writer.get_logdir()
    logdir_delete = logdir + f'/test_with_teacher/epoch_{epoch}_batch_{bat_id}'
    writer_delete=SummaryWriter(logdir_delete)
    writer_delete.close()

def gen_overall_tab(results: dict, out_dir: str) -> None:
    # get multi-indexes first
    problems = []
    statics = ['mean','std']
    optimizers = []
    for problem in results.keys():
        problems.append(problem)

    # for comparing sort the problem list
    def sort_key(item):
        return int(item[1:])
    problems=sorted(problems,key=sort_key)
    # print(problems)

    for optimizer in results[problems[0]].keys():
        optimizers.append(optimizer)
    multi_columns = pd.MultiIndex.from_product(
        [problems,statics], names=('Problem', 'cost')
    )
    df_results = pd.DataFrame(np.ones(shape=(len(optimizers),len(problems)*len(statics))),
                              index=optimizers,
                              columns=multi_columns)

    

    # calculate each Obj
    for problem in problems:
        for optimizer in optimizers:
            df_results.loc[optimizer,(problem,'mean')] = np.format_float_scientific(results[problem][optimizer]['mean'], precision=3, exp_digits=1)
            df_results.loc[optimizer,(problem,'std')] = np.format_float_scientific(results[problem][optimizer]['std'], precision=3, exp_digits=1)
            
    
    df_results.to_excel(os.path.join(out_dir,'overall_table.xlsx'))