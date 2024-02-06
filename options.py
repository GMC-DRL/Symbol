import os
import time
import argparse
import torch


def get_options(args=None):

    parser = argparse.ArgumentParser(description="SYMBOL")
    
    # core setting
    parser.add_argument('--train',default=None,action='store_true',help='switch to train mode')
    parser.add_argument('--test',default=None,action='store_true', help='switch to inference mode')
    parser.add_argument('--run_name',default='test',help='name to identify the run')
    parser.add_argument('--load_path', default = None, help='path to load model parameters and optimizer state from')
    parser.add_argument('--resume_path', default = None, help='resume training from previous checkpoint file')
    
    
    # several train_mode
    # 1: imitation 2: pure optimization 3: imitation + breward 4: pre: imitation post: pure optimization 5: only breward 6: section imitation+breward 
    # 7: lamda imitation + breward 8: imitation + optimization 9 if else imitation + optimization 10 lamda 10: epoch lamda imitation + breward
    parser.add_argument('--train_mode', default='3',choices=['1','2','3','4','5','6','7','8','9','10'], help='different training modes')
    parser.add_argument('--init_pop',default='uniform',choices=['best','harf','random','uniform'],help='how the init population is formed from teacher')
    parser.add_argument('--lr_mode',default='big',choices=['big','small','decay','medium'], help='learning rate configuration')
    parser.add_argument('--fea_mode',default='full',choices=['full','no_fit','no_dis','no_opt','only_dis','only_opt','only_fit','xy'], help='feature selection')
    parser.add_argument('--tea_step',default='step',choices=['step','fes'], help='alignment mode of teacher and student optimizers')

    parser.add_argument('--gap_mode',default='after',choices=['before','after'])

    # teacher optimizer
    parser.add_argument('--teacher',default='madde',choices=['madde','cmaes','pso','de'])


    # environment settings
    parser.add_argument('--population_size', type = int, default= 100,help='population size use in backbone algorithm')  # recommend 100
    
    parser.add_argument('--dim', type=int, default=10,help='dimension of the sovling problems')
    parser.add_argument('--max_x',type=float,default=100,help='the upper bound of the searching range')
    parser.add_argument('--min_x',type=float,default=-100,help='the lower bound of the searching range')
    parser.add_argument('--boarder_method',default='clipping',choices=['clipping','random','periodic','reflect'], help='boarding methods')
    parser.add_argument('--skip_step',default=5,type=int,help='apply the update function every skip_step step of updating')
    parser.add_argument('--max_fes',type=int,default=50000,help='max function evaluation times')

    parser.add_argument('--reward_func',default='gap_near',choices=['w','gap_near'],help='several dist functions for comparison')
    parser.add_argument('--b_reward_func', default='5', choices=['1','2','3','4','2div2','5','6','7','8','9','10'], help='different baseline reward selections')
    
    parser.add_argument('--fea_dim',type=int,default=9,help='dim of feature encoding( excluding those historical information)')
    

    # expr setting
    parser.add_argument('--max_c',default=1.,help='max const output by the model')
    parser.add_argument('--min_c',default=-1.,help='min const output by the model')
    parser.add_argument('--c_interval',default=0.1,help='interval of consts')
    parser.add_argument('--max_layer',type=int,default=5,help='max num of layer of synax tree')

    parser.add_argument('--value_dim',default=1,type=int,help='output dim in critic net')
    parser.add_argument('--lr_critic',type=float,default=1e-3,help='learning rate in critic net')
    # parameter in RNN
    parser.add_argument('--hidden_dim',default=16,help='hidden dim in RNN layer')
    parser.add_argument('--num_layers',type=int,default=1,help='num of layers used in RNN')
    parser.add_argument('--lr',default=1e-3,help='learning rate')

    # regular settings
    parser.add_argument('--no_cuda', action='store_true', help='disable GPUs')
    parser.add_argument('--no_tb', action='store_true', help='disable Tensorboard logging')
    parser.add_argument('--no_saving', action='store_true', help='disable saving checkpoints')
    parser.add_argument('--seed', type=int, default=1024, help='random seed to use')

    # no need to config
    parser.add_argument('--is_linux',default=False,help='for the usage of parallel environment, os should be known by program')     
    parser.add_argument('--require_baseline',type=bool,default=True,help='whether to record the baseline data during training, baseline method is initial model without training')


    # Net parameters
    parser.add_argument('--v_range', type=float, default=6., help='to control the entropy')
    parser.add_argument('--encoder_head_num', type=int, default=4, help='head number of encoder')
    parser.add_argument('--decoder_head_num', type=int, default=4, help='head number of decoder')
    parser.add_argument('--critic_head_num', type=int, default=4, help='head number of critic encoder')
    parser.add_argument('--embedding_dim', type=int, default=16, help='dimension of input embeddings') # 
    # parser.add_argument('--hidden_dim', type=int, default=16, help='dimension of hidden layers in Enc/Dec') # 减小
    parser.add_argument('--n_encode_layers', type=int, default=1, help='number of stacked layers in the encoder') # 减小一点
    parser.add_argument('--normalization', default='layer', help="normalization type, 'layer' (default) or 'batch'")
    parser.add_argument('--node_dim',default=9,type=int,help='feature dimension for backbone algorithm')
    parser.add_argument('--hidden_dim1_critic',default=32,help='the first hidden layer dimension for critic')
    parser.add_argument('--hidden_dim2_critic',default=16,help='the second hidden layer dimension for critic')
    parser.add_argument('--hidden_dim1_actor',default=32,help='the first hidden layer dimension for actor')
    parser.add_argument('--hidden_dim2_actor',default=8,help='the first hidden layer dimension for actor')
    parser.add_argument('--output_dim_actor',default=1,help='output action dimension for actor')
    parser.add_argument('--lr_decay', type=float, default=0.9862327, help='learning rate decay per epoch',choices=[0.998614661,0.9862327])
    parser.add_argument('--max_sigma',default=0.7,type=float,help='upper bound for actor output sigma')
    parser.add_argument('--min_sigma',default=0.01,type=float,help='lowwer bound for actor output sigma')

    # Training parameters
    parser.add_argument('--max_learning_step',default=4000000,help='the maximum learning step for training')
    parser.add_argument('--RL_agent', default='ppo', choices = ['ppo'], help='RL Training algorithm')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor for future rewards')
    parser.add_argument('--K_epochs', type=int, default=3, help='mini PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='PPO clip ratio')
    parser.add_argument('--T_train', type=int, default=1800, help='number of itrations for training')
    parser.add_argument('--n_step', type=int, default=10, help='n_step for return estimation')
    parser.add_argument('--k_epoch',type=int,default=3,help='k_epoch in ppo alg')
    parser.add_argument('--batch_size', type=int, default=32,help='number of instances per batch during training')
    parser.add_argument('--epoch_start', type=int, default=0, help='start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--epoch_end', type=int, default=100, help='maximum training epoch')
    parser.add_argument('--epoch_size', type=int, default=1024, help='number of instances per epoch during training')
    parser.add_argument('--lr_model', type=float, default=1e-3, help="learning rate for the actor network")
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='maximum L2 norm for gradient clipping')
    
    # validate parameters
    parser.add_argument('--update_best_model_epochs',type=int,default=3,help='update the best model every n epoch')
    parser.add_argument('--val_size', type=int, default=128, help='number of instances for validation/inference')
    parser.add_argument('--per_eval_time',type=int,default=10,help='per problem eval n time')

    # logs/output settings
    parser.add_argument('--log_dir', default='logs', help='directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=50, help='log info every log_step gradient steps')
    parser.add_argument('--output_dir', default='outputs', help='directory to write output models to')
    parser.add_argument('--data_saving_dir',default='output_data',help='director to save the origin output data')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='save checkpoint every n epochs (default 1), 0 to save no checkpoints')

    opts = parser.parse_args(args)
    
    opts.world_size = 1
    opts.distributed = False
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'


    
    if opts.teacher=='madde':
        opts.tea_fes=opts.max_fes/(opts.population_size//50)
    elif opts.teacher=='glpso':
        opts.tea_fes=2*opts.max_fes
    elif opts.teacher == 'cmaes'  or opts.teacher == 'de' or opts.teacher=='srpso':
        opts.tea_fes=opts.max_fes
    

    # figure out feature dim
    if opts.fea_mode=='full':
        opts.fea_dim=opts.fea_dim
    elif opts.fea_mode[0]=='n':
        opts.fea_dim=opts.fea_dim-3
    elif opts.fea_mode[0]=='o':
        opts.fea_dim=opts.fea_dim-6
    elif opts.fea_mode=='xy':
        opts.fea_dim=opts.dim+1

    # figure out lr mode
    if opts.lr_mode=='small':
        opts.lr=1e-4
        opts.lr_critic=1e-4
        opts.lr_decay=1.
    elif opts.lr_mode=='medium':
        opts.lr=5e-4
        opts.lr_critic=5e-4
        opts.lr_decay=1.
    elif opts.lr_mode=='big':
        opts.lr=1e-3
        opts.lr_critic=1e-3
        opts.lr_decay=1.
    elif opts.lr_mode=='decay':
        opts.lr=1e-3
        opts.lr_critic=1e-3
        opts.lr_decay=pow((1e-4/1e-3),1/(opts.epoch_end))
    
    # processing settings
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name,time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume_path else opts.resume_path.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}D".format(opts.dim),
        opts.run_name
    ) if not opts.no_saving else None
    opts.data_saving_dir=os.path.join(
        opts.data_saving_dir,
        f'{opts.dim}D',
        opts.run_name
    ) if not opts.no_saving else None

    return opts



