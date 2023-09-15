import torch
import platform
import pprint
from utils.utils import set_seed
import os
import json
from tensorboardX import SummaryWriter
import warnings
from model.rnn import rnn
from model.lstm import LSTM
from model.rnn_new import rnn_new
from execute.rollout import rollout
from execute.train import trainer
from expr.tokenizer import MyTokenizer
from dataset.generate_dataset import generate_dataset,construct_problem_set
from options import get_options

def load_agent(model_name):
    agent={
        'rnn':rnn,
        'lstm':LSTM,
        'rnn_new':rnn_new,
    }.get(model_name,None)
    assert agent is not None, "Currently unsupported agent: {}!".format(model_name)
    return agent

def run(opts):
    # only one mode can be specified in one time, test or train
    assert opts.train==None or opts.test==None, 'Between train&test, only one mode can be given in one time'
    
    sys=platform.system()
    opts.is_linux=True if sys == 'Linux' else False

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tb:
        tb_logger = SummaryWriter(os.path.join(opts.log_dir,opts.model, "{}D".format(opts.dim), opts.run_name))

    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)
    
    # Set the device, you can change it according to your actual situation
    opts.device = torch.device("cuda:1" if opts.use_cuda else "cpu")

    # Set the random seed to initialize the network
    set_seed(opts.seed)
    
    # init agent
    model=load_agent(opts.model)(opts,tokenizer=MyTokenizer())
    
    # opts.load_path=r'/home/chenjiacheng/L2E/L2E_0417/outputs/RNN/10D/0419_test_20230419T010744/epoch-28.pt'
    # Load data from load_path(if provided)
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        runner=trainer(model,opts)
        runner.load(load_path)
    
    # test only
    if opts.test:
        from env import SubprocVectorEnv,DummyVectorEnv
        # init task
        set_seed()
        runner.vector_env=SubprocVectorEnv if opts.is_linux else DummyVectorEnv
        
        # construct the dataset
        # if opts.dataset=='cec':
        #     set_seed(42)
        #     train_set,train_pro_id,test_set,test_pro_id=generate_dataset(10,20,8)
        # elif opts.dataset=='coco':
        #     train_set,train_pro_id,test_set,test_pro_id=construct_problem_set(opts)
            
        print(f'run_name:{opts.run_name}')

        rollout(opts,runner,-1,tb_logger,MyTokenizer(),testing=True)
    else:
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            opts.epoch_start = epoch_resume + 1
            runner.start_training(tb_logger)
        else:
            # training
            runner=trainer(model,opts)
            runner.start_training(tb_logger)
        
    if not opts.no_tb:
        tb_logger.close()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run(get_options())
    