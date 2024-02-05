import torch
import math
import time
import numpy as np
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import copy


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) or isinstance(model, DDP) else model

def set_seed(seed=None):
    if seed is None:
        seed=int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def move_to_cuda(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.cuda(device)

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def get_init_cost(env,dataset,bs,seed):
    print('getting init cost...')
    init_costs=[]
    for pro in dataset:
        action=[{'problem':copy.deepcopy(pro),'sgbest':0} for i in range(bs)]
        env.step(action)
        pop=env.reset()
        init_cost_list=[p.gworst_cost for p in pop]
        init_costs.append(np.max(init_cost_list))
    print('done...')
    return np.array(init_costs)

def get_surrogate_gbest(env,dataset,ids,bs,seed,fes):
    print('getting surrogate gbest...')
    gbests={}
    set_seed(seed)
    for id,pro in zip(ids,dataset):
        action=[{'problem':copy.deepcopy(pro),'sgbest':0} for i in range(bs)]
        env.step(action)
        env.reset()
        is_done=False
        while not is_done:
            action=[{'fes':fes} for i in range(bs)]
            pop,_,is_done,_=env.step(action)
            is_done=is_done.all()
        gbest_list=[p.gbest_cost for p in pop]
        gbests[id]=np.min(gbest_list)
    print('done...')
    return gbests