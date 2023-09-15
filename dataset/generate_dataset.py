from .cec_test_func import *
from .coco.bbob import *
from utils.utils import set_seed


import numpy as np
import csv
import random

cec2013={
    1:Sphere_13,
    2:Elliptic_13,
    3:Bent_cigar_13,
    4:Discus_13,
    5:Dif_powers_13,
    6:Rosenbrock_13,
    7:Scaffer_F7_13,
    8:Ackley_13,
    9:Weierstrass_13,
    10:Griewank_13,
    11:Rastrigin_13,
    12:Ro_Rastrigin_13,
    13:NonConti_Ro_Rastrigin_13,
    14:Schwefel_13,
    15:Ro_Schwefel_13,
    16:Katsuura_13,
    17:bi_Rastrigin_13,
    18:Ro_bi_Rastrigin_13,
    19:Grie_rosen_13,
    20:Escaffer6_13,
    21:Composition2013,
    22:Composition2013,
    23:Composition2013,
    24:Composition2013,
    25:Composition2013,
    26:Composition2013,
    27:Composition2013,
    28:Composition2013,
}

composition={
    21:{'sub_problems_id':[6,5,3,4,1],'lamda':[1,1e-6,1e-26,1e-6,0.1],'sigma':[10,20,30,40,50],'bias':[0,100,200,300,400]},
    22:{'sub_problems_id':[14,14,14],'lamda':[1,1,1],'sigma':[20,20,20],'bias':[0,100,200]},
    23:{'sub_problems_id':[15,15,15],'lamda':[1,1,1],'sigma':[20,20,20],'bias':[0,100,200]},
    24:{'sub_problems_id':{15,12,9},'lamda':[0.25,1,2.5],'sigma':[20,20,20],'bias':[0,100,200]},
    25:{'sub_problems_id':{15,12,9},'lamda':[0.25,1,2.5],'sigma':[10,30,50],'bias':[0,100,200]},
    26:{'sub_problems_id':[15,12,2,9,10],'lamda':[0.25,1,1e-7,2.5,10],'sigma':[10,10,10,10,10],'bias':[0,100,200,300,400]},
    27:{'sub_problems_id':[10,12,15,9,1],'lamda':[100,10,2.5,25,0.1],'sigma':[10,10,10,20,20],'bias':[0,100,200,300,400]},
    28:{'sub_problems_id':[19,7,15,20,1],'lamda':[2.5,2.5e-3,2.5,5e-4,0.1],'sigma':[10,20,30,40,50],'bias':[0,100,200,300,400]}
}

F=[0,-1400,-1300,-1200,-1100,-1000,-900,-800,-700,-600,-500,-400,-300,-200,-100,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400]

rf=[-1,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1]

def read_data(dim):
    csv_file = open('dataset/input_data/M_D' + str(dim) + '.txt')
    csv_data = csv.reader(csv_file, delimiter=' ')
    csv_data_not_null = [[float(data) for data in row if len(data) > 0] for row in csv_data]
    rotate_data = np.array(csv_data_not_null)
    csv_file = open('dataset/input_data/shift_data.txt')
    csv_data = csv.reader(csv_file, delimiter=' ')
    csv_data = csv.reader(csv_file, delimiter=' ')
    sd = []
    for row in csv_data:
        sd += [float(data) for data in row if len(data) > 0]
    return rotate_data,sd

def read_M(dim , m,rotate_data):
    return rotate_data[m * dim : (m + 1) * dim]

def read_O(dim , m, sd):
    return np.array(sd[m * dim : (m + 1) * dim])


def get_func_cec2013(dim,pro_id,rotate_data,shift_data):
    assert pro_id>=1 and pro_id<=28, 'Problem id for cec 2013 range from 1 to 28'
    # single problem
    if pro_id<=20:
        if rf[pro_id]==1:
            M1=read_M(dim,0,rotate_data)
            M2=read_M(dim,1,rotate_data)
        else:
            M1= np.eye(dim)
            M2= np.eye(dim)
        O=read_O(dim,0,shift_data)
        return cec2013.get(pro_id)(dim,M1,M2,O,F[pro_id])
    else:
        # composition
        config=composition.get(pro_id)
        sub_problems_id=config['sub_problems_id']
        bias=config['bias']
        sub_problems=[]
        for i,id in enumerate(sub_problems_id):
            if rf[pro_id]:
                M1=read_M(dim,i,rotate_data)
                M2=read_M(dim,i+1,rotate_data)
            else:
                M1=np.eye(dim)
                M2=np.eye(dim)
            O = read_O(dim,i,shift_data)
            sub_problems.append(cec2013.get(id)(dim,M1,M2,O,bias[i]))
        # print(f'cf_num:{len(sub_problems)}')
        return Composition2013(dim,len(sub_problems),config['lamda'],config['sigma'],config['sigma'],sub_problems,F[pro_id])

# deprecated
# # return the pro_id function
# def get_func_cec2013(pro_id,dim):
#     pro=None
#     # file_path=f'./input_data/M_D{dim}'
#     # rotate_matrix=np.loadtxt(file_path)
    
#     if pro_id==1:
#         pro=Sphere(dim=dim)
#     elif pro_id==2:
#         pro=Ellipsoidal(dim=dim)
#     elif pro_id==3:
#         pro=Bent_cigar(dim=dim)
#     elif pro_id==4:
#         pro=Discus(dim=dim)
#     elif pro_id==5:
#         pro=Dif_powers(dim=dim)
#     elif pro_id==6:
#         pro=Rosenbrock(dim=dim)
#     elif pro_id==7:
#         pro=Scaffer_F7(dim=dim)
#     elif pro_id==8:
#         pro=Ackley(dim=dim)
#     elif pro_id==9:
#         pro=Weierstrass(dim=dim)
#     elif pro_id==10:
#         pro=Grie_rosen(dim=dim)
#     elif pro_id==11:
#         pro=Rastrigin(dim=dim)
#     elif pro_id==12:
#         pro=Rastrigin(dim=dim)
#     elif pro_id==13:
#         pro=Rastrigin(dim=dim)
#     elif pro_id==14:
#         pro=Schwefel(dim=dim)
#     elif pro_id==15:
#         pro=Schwefel(dim=dim)
#     elif pro_id==16:
#         pro=Katsuura(dim=dim)
#     elif pro_id==17:
#         pro=bi_Rastrigin(dim=dim)
#     elif pro_id==18:
#         pro=bi_Rastrigin(dim=dim)
#     elif pro_id==19:
#         pro=Grie_rosen(dim=dim)
#     elif pro_id==20:
#         pro=Scaffer_F7(dim=dim)
    
#     pass

'''记得改rand_index!!!'''
def generate_dataset(dim,train_set_num=20,test_set_num=8):
    rotate_data,shift_data=read_data(dim)
    problem_list=[]
    for pro_id in range(1,29):
        problem_list.append(get_func_cec2013(dim,pro_id,rotate_data,shift_data))
        # problem_list.append(cec2013.get(pro_id)(dim,np.zeros(dim),np.eye(dim),0))
    
    assert train_set_num+test_set_num<=len(problem_list)
    rand_index=np.random.choice(len(problem_list),size=(len(problem_list),),replace=False)
    problem_list=np.array(problem_list)

    # for debuging
    # rand_index=np.arange(28)
    problem_ids=np.arange(1,29)[rand_index]

    rand_problem_list=problem_list[rand_index]
    return rand_problem_list[:train_set_num],problem_ids[:train_set_num],rand_problem_list[train_set_num:train_set_num+test_set_num],problem_ids[train_set_num:train_set_num+test_set_num]

from .cec_dataset import *

'''for augmented cec2021 dataset generation'''

pro_name=['Bent_cigar','Schwefel','bi_Rastrigin','Grie_rosen','Hybrid','Hybrid','Hybrid','Composition','Composition','Composition','Mix21']
def get_config(problem_id):
    pro=pro_name[problem_id-1]
    subproblems=None
    sublength=None
    Comp_sigma=None
    Comp_lamda=None
    indicated_dataset=None
    if problem_id==5:
        # hybrid 1
        subproblems=['Schwefel','Rastrigin','Ellipsoidal']
        sublength=[0.3,0.3,0.4]
    elif problem_id==6:
        # hybrid 2
        subproblems=['Escaffer6','Hgbat','Rosenbrock','Schwefel']
        sublength=[0.2,0.2,0.3,0.3]
    elif problem_id==7:
        # hybrid 3
        subproblems=['Escaffer6','Hgbat','Rosenbrock','Schwefel','Ellipsoidal']
        sublength=[0.1,0.2,0.2,0.2,0.3]
    elif problem_id==8:
        # # composition 1
        subproblems=['Rastrigin','Griewank','Schwefel']
        sublength=None
        Comp_sigma=[10,20,30]
        Comp_lamda=[1,10,1]
    elif problem_id==9:
        # composition 2
        subproblems=['Ackley','Ellipsoidal','Griewank','Rastrigin']
        sublength=None
        Comp_sigma=[10,20,30,40]
        Comp_lamda=[10,1e-6,10,1]
    elif problem_id==10:
        # composition 3
        subproblems=['Rastrigin','Happycat','Ackley','Discus','Rosenbrock']
        sublength=None
        Comp_sigma=[10,20,30,40,50]
        Comp_lamda=[10,1,10,1e-6,1]
    elif problem_id==11:              
        indicated_dataset=cec2021
    return (pro,subproblems,sublength,Comp_sigma,Comp_lamda,indicated_dataset)

def make_dataset(dim, max_x,min_x,trainset_num,testset_num):
    '''return a dataloader'''
    shifted=False
    rotated=False
    biased=False
    problem_list=[]
    for problem_id in range(1,11):
        pro,subproblems,sublength,Comp_sigma,Comp_lamda,indicated_dataset=get_config(problem_id)
        dataloader=Training_Dataset(filename=None, dim=dim, num_samples=1, problems=pro, biased=biased, shifted=shifted, rotated=rotated,
                                batch_size=1,xmin=min_x,xmax=max_x,indicated_specific=True,indicated_dataset=indicated_dataset,
                                problem_list=subproblems,problem_length=sublength,Comp_sigma=Comp_sigma,Comp_lamda=Comp_lamda)
        problem_list.append(dataloader[0][0])
    return problem_list[:trainset_num],problem_list[trainset_num:trainset_num+testset_num]
        
from .coco import bbob


def construct_problem_set(config):
    problem = config.problem
    if problem in ['bbob', 'bbob-noisy']:
        return bbob.BBOB_Dataset.get_datasets(suit=config.problem,
                                              dim=config.dim,
                                              upperbound=config.max_x,
                                              train_batch_size=1,
                                              test_batch_size=1,
                                              difficulty=config.difficulty)
    else:
        raise ValueError(problem + ' is not defined!')

problem_ids={
    'small':[1],
    'medium':[1,8,11,15,20],
    'large':[1,8,11,15,20,4,6,16,17,19],
    'train':[1,8,11,15,20,4,6,16,17,19],
}
# D={
#     'small':[2],
#     'medium':[1,2,3,4,5],
#     'large':[1,2,3,4,5,6,7,8,9,10],
#     'train':[2,3,4,5,6,7,8,9,10],
# }

# def sample_batch_task(config,id):

#     instances=[]
#     # dim=random.choice(D.get(config.training_set))
#     dim=config.dim
#     # f_id=random.choice(problem_ids.get(config.training_set))
#     # no noise
#     # noise=random.uniform(0,0.1)
#     for i in range(config.batch_size):
#         noise=0
#         # randomly sample offset and rotate matrix
#         # offset=random.uniform(-5,5,size=(dim,))
#         offset=0.8 * (np.random.random(dim) * (config.max_x - config.min_x) + config.min_x)
#         H = rotate_gen(dim)
#         # instance.shift=offset
#         # instance.rotate=
#         instance = eval(f'F{id}')(dim=dim, shift=offset, rotate=H, bias=0, lb=-5, ub=5)
#         instances.append(copy.deepcopy(instance))
    
#     return instances


def rotate_gen(dim):  # Generate a rotate matrix
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        mat = np.eye(dim)
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


def sample_batch_task(opts):
    dim=opts.dim
    mode=opts.training_set
    batch_size=opts.batch_size
    instances=[]
    ids=[]
    # dim=config.dim
    
    # no noise
    # noise=random.uniform(0,0.1)
    for i in range(batch_size):
        f_id=random.choice(problem_ids.get(mode))
        ids.append(f'F{f_id}')
        # dim=random.choice(D.get(mode))
        # no bias
        noise=0
        # shift
        offset=np.random.random(dim) * 10 - 5
        # rotate
        H = rotate_gen(dim)
        
        instance = eval(f'F{f_id}')(dim=dim, shift=offset, rotate=H, bias=noise, lb=-5, ub=5)
        instances.append(copy.deepcopy(instance))
    
    return instances,ids

def sample_batch_task_id(dim,batch_size,f_id,seed=None):
    # dim=opts.dim
    # mode=opts.training_set
    # batch_size=opts.batch_size
    instances=[]
    # ids=[]
    # dim=config.dim
    set_seed(seed)
    # no noise
    # noise=random.uniform(0,0.1)
    
    for i in range(batch_size):
        # f_id=random.choice(problem_ids.get(mode))
        # ids.append(f'F{f_id}')
        # dim=random.choice(D.get(mode))
        # no bias
        noise=0
        # shift
        offset=np.random.random(dim) * 10 - 5
        # rotate
        H = rotate_gen(dim)
        
        instance = eval(f'F{f_id}')(dim=dim, shift=offset, rotate=H, bias=noise, lb=-5, ub=5)
        instances.append(copy.deepcopy(instance))
    p_name=instances[-1].__str__()

    return instances,p_name

def sample_batch_task_cec21(config):
    dataloader=Training_Dataset(filename=None, dim=config.dim, num_samples=config.batch_size, biased=False, shifted=True, rotated=True,
                            batch_size=config.batch_size,xmin=config.min_x,xmax=config.max_x,indicated_specific=True,indicated_dataset=cec2021)
    ids=dataloader.keys
    problem_list=dataloader[0]
    return problem_list,ids


def get_train_set(config):
    train_ids=problem_ids.get(config.training_set)
    train_set=[]
    dim=config.dim
    shifted=True
    ub=config.max_x
    lb=config.min_x
    biased=False
    rotated=True
    instance_seed=3849
    # np.random.seed(instance_seed)
    np.random.seed(instance_seed)
    for id in train_ids:
        if shifted:
            shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
        else:
            shift = np.zeros(dim)
        if rotated:
            H = rotate_gen(dim)
        else:
            H = np.eye(dim)
        if biased:
            bias = np.random.randint(1, 26) * 100
        else:
            bias = 0
        instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
        
        train_set.append(instance)
    return train_set,train_ids

# 直接返回所有问题
def get_test_set(config):
    # test_ids=list(set([i for i in range(1, 25)])-set(problem_ids.get(config.training_set)))
    test_ids=[i for i in range(1, 25)]
    test_set=[]
    dim=config.dim
    shifted=True
    ub=config.max_x
    lb=config.min_x
    biased=True
    rotated=True
    instance_seed=3849
    # np.random.seed(instance_seed)
    np.random.seed(instance_seed)
    for id in test_ids:
        if shifted:
            shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
        else:
            shift = np.zeros(dim)
        if rotated:
            H = rotate_gen(dim)
        else:
            H = np.eye(dim)
        if biased:
            bias = np.random.randint(1, 26) * 100
        else:
            bias = 0
        instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
        
        test_set.append(instance)
    
    return test_set,test_ids


real_pro_name=['Bent_cigar','Schwefel','bi_Rastrigin','Grie_rosen','Hybrid1','Hybrid2','Hybrid3','Composition1','Composition2','Composition3']
def get_train_set_cec21(config):
    train_ids=range(1,11)
    problem_list=[]
    dim=config.dim
    min_x=-100
    max_x=100
    biased=False
    shifted=True
    rotated=True
    for id in train_ids:
        pro,subproblems,sublength,Comp_sigma,Comp_lamda,indicated_dataset=get_config(id)
        pro=Training_Dataset(filename=None, dim=dim, num_samples=1, problems=pro, biased=biased, shifted=shifted, rotated=rotated,
                            batch_size=1,xmin=min_x,xmax=max_x,indicated_specific=True,indicated_dataset=indicated_dataset,
                            problem_list=subproblems,problem_length=sublength,Comp_sigma=Comp_sigma,Comp_lamda=Comp_lamda)[0][0]
        problem_list.append(pro)
    return problem_list,real_pro_name

'''for augmented cec2021 dataset generation'''
pro_name=['Bent_cigar','Schwefel','bi_Rastrigin','Grie_rosen','Hybrid','Hybrid','Hybrid','Composition','Composition','Composition']

def get_config(problem_id):
    pro=pro_name[problem_id-1]
    subproblems=None
    sublength=None
    Comp_sigma=None
    Comp_lamda=None
    indicated_dataset=None
    if problem_id==5:
        # hybrid 1
        subproblems=['Schwefel','Rastrigin','Ellipsoidal']
        sublength=[0.3,0.3,0.4]
    elif problem_id==6:
        # hybrid 2
        subproblems=['Escaffer6','Hgbat','Rosenbrock','Schwefel']
        sublength=[0.2,0.2,0.3,0.3]
    elif problem_id==7:
        # hybrid 3
        subproblems=['Escaffer6','Hgbat','Rosenbrock','Schwefel','Ellipsoidal']
        sublength=[0.1,0.2,0.2,0.2,0.3]
    elif problem_id==8:
        # # composition 1
        subproblems=['Rastrigin','Griewank','Schwefel']
        sublength=None
        Comp_sigma=[10,20,30]
        Comp_lamda=[1,10,1]
    elif problem_id==9:
        # composition 2
        subproblems=['Ackley','Ellipsoidal','Griewank','Rastrigin']
        sublength=None
        Comp_sigma=[10,20,30,40]
        Comp_lamda=[10,1e-6,10,1]
    elif problem_id==10:
        # composition 3
        subproblems=['Rastrigin','Happycat','Ackley','Discus','Rosenbrock']
        sublength=None
        Comp_sigma=[10,20,30,40,50]
        Comp_lamda=[10,1,10,1e-6,1]
    elif problem_id==11:              
        indicated_dataset=cec2021
    return (pro,subproblems,sublength,Comp_sigma,Comp_lamda,indicated_dataset)

def sample_batch_task_id_cec21(dim, batch_size,problem_id,seed=None):
    '''return a dataloader'''
    pro,subproblems,sublength,Comp_sigma,Comp_lamda,indicated_dataset=get_config(problem_id)
    num_samples=batch_size
    min_x=-100
    max_x=100
    biased=False
    shifted=True
    rotated=True
    problem_list=Training_Dataset(filename=None, dim=dim, num_samples=num_samples, problems=pro, biased=biased, shifted=shifted, rotated=rotated,
                            batch_size=batch_size,xmin=min_x,xmax=max_x,indicated_specific=True,indicated_dataset=indicated_dataset,
                            problem_list=subproblems,problem_length=sublength,Comp_sigma=Comp_sigma,Comp_lamda=Comp_lamda)[0]
    p_name=real_pro_name[problem_id-1]
    return problem_list,p_name