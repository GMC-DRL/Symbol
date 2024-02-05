from .cec_test_func import *
from utils.utils import set_seed


import numpy as np
import csv
import random


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



def sample_batch_task_cec21(config):
    dataloader=Training_Dataset(filename=None, dim=config.dim, num_samples=config.batch_size, biased=False, shifted=True, rotated=True,
                            batch_size=config.batch_size,xmin=config.min_x,xmax=config.max_x,indicated_specific=True,indicated_dataset=cec2021)
    ids=dataloader.keys
    problem_list=dataloader[0]
    return problem_list,ids




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