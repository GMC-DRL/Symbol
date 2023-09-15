import torch
import torch.nn as nn

from expr.tokenizer import MyTokenizer
from expr.expression import *
from torch.distributions import Categorical
import numpy as np
import math

binary_code_len=4

from .utils import *


class rnn_new(nn.Module):
    def __init__(self,opts,tokenizer:Tokenizer) -> None:
        super().__init__()
        self.opts=opts
        self.max_layer=opts.max_layer
        self.output_size=tokenizer.vocab_size
        self.hidden_size=opts.hidden_dim
        self.num_layers=opts.num_layers
        self.max_c=opts.max_c
        self.min_c=opts.min_c
        self.fea_size=opts.fea_dim
        self.tokenizer=tokenizer
        self.interval=opts.c_interval
        self.rnn=nn.RNN(int(2**self.max_layer-1)*binary_code_len+opts.fea_dim,self.hidden_size,self.num_layers,batch_first=True)
        self.output_net=nn.Linear(self.hidden_size,self.output_size)
        # self.x_to_h=nn.Linear(self.fea_size,self.hidden_size)
        # get const val
        self.constval_net=nn.Linear(self.hidden_size,int((self.max_c-self.min_c)//self.interval))
    

    def forward(self,x,save_data=False,fix_action=None):
        # depend on diff
        x=x[:,:self.opts.fea_dim]

        bs,fea_dim=x.size()
        device=x.device

        log_prob_whole=torch.zeros(bs).to(device)

        h_0=torch.zeros((bs,self.hidden_size))
        
        h=h_0.unsqueeze(dim=0)

        if save_data:
            memory=Memory()

        if not fix_action:
            # 使用数组表示树的方式
            len_seq=int(2**self.max_layer-1)
            seq=(torch.ones((bs,len_seq),dtype=torch.long)*-1)
            const_vals=torch.zeros((bs,len_seq))

            # prefix_seq=[]
            # prefix_const_vals=[]

            # initial input,h for rnn
            # h_0=torch.zeros((bs,self.hidden_size))
            
            x_in=torch.zeros((bs,len_seq*binary_code_len))
            x_in=torch.cat((x_in,x),dim=-1).unsqueeze(-2).to(device)
            
            
            # the generating position of the seq
            position=torch.zeros((bs,),dtype=torch.long)
            finished=torch.zeros((bs,))
            
            working_index=torch.arange(bs)
            # generate sequence
            while working_index.shape[0]>0 :

                output,h=self.rnn(x_in,h)
                out=self.output_net(output)

                # 如果position为全-1，则mask为全0
                mask=get_mask(seq[working_index],self.tokenizer,position,self.max_layer)
                
                # mask=get_mask(pre_seq,self.tokenizer,position)
                log_prob,choice,binary_code=get_choice(out,mask)
                # prefix_seq.append(choice)
                

                c_index=self.tokenizer.is_consts(choice)
                
                if np.any(c_index):
                    out_c=self.constval_net(output[c_index])
                    log_prob_c,c=get_c(out_c,self.min_c,self.interval)
                    log_prob_whole[working_index[c_index]]+=log_prob_c
                    const_vals[working_index[c_index],position[c_index]]=c.cpu()

                # if self.tokenizer.is_consts(choice):
                #     out_c=self.constval_net(output)
                #     log_prob_c,c=get_c(out_c,self.min_c,self.interval)
                #     log_prob_whole+=log_prob_c
                #     const_vals[torch.arange(bs),position]=c
                #     prefix_const_vals.append(c)
                # else:
                #     prefix_const_vals.append(0)

                # store if needed 
                if save_data:
                    memory.c_index.append(c_index)
                    memory.position.append(position)
                    memory.working_index.append(working_index)
                    memory.mask.append(mask)
                    memory.x_in.append(x_in.clone().detach())

                # udpate
                # need to test!!!!
                x_in=x_in.clone().detach()
                for i in range(binary_code_len):
                    x_in[range(len(working_index)),0,position*binary_code_len+i]=binary_code[:,i]
                # print(f'choice:{choice[0]},position:{position[0]}')
                # print(f'binary_code:{binary_code[0]}')
                # print(f'x_in:{x_in[0,0]}')
                
                # print(f'x_in.shape:{x_in.shape}')

                log_prob_whole[working_index]+=log_prob
                # pre_seq.append(choice)


                seq[working_index,position]=choice.cpu()
                # print(f'seq:{seq[0]}')
                position=get_next_position(seq[working_index],choice,position,self.tokenizer)
                
                # update working index when position is -1
                filter_index=(position!=-1)
                working_index=working_index[filter_index]
                position=position[filter_index]
                x_in=x_in[filter_index]
                h=h[:,filter_index]
                if save_data:
                    memory.filter_index.append(filter_index)
            # torch.rand()
                
            
            if self.opts.require_baseline:
                rand_seq,rand_c_seq=self.get_random_seq(bs)
                if not save_data:
                    return seq.numpy(),const_vals.numpy(),log_prob_whole,rand_seq,rand_c_seq
                else:
                    memory.seq=seq
                    memory.c_seq=const_vals
                    return seq.numpy(),const_vals.numpy(),log_prob_whole,rand_seq,rand_c_seq,memory.get_dict()
            
            if not save_data:
                # 返回等长的序列，数组表示的二叉树
                return seq.numpy(),const_vals.numpy(),log_prob_whole
            else:
                memory.seq=seq
                memory.c_seq=const_vals

                return seq.numpy(),const_vals.numpy(),log_prob_whole,memory.get_dict()
        else:
            # fix_action get the new log_prob
            x_in=fix_action['x_in']     # x_in shape: (len, [bs,1,31*4])
            mask=fix_action['mask']     # mask shape: (len, [bs,vocab_size])
            working_index=fix_action['working_index']   # working_index
            # seq=torch.FloatTensor(fix_action['seq']).to(device)
            seq=fix_action['seq']
            c_seq=fix_action['c_seq']
            # c_seq=torch.FloatTensor(fix_action['c_seq']).to(device)
            position=fix_action['position']
            c_indexs=fix_action['c_index']
            filter_index=fix_action['filter_index']

            for i in range(len(x_in)):
                output,h=self.rnn(x_in[i],h)
                out=self.output_net(output)

                w_index=working_index[i]
                pos=position[i]
                log_prob=get_choice(out,mask[i],fix_choice=seq[w_index,pos])
                log_prob_whole[w_index]+=log_prob

                c_index=c_indexs[i]
                # todo get c log_prob
                if np.any(c_index):
                    out_c=self.constval_net(output[c_index])
                    log_prob_c=get_c(out_c,self.min_c,self.interval,fix_c=c_seq[w_index[c_index],pos[c_index]])
                    log_prob_whole[w_index[c_index]]+=log_prob_c

                # update h
                h=h[:,filter_index[i]]
            return log_prob_whole


    def get_random_seq(self,bs):
        # 使用数组表示树的方式
        len_seq=int(2**self.max_layer-1)
        seq=(torch.ones((bs,len_seq),dtype=torch.long)*-1)
        const_vals=torch.zeros((bs,len_seq))
        position=torch.zeros((bs,),dtype=torch.long)

        working_index=torch.arange(bs)
        # generate sequence
        while working_index.shape[0]>0 :

            output=torch.rand((working_index.shape[0],1,self.output_size))

            # 如果position为全-1，则mask为全0
            mask=get_mask(seq[working_index],self.tokenizer,position,self.max_layer)
            
            # mask=get_mask(pre_seq,self.tokenizer,position)
            _,choice,_=get_choice(output,mask)
            # prefix_seq.append(choice)
            
            c_index=self.tokenizer.is_consts(choice)
            
            if np.any(c_index):
                out_c=torch.rand_like(output[c_index])
                _,c=get_c(out_c,self.min_c,self.interval)
                const_vals[working_index[c_index],position[c_index]]=c

            seq[working_index,position]=choice

            position=get_next_position(seq[working_index],choice,position,self.tokenizer)
            
            # update working index when position is -1
            filter_index=(position!=-1)
            working_index=working_index[filter_index]
            position=position[filter_index]
            

        return seq.numpy(),const_vals.numpy()
    
    