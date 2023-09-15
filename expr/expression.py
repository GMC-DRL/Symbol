'''imprement the expression related operation'''


import sympy
import torch
import numpy as np
from .tokenizer import Tokenizer
import math


# expression related function

# pre_seq 是一个完整的数组（用数组来表达二叉树）
# 先实现一个不是batch的版本, use for loop to implement batch operation
# pre_
def get_mask(pre_seq,tokenizer,position,max_layer):
    if len(pre_seq.shape)==1:
        pre_seq=[pre_seq]
    bs,_=pre_seq.size()
    old_device=pre_seq.device
    pre_seq=pre_seq.cpu().numpy()
    position=position.cpu().numpy()
    masks=[]
    for sub_seq,pos in zip(pre_seq,position):
        # 规则0：if position==-1: mask all to be zero
        if pos == -1:
            mask=np.zeros(tokenizer.vocab_size)
            masks.append(mask)
            continue
        # init mask
        mask=np.ones(tokenizer.vocab_size)
        # 规则一：第一个token不能是算式
        if pos==0:
            mask[tokenizer.leaf_index]=0
            # mask[tokenizer.encode('sign')]=0
            # mask[tokenizer.encode('sin')]=0
            # mask[tokenizer.encode('cos')]=0
            
            # mask[tokenizer.encode('*')]=0
            mask[tokenizer.encode('-')]=0
        else:
            # 规则二：＋-无效运算不出现
            # 寻找连续+，得到一个vocab里装有
            # 先找出它对应于祖先减法的prefix, 判断其祖先是否有减号
            father_token=tokenizer.decode(sub_seq[(pos-1)//2])
            
            # 优化：只有在输出最后一个叶子的时候才判断逆运算
            if (tokenizer.is_binary(father_token) and pos%2 == 0) or tokenizer.is_unary(father_token):
                neg_ancestor,target_vocab=find_prefix_of_token_ancestor(tokenizer,sub_seq,pos,'-')
                # - 的直接孩子不能是 +,-
                if neg_ancestor == (pos-1)//2:
                    mask[tokenizer.encode('+')]=0
                    mask[tokenizer.encode('-')]=0
                    # 位于root的-后面不能是x
                    if neg_ancestor == 0:
                        mask[tokenizer.encode('x')]=0
                # print(f'neg_ancestor:{neg_ancestor}')
                # print(f'target_vocab:{target_vocab}')
                # 如果有，则沿着+的通路得到一个pre_vacab
                if target_vocab is not None:
                    pre_vocab=along_continuous_plus(tokenizer,sub_seq,neg_ancestor)
                    # print(f'target_vocab:{target_vocab}, pre_vocab:{pre_vocab}')
                    # 检查target_vocab是否在pre_vacab之中
                    if pre_vocab is not None:
                        # 主要的思路是最后一个不能一样，其他都可以一样, 即检测是否为前缀
                        mask_index=test_pre(target_vocab[1:],pre_vocab,tokenizer)
                        mask[mask_index]=0
            
            # 对于＋的前置条件多一个，如果直接父亲是＋的话，无论左孩子还是右孩子都得检查
            if father_token == '+' or (tokenizer.is_binary(father_token) and pos%2 == 0) or tokenizer.is_unary(father_token):
                plus_ancestor,target_vocab=find_prefix_of_token_ancestor(tokenizer,sub_seq,pos,'+')
                # print(f'plus_ancestor:{plus_ancestor}')
                if target_vocab is not None:
                    visited=np.zeros_like(sub_seq)
                    # todo: target vocab 到底要取哪边
                    if father_token=='+' and left_or_right(pos,plus_ancestor)=='l':
                        visited[2*plus_ancestor+1]=1
                        target_vocab=get_prefix(sub_seq,2*plus_ancestor+1)
                    else:
                        visited[2*plus_ancestor+2]=1
                        target_vocab=get_prefix(sub_seq,2*plus_ancestor+2)
                    # if sub_seq[2*plus_ancestor+2]!=-1:
                        
                    # else:
                    #     visited[2*plus_ancestor+1]=1
                        # target_vocab=get_prefix(sub_seq,2*plus_ancestor+1)
                    # print(f'target_vocab:{[tokenizer.decode(i) for i in target_vocab]}')
                    sub_root_list=get_along_continuous_plus_with_minus(tokenizer,sub_seq,plus_ancestor,visited)
                    # print(f'sub_root:{sub_root_list}')
                    pre_vocab=[get_prefix(sub_seq,sub_root) for sub_root in sub_root_list]
                    if pre_vocab is not None:
                        # 主要的思路是最后一个不能一样，其他都可以一样, 即检测是否为前缀
                        mask_index=test_pre(target_vocab,pre_vocab,tokenizer)
                        mask[mask_index]=0
            # 规则三：不能出现纯常数运算,  todo 不能连乘，找连续×号
            # print('enter finding continuous consts')
            # if have_continous_const(sub_seq,pos,tokenizer) or continus_mul_c(sub_seq,pos,tokenizer):
            #     mask[tokenizer.constants_index]=0
            if have_continous_const(sub_seq,pos,tokenizer):
                mask[tokenizer.constants_index]=0
            
            # 规则五：sin,cos,sign之间嵌套关系
            
            # if father_token in ['sin','cos']:
            #     mask[tokenizer.encode('sign')]=0
            #     mask[tokenizer.encode('sin')]=0
            #     # mask[tokenizer.encode('cos')]=0
            # if father_token == 'sign':
            #     mask[tokenizer.encode('sign')]=0
            # ＋ - 的直接孩子不能是常数
            if father_token == '+' or father_token == '-':
                mask[tokenizer.constants_index]=0

            # 规则六：+的直接孩子不能是sign
            if father_token == '+':
                # mask[tokenizer.encode('sign')]=0

                # 规则七：不能出现x+x，gbest+gbest类似的东西，当然randx+randx可以
                if pos%2==0:    # 右孩子
                    # 右孩子不能直接等于左孩子
                    left_token=tokenizer.decode(sub_seq[pos-1])
                    if tokenizer.is_leaf(left_token) and left_token!='randx':
                        mask[sub_seq[pos-1]]=0
            
             # 规则七：乘法的孩子不能是同类
            if father_token == '*':
                mask[tokenizer.encode('*')]=0
                mask[tokenizer.encode('-')]=0
                if pos%2==0:
                    left_id=sub_seq[pos-1]
                    # 左孩子不是常数，则右孩子必须为常数
                    if not tokenizer.is_consts(left_id):
                        mask[tokenizer.non_const_index]=0
                    else:
                        mask[tokenizer.constants_index]=0
        
            # 可选规则：树的大小至少要第三层
            # if which_layer(position=pos)<=2:
            #     if father_token=='*':
            #         mask[tokenizer.var_index]=0
            #     elif (tokenizer.is_binary(father_token) and pos%2 == 0 and tokenizer.is_leaf(tokenizer.decode(sub_seq[pos-1]))) or tokenizer.is_unary(father_token):
            #         mask[tokenizer.leaf_index]=0

            # 规则四：最后一层不能是operator
            if pos >= int(2**(max_layer-1)-1):
                mask[tokenizer.operator_index]=0
        if np.all(mask<=0.2):
            # todo: delect
            # mask[tokenizer.leaf_index]=1
            print(f'mask:{mask}, pos:{pos}, seq:{sub_seq}')
        masks.append(mask)
    
    # print(f'post_position:{position}')
    return torch.FloatTensor(masks).to(old_device)

def which_layer(position):
    level = math.floor(math.log2(position + 1))
    return level+1

def left_or_right(position,root):
    tmp=position
    while tmp!=root:
        position=(position-1)//2
        if position == root:
            if 2*root+1 == tmp:
                return 'l'
            else:
                return 'r'
        tmp=position

# 简单的判断规则：1. 对于单目运算符，其孩子不能是常数 2. 对于双目运算符，只有当左兄弟为常数时，限制右兄弟不能为常数
def have_continous_const(seq,position,tokenizer):
    father_index=(position-1)//2
    father_token=tokenizer.decode(seq[father_index])
    if tokenizer.is_unary(father_token):
        # print('in const test')
        # print(f'position:{position}')
        # print(f'father_token:{father_token}')
        return True
    if tokenizer.is_binary(father_token):
        if position==father_index*2+1:
            return False
        elif tokenizer.is_consts(seq[father_index*2+1]):
            return True

def continus_mul_c(seq,position,tokenizer):
    list=[]
    sub_root=(position-1)//2
    # 看其父亲是否为*，是则从其父亲开始找连续*通路
    if tokenizer.decode(seq[sub_root])=='*':
        visited=np.zeros_like(seq)
        visited[position]=1
        
        return get_along_continuous_mul(tokenizer,seq,sub_root,visited)
    else:
        return False

# 找连续*的通路
def get_along_continuous_mul(tokenizer,seq,begin,visited):
    
    # list.append(begin)
    visited[begin]=1
    
    if begin!=0 and visited[(begin-1)//2]!=1:
        father_token=tokenizer.decode(seq[(begin-1)//2])
        if father_token=='*':
            if get_along_continuous_mul(tokenizer,seq,(begin-1)//2,visited):
                return True
    
    if visited[begin*2+1]==0 and seq[begin*2+1]!=-1:
        left_child_token=tokenizer.decode(seq[begin*2+1])
        if left_child_token=='*':
            if get_along_continuous_mul(tokenizer,seq,begin*2+1,visited):
                return True
        elif left_child_token[0]=='C':
            return True
    
    if visited[begin*2+2]==0 and seq[begin*2+2]!=-1:
        right_child_token=tokenizer.decode(seq[begin*2+2])
        if right_child_token=='*':
            if get_along_continuous_mul(tokenizer,seq,begin*2+2,visited):
                return True
        elif right_child_token[0]=='C':
            return True
    
    return False

def test_pre(target_vocab,pre_vocab,tokenizer):
    target_len=len(target_vocab)
    mask_index=[]
    for pre_prefix in pre_vocab:
        # 检查除最后一个的前缀是否相等
        if len(pre_prefix)==target_len+1 and np.all(pre_prefix[:-1]==target_vocab):
            # 但是randx-randx是被允许的，或者说randx被视作不同的variable
            last_token=tokenizer.decode(pre_prefix[-1])
            if last_token != 'randx' and last_token[0] != 'C':
                mask_index.append(pre_prefix[-1])
            
            
    return mask_index
        

# 找连续＋的通路, 并返回其中儿子为-的前序遍历
def get_along_continuous_plus_with_minus(tokenizer,seq,begin,visited):
    list=[]
    
    # list.append(begin)
    visited[begin]=1
    

    if begin!=0 and visited[(begin-1)//2]==0:
        father_token=tokenizer.decode(seq[(begin-1)//2])
        if father_token=='+':
            l=get_along_continuous_plus_with_minus(tokenizer,seq,(begin-1)//2,visited)
            list.extend(l)
   
    
    if visited[begin*2+1]==0 and seq[begin*2+1]!=-1:
        left_child_token=tokenizer.decode(seq[begin*2+1])
        if left_child_token=='+':
            l=get_along_continuous_plus_with_minus(tokenizer,seq,begin*2+1,visited)
            list.extend(l)
        elif left_child_token == '-':
            list.append(2*(begin*2+1)+1)
    
    if visited[begin*2+2]==0 and seq[begin*2+2]!=-1:
        right_child_token=tokenizer.decode(seq[begin*2+2])
        if right_child_token=='+':
            l=get_along_continuous_plus_with_minus(tokenizer,seq,begin*2+2,visited)
            list.extend(l)
        elif left_child_token == '-':
            list.append(2*(begin*2+2)+1)
    
    return list

# 找连续＋的通路
def get_along_continuous_plus(tokenizer,seq,begin,visited):
    list=[]
    # list.append(begin)
    along_root=False
    visited[begin]=1
    if begin == 0 and seq[begin] == tokenizer.encode('+'):
        along_root = True

    if begin!=0 and visited[(begin-1)//2]==0:
        father_token=tokenizer.decode(seq[(begin-1)//2])
        if father_token=='+':
            l,flag=get_along_continuous_plus(tokenizer,seq,(begin-1)//2,visited)
            list.extend(l)
            if flag:
                along_root=True
   
    
    if visited[begin*2+1]==0 and seq[begin*2+1]!=-1:
        left_child_token=tokenizer.decode(seq[begin*2+1])
        if left_child_token=='+':
            l,flag=get_along_continuous_plus(tokenizer,seq,begin*2+1,visited)
            list.extend(l)
            if flag:
                along_root=True
        else:
            list.append(begin*2+1)
    
    if visited[begin*2+2]==0 and seq[begin*2+2]!=-1:
        right_child_token=tokenizer.decode(seq[begin*2+2])
        if right_child_token=='+':
            l,flag=get_along_continuous_plus(tokenizer,seq,begin*2+2,visited)
            list.extend(l)
            if flag:
                along_root=True
        else:
            list.append(begin*2+2)
    
    return list,along_root

# 返回连续＋通路中＋节点的子树的前序遍历
def along_continuous_plus(tokenizer,seq,neg_ancestor):
    list=[]
    sub_root=(neg_ancestor-1)//2
    # 看其父亲是否为+，是则从其父亲开始找连续+通路
    if tokenizer.decode(seq[sub_root])=='+':
        visited=np.zeros_like(seq)
        visited[neg_ancestor]=1
        continuous_plus_token_list,along_root=get_along_continuous_plus(tokenizer,seq,sub_root,visited)
        
        pre_vocab=[get_prefix(seq,sub_root) for sub_root in continuous_plus_token_list]

        if along_root:
            pre_vocab.append([tokenizer.encode('x')])
            # pre_vocab.append([tokenizer.encode('-'),tokenizer.encode('x')])
        return pre_vocab
    else:
        return None
    


# 一个问题：-号下面还有+号吗
# 找祖先节点是否有-，有则返回该祖先节点字数的前序遍历
def find_prefix_of_token_ancestor(tokenizer,seq,position,token):
    
    while True:
        father_index=(position-1)//2
        father_token=tokenizer.decode(seq[father_index])
        if father_token!=token:
            position=father_index
            if position==0:
                break
        else:
            return father_index,get_prefix(seq,father_index)
    return -1,None

# 前序遍历
def get_prefix(seq,sub_root):
    if  sub_root>=len(seq) or seq[sub_root]==-1:
        return []
    list=[]
    list.append(seq[sub_root])
    list.extend(get_prefix(seq,2*sub_root+1))
    list.extend(get_prefix(seq,2*sub_root+2))
    return list

def get_prefix_with_consts(seq,consts,sub_root):
    if  sub_root>=len(seq) or seq[sub_root]==-1:
        return [],[]
    list_expr=[]
    list_c=[]
    list_expr.append(seq[sub_root])
    list_c.append(consts[sub_root])
    left_output=get_prefix_with_consts(seq,consts,2*sub_root+1)
    list_expr.extend(left_output[0])
    list_c.extend(left_output[1])
    right_output=get_prefix_with_consts(seq,consts,2*sub_root+2)
    
    list_expr.extend(right_output[0])
    list_c.extend(right_output[1])
    return list_expr,list_c

def get_next_position(seq,choice,position,tokenizer):
    old_device=position.device
    position=position.cpu().numpy()
    choice=choice.cpu().numpy()
    seq=seq.cpu().numpy()
    next_position=[]
    for i in range(len(position)):
        c=choice[i]
        pos=position[i]
        sub_seq=seq[i]
        # 上一步输出了个operator，则下一步就要输出其左孩子
        if c in tokenizer.operator_index:
            next_position.append(2*pos+1)
        else:
            append_index=-1
            # 上一步输出了叶子节点，则回溯，找到第一个没有右孩子节点的节点
            while True:
                father_index=(pos-1)//2
                if father_index<0:
                    break
                if sub_seq[father_index] in tokenizer.binary_index and sub_seq[2*father_index+2]==-1:
                    append_index=father_index*2+2
                    break
                pos=father_index
            # 返回-1表示下一个位置不存在，也即生成式已经完整
            next_position.append(append_index)
        
    return torch.tensor(next_position,dtype=torch.long).to(old_device)
# 
def get_str_prefix(seq,const_vals,tokenizer):
    str_expr=[]
    c=[]
    for i,token_id in enumerate(seq):
        if token_id != -1:
            str_expr.append(tokenizer.decode(token_id))
            c.append(const_vals[i])
    return str_expr,c


# copy from symformer
# 将前序遍历生成中序遍历
def prefix_to_infix(
    expr, constants, tokenizer: Tokenizer
):
    stack = []
    for i, symbol in reversed(list(enumerate(expr))):
        if tokenizer.is_binary(symbol):
            if len(stack) < 2:
                return False, None
            tmp_str = "(" + stack.pop() + symbol + stack.pop() + ")"
            stack.append(tmp_str)
        elif tokenizer.is_unary(symbol) or symbol == "abs":
            if len(stack) < 1:
                return False, None
            if symbol in tokenizer.SPECIAL_SYMBOLS:
                stack.append(tokenizer.SPECIAL_SYMBOLS[symbol].format(stack.pop()))
            else:
                stack.append(symbol + "(" + stack.pop() + ")")
        elif tokenizer.is_leaf(symbol):
            if symbol == "C":
                stack.append(str(constants[i]))
            elif "C" in symbol:
                exponent = int(symbol[1:])
                stack.append(str(constants[i] * 10 ** exponent))
            else:
                stack.append(symbol)

    if len(stack) != 1:
        return False, None

    return True, stack.pop()

from typing import List

from sympy import lambdify

def expr_to_func(sympy_expr, variables: List[str]):
    

    return lambdify(
        variables,
        sympy_expr,
        modules=["numpy"],
    )