
import numpy as np
from pyparsing import col
import torch
import pickle
import os
from pathlib import Path
from collections import Counter
import random
import sklearn.metrics
from Hungary_matched import hungarian_acc, hungarian_kappa
import collections
import matplotlib.pyplot as plt


#%% most frequent in a tensor:
def most_frequent(List):
    return max(set(List), key = List.count)

#%% majority vote for ground truth: shaped(batch, epochs)
def majority_vote_rl(rl):
    major_rl = torch.empty(rl.shape[0],rl.shape[-1])
    for b in range(rl.shape[0]):
        for i in range(rl.shape[-1]):
            hsh = {0:0, 1:0, 2:0, 3:0, 4:0, 7:0}
            cur = set(rl[b,:,i])
            for c in cur: hsh[int(c)]+=1
            major_rl[b,i] = list(hsh.keys())[np.argmax(list(hsh.values()))]
    return major_rl

#%% majority accuracy:
def majority_acc(major_pre, major_rl):
    if major_pre.shape != major_rl.shape: 
        print('error')
        return False
    return ((major_pre==major_rl).mul(major_rl!=7)).sum()/(major_rl!=7).sum()

#%% majority_kappa(y, gt):
def majority_kappa(y, gt):
    kappa = 0
    for i in range(y.shape[0]):
        cur_y, cur_gt = y[i].squeeze(0), gt[i].squeeze(0)
        cur_y = cur_y[cur_gt!=7]
        cur_gt = cur_gt[cur_gt!=7]
        kappa += sklearn.metrics.cohen_kappa_score(cur_y, cur_gt)
        
    return kappa/y.shape[0]


#%% 100
def hungarian100(record, rl):
    rl_100 = torch.empty([1, rl.shape[-1]])
    for i in range(17):
        rl_100 = torch.cat((rl_100, rl.squeeze(0).cpu()), 0)
    rl_100 = rl_100[1:101]
    
    y_100 = torch.empty((record[0].argmax(1).shape))
    for i in record:
        y_100 = torch.cat((y_100, i.argmax(1).cpu()), 0)
    y_100 = y_100[1:]
    
    return y_100, rl_100
        
#%% SOL: x shaped [n, t]
def sol(x):
    all = []
    for i in range(x.shape[0]):
        cur = x[i]
        for idx, t in enumerate(cur):
            if t == 7: continue
            if t!= 0: 
                all.append(idx)
                break
    return all

#%% value to distribution:
# use this function to make values to categorical distributions
def cate_dist(x, y):
    
    hsh_x, n_x = {}, len(x)
    for i in x:
        hsh_x[i] = hsh_x.get(i, 0)+1
        
    hsh_y, n_y = {}, len(y)
    for i in y:
        hsh_y[i] = hsh_y.get(i, 0)+1
    
    for i in hsh_x.keys():
        hsh_y[i] = hsh_y.get(i, 0)
    for i in hsh_y.keys():
        hsh_x[i] = hsh_x.get(i, 0)
    
    x = list(collections.OrderedDict(sorted(hsh_x.items())).values())
    y = list(collections.OrderedDict(sorted(hsh_y.items())).values())
    
    x = [i/n_x for i in x]
    y = [i/n_y for i in y]
    
    return x, y

# normal distribution kl:
def normal_kl(x,y):
    mu_x, sigma_x = np.mean(x), np.sqrt(np.var(x))
    mu_y, sigma_y = np.mean(y), np.sqrt(np.var(y))
    
    if sigma_x==0 or sigma_y==0: return float('inf')
    else: return np.log(sigma_y/sigma_x) + (sigma_x*sigma_x + (mu_x-mu_y)*(mu_x-mu_y))/(2*sigma_y*sigma_y) - 0.5
        

# average length of sleepstage:
def length_sleep(x):
    ans = []
    for j in range(x.shape[0]):
        cur = 0
        for i in x[j]:
            if i==7: continue
            if i!=0: cur+=1
        ans.append(cur)
    return ans

# awakenings
def awake(x):
    ret= []
    for j in range(x.shape[0]):
        ans, pre = 0, 0
        for cur in x[j]:
            if pre == 7 and cur == 0: continue
            if pre!=0 and cur==0: ans+=1
            pre = cur
        ret.append(ans)
    return ret

#%% kl between two normal distributions


# deal with predictions on ONE input:
# record 100 predictions, rl 6 annotations
def analysis(record, rl):
    
    # hungarian kappa
    votes, rl_100 = hungarian100(record, rl)
        
    # majority vote:
    majority_vote = torch.empty(record[0].argmax(1).shape)
    for i in range(votes.shape[-1]):
        cur = set(votes[:,i])
        hsh = {0:0, 1:0, 2:0, 3:0, 4:0, 7:0}
        for c in cur: hsh[int(c)]+=1     
        majority_vote[0,i]= list(hsh.keys())[np.argmax(list(hsh.values()))]

    # majority vote output vs gt accuracy:
    major_rl = majority_vote_rl(rl)
    acc = majority_acc(majority_vote, major_rl)
    
    # majority vote kappa
    kappa = majority_kappa(majority_vote,major_rl)
    
    # hungarian acc and kappa
    h_acc = hungarian_acc(votes, rl_100, 100)
    h_kappa = hungarian_kappa(votes, rl_100, 100)
    
    # sol
    sol_votes = sol(votes)
    sol_rl = sol(rl.squeeze(0))
    sol_kl = normal_kl(sol_votes, sol_rl)
    
    # average length of sleep:
    len_sleep_votes = length_sleep(votes)
    len_sleep_rl = length_sleep(rl.squeeze(0))
    len_sleep_kl = normal_kl(len_sleep_votes, len_sleep_rl)
    
    # numbers of awakenings:
    awake_votes = awake(votes)
    awake_rl = awake(rl.squeeze(0))
    awake_kl = normal_kl(awake_votes, awake_rl)
    
    
    ret = {
        'majority_vote': majority_vote,
        'majority_acc' : acc,
        'majority_kappa': kappa,
        'hungarian_acc': h_acc,
        'hungarian_kappa':h_kappa,
        'sol_votes': sol_votes,
        'sol_rl': sol_rl,
        'sol_kl': sol_kl,
        'len_sleep_votes': len_sleep_votes,
        'len_sleep_rl': len_sleep_rl,
        'len_sleep_kl': len_sleep_kl,
        'awake_votes': awake_votes,
        'awake_rl': awake_rl,
        'awake_kl': awake_kl
    }
    
    return ret




#%% tests:
if __name__ == "__main__":
    loc = os.path.abspath(__file__)[:60]+'\\records_for_latentspace'
    
    validation_analysis = []
    dist, sol_pre, sol_rl, awake_pre, awake_rl, len_pre, len_rl = [], [], [], [], [], [], []
    for i in range(10):
        with open(os.path.join(loc, 'out_'+str(i)), 'rb') as fp:
            [pres, rl, ret, latent_space] = pickle.load(fp)

        for j in latent_space: dist.append(np.array(j[0].cpu()))
        for j in ret['sol_votes']: sol_pre.append(j)
        for j in ret['sol_rl']: sol_rl.append(j)
        for j in ret['awake_votes']: awake_pre.append(j)
        for j in ret['awake_rl']: awake_rl.append(j)
        for j in ret['len_sleep_votes'] : len_pre.append(j)
        for j in ret['len_sleep_rl']: len_rl.append(j)
        # ret = analysis(pres, rl)
        
        validation_analysis.append(ret)
    
    major_acc, hung_acc= 0, 0
    for i in validation_analysis:
        major_acc += i['majority_acc']
        hung_acc += i['hungarian_acc']
    print(major_acc/10)
    print(hung_acc/10)
    
    
    num = 11
    idx = np.linspace(0,1000,num=num)
    color = ['maroon','coral','powderblue','slategrey','goldenrod','darkseagreen','teal','slateblue', 
             'navy','lightsteelblue','rosybrown']
    awake_pre_sort = np.argsort(awake_pre)
    awake_rl_sort = np.argsort(awake_rl)
    plt.figure(figsize=[3.5,3.5])
    for i in range(num-1):
        if i==3 or i==(num-2):
            cur_idx = awake_pre_sort[int(idx[i]):int(idx[i+1])]
            cur_awake_pre=[awake_pre[j] for j in cur_idx]
            cur_dist=[dist[j] for j in cur_idx]
            for j in cur_dist:
                if i == 3:
                    plt.scatter(j[0],j[1],c='teal')
                else:
                    plt.scatter(j[0], j[1], c='maroon')
    plt.title('Numbers of awakenings in latent space', fontsize=10)
    plt.legend(['low', 'high'], fontsize=10)
    plt.show()
    
        
        
        
    
    pass
    
