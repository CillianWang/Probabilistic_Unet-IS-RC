import imp
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import datetime
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
import quick_analysis
from  analysis.plot import *

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def train(model, data, val_data, args, device, optimizer, scheduler):
    things = []
    time = str(datetime.datetime.now()).replace(" ","_").replace(":","_")[:10]
    # record the starting time
    
    ACC, LOSS = [], []
    eval_ACC, eval_LOSS = [], []
    for epoch in range(args['epoch']):
        epoch_acc, epoch_loss, cnt = 0, 0, 0
        
        # "Training:"
        if args['training']:
            for dl, rl in data:
                mask = None
                dl, rl = dl.to(device, dtype=torch.float), rl.to(device).unsqueeze(1)
                segm = torch.repeat_interleave(rl, 30*128,dim=2)
                model.forward(dl, segm, training = True)
                elbo, acc = model.elbo(rl, mask)
                reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.fcomb.layers)
                loss = - elbo + 1e-5 * reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_acc+=acc
                epoch_loss+=loss
                cnt+=1

            train_acc, train_loss = epoch_acc/cnt, epoch_loss/cnt
        
        # "Evaluation"
        cur = 0
        for dl, rl in val_data:
            with torch.no_grad():
                dl, rl = dl.to(device, dtype=torch.float), rl.to(device)
                pres, latent_space = [], []
                
                pres_plot = []
                for plot_idx in range(6):
                    
                    rl_ = rl[:,plot_idx,:].unsqueeze(1)
                    segm = torch.repeat_interleave(rl_, 30*128,dim=2)
                    model.forward(dl, segm, training = True)
                    
                    prior = model.posterior_latent_space.rsample()
                    cur_pre = model.reconstruct(use_posterior_mean=False, calculate_posterior=False, z_posterior=prior)
                    pres_plot.append(cur_pre.cpu().argmax(1).reshape(1610))

                plot_pre_ann(pres_plot, rl)

                    
                
                
                for val_idx in range(100):
                    rl_ = rl[:,random.randint(0,5),:].unsqueeze(1)
                    segm = torch.repeat_interleave(rl_, 30*128,dim=2)
                    model.forward(dl, segm, training = True)
                    
                    prior = model.prior_latent_space.rsample()
                    cur_pre = model.reconstruct(use_posterior_mean=False, calculate_posterior=False, z_posterior=prior)
                    latent_space.append(prior)
                    pres.append(cur_pre)
                ret = quick_analysis.analysis(pres, rl)
                    
                        
                    
                if args['sys']=='windows':
                    with open(r'C:\Users\cilli\Codes\Punet\record\out_'+str(cur), 'wb') as fp:
                        pickle.dump([pres,rl,ret,latent_space], fp)
                if args['sys']=='linux':
                    with open('/home/IS-RC/record/out_'+str(cur), 'wb') as fp:
                        pickle.dump([pres,rl,ret,latent_space], fp)
                cur += 1




        "Saving training progress results and model parameters"
        # ACC.append(train_acc)
        # LOSS.append(train_loss)
        # eval_ACC.append(eval_acc)
        # eval_LOSS.append(eval_loss)
        # things = [ACC,LOSS,eval_ACC,eval_LOSS]
        # print('val'+str(float(eval_acc))+', '+str(float(eval_loss)))
        
        
        # print('train'+str(float(train_acc))+', '+str(float(train_loss)))
        # print('')
        
        # if epoch%10 == 0:
        #     if args['sys']=='windows':
        #         with open(r'C:\Users\cilli\Codes\Punet\record\record_'+time, 'wb') as fp:
        #             pickle.dump(things, fp)
        #     if args['sys']=='linux':
        #         with open('/home/Utime/Punet/record/record_'+time, 'wb') as fp:
        #             pickle.dump(things, fp)
        path = Path(os.path.abspath(__file__)).parent.absolute()
        para_path = os.path.join(path,'para')
        torch.save(model.state_dict(),os.path.join(para_path,'punet_ISRC'+time+'.pth'))
    return