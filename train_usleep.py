import numpy as np
import torch
import torch.nn as nn
import datetime
import pickle
import os
from pathlib import Path
from quick_analysis import *
from  analysis.plot import *

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
def masked_loss_acc(out, rl):
    # generate a mask, maybe should put in the dataloader in advance：
    # n = rl.shape[-1]
    # rl = rl.reshape([rl.shape[0], n])
    mask = (rl!=7)
    
    # accuracy
    ans = out.argmax(1)
    crt = (ans==rl).sum()
    
    # one_hot
    rl = nn.functional.one_hot(rl.mul(mask).long().squeeze(1), num_classes=5)
    rl = torch.transpose(rl, 1, 2)
    out = out
    
    # loss
    loss = -torch.log(out).mul(rl).mul(mask).sum(axis=1)
    
    n = mask.sum()
    
    
    return loss.sum()/n, crt/n

def generate_mask(mask):
    Mask = np.zeros(shape=(len(mask),1288))
    for i in range(len(mask)):
        Mask[i] = np.r_[np.ones(mask[i]),np.zeros(1288-mask[i])]
    return Mask
        

def train(data, model, args, device, optimizer, scheduler,scheduler_RLRoP, eval_data, load_para=False):
    print("Start training.")
    epochs = args['epoch']
    time = str(datetime.datetime.now()).replace(" ","_").replace(":","_")
    # get information about the data:
    batch_amount = torch.numel(next(iter(data))[-2])
    ACC, LOSS = [], []
    eval_ACC, eval_LOSS = [], []
    
    # load model:
        
    
    for epoch in range(epochs):
        # compute accucary:
        acc = []
        Loss = 0
        count = 0
        if args['training']:
            for dl, rl in data:
                

                dl, rl = dl.to(device, dtype=torch.float), rl.to(device).unsqueeze(1)
                count+=1

                out = model(dl)
                
                loss, accuracy = masked_loss_acc(out, rl)

                loss.backward()
                optimizer.step()
                Loss += loss.cpu()
                # estimate = out.argmax(-2)
                # accuracy = float((np.true_divide((estimate==rl).cpu().mul(torch.tensor(mask)).sum(),mask.sum())))
                acc.append(accuracy.cpu())
            
            if epoch%5 == 1: scheduler.step()
            
            
            Loss = Loss/count
            ACC.append(np.average(acc))
            LOSS.append(Loss)
        
        
        eval_acc = []
        eval_Loss, count = 0, 0
        all_analysis = []
        for dl, rl in eval_data:
            with torch.no_grad():
                if args['mcdo']==True: enable_dropout(model)
                dl, rl = dl.to(device, dtype=torch.float), rl.to(device)
                out = model(dl)
                
                cur_rl = majority_vote_rl(rl)
                eval_acc.append(majority_acc(out.argmax(1).cpu(), cur_rl))
                
                if args['mcdo']==True: 
                    pres_plot = []
                    for plot_idx in range(6):
                        
                        rl_ = rl[:,plot_idx,:].unsqueeze(1)
                        
                        cur_pre = model(dl)
                        pres_plot.append(cur_pre.cpu().argmax(1).reshape(1610))

                    # plot_pre_ann(pres_plot, rl)
                    
                    pres = []
                    for idx in range(100):
                        pres.append(model(dl))
                
                # multi predictions:
                things = [dl,rl,pres_plot]
                with open(r'C:\Users\cilli\Codes\Punet\record\plot_data', 'wb') as fp:
                    pickle.dump(things, fp)
                pres = []
                for idx in range(100):
                    pres.append(model(dl))
                # # save rl and pres
                # if args['sys']=='windows':
                #     with open(r'C:\Users\cilli\Codes\Punet\record\pres', 'wb') as fp:
                #         pickle.dump(pres, fp)
                #     with open(r'C:\Users\cilli\Codes\Punet\record\rl', 'wb') as fp:
                #         pickle.dump(rl, fp)
                
                all_analysis.append(analysis(pres, rl))
                
                count += 1
        pass
                # loss,accuracy = masked_loss_acc(out, rl)
                
                # eval_Loss += loss
                # accuracy = float((np.true_divide((estimate==rl).cpu().mul(torch.tensor(mask)).sum(),mask.sum())))
                # eval_acc.append(accuracy.cpu())
        # eval_ACC.append(np.average(eval_acc))
        # eval_LOSS.append(eval_Loss/count)
        # scheduler_RLRoP.step(eval_Loss/count)

             
        # print('Epoch '+str(epoch)+' Accuracy: '+str(np.average(acc))+' Loss:'+str(Loss))
        # print('Val: '+ str(np.average(eval_acc)) + ', Val Loss: '+str(eval_Loss/count))
        # things = [ACC,LOSS,eval_ACC,eval_LOSS]
        # if epoch%10 == 0:
        #     if args['sys']=='windows':
        #         with open(r'C:\Users\cilli\Codes\Punet\record\record_'+time, 'wb') as fp:
        #             pickle.dump(things, fp)
        #     if args['sys']=='linux':
        #         with open('/home/Utime/Punet/record/record_'+time, 'wb') as fp:
        #             pickle.dump(things, fp)
    
    path = Path(os.path.abspath(__file__)).parent.absolute()
    para_path = os.path.join(path,'para')
    torch.save(model.state_dict(),os.path.join(para_path,'usleep'+time+'.pth'))
    
# with open(r'C:\Users\cilli\Codes\Punet\record\acc', 'rb') as fp:
#     b = pickle.load(fp) 