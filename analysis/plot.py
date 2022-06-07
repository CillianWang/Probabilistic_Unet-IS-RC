import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import pickle
import sys
from pathlib import Path
# insert at 1, 0 is the script path (or '' in REPL)
path = Path(os.path.abspath(__file__)).parent.parent.absolute()
sys.path.insert(0, str(path))
from models.Punet import ProbabilisticUnet
import random


# %% plot a single hypnogram
def majority_vote_rl(rl):
    major_rl = torch.empty(rl.shape[0],rl.shape[-1])
    for b in range(rl.shape[0]):
        for i in range(rl.shape[-1]):
            hsh = {0:0, 1:0, 2:0, 3:0, 4:0, 7:0}
            cur = set(rl[b,:,i])
            for c in cur: hsh[int(c)]+=1
            major_rl[b,i] = list(hsh.keys())[np.argmax(list(hsh.values()))]
    return major_rl


def plot_hypnogram(y):
    y = y.float().cpu()

    # create time variable
    time = (torch.arange(y.size(0))*30)/3600

    # rearange order
    y2 = torch.zeros_like(y)
    y2[y==0] = 4
    y2[y==1] = 2
    y2[y==2] = 1
    y2[y==3] = 0
    y2[y==4] = 3

    # create REM line
    y2_R = y2.clone()
    t_R = time.clone()

    y2_R[y2!=3] = np.nan
    t_R[y2!=3] = np.nan

    # plot
    plt.plot(time,y2,c='k')
    plt.plot(t_R,y2_R,c='r',linewidth=4)
    plt.grid()
    plt.yticks([0,1,2,3,4],['N3','N2','N1','REM','Wake'])
    plt.xlim(0,10)
    plt.ylim(-0.2,4.2)
    
def plot_pre_ann(mcdo,punet,a):
    
    # for i in range(6):
    #     y[i] = [x*(x!=7) for x in y[i]]
    #     a[i] = [x*(x!=7) for x in a[i]]
    
    plt.figure(figsize=[7,5])
    for i in range(6):
        plt.subplot(6,3,i*3+1)
        plt.plot(mcdo[i][300:500],c='blue')
        plt.ylim(0,4)
        plt.axis('off')
        
        
        plt.subplot(6,3,i*3+2)
        plt.plot(punet[i][300:500], c='green')
        plt.ylim(0,4)
        plt.axis('off')
        
        plt.subplot(6,3,i*3+3)
        plt.plot(a[:,i,300:500].cpu().reshape(200),c='red')
        plt.ylim(0,4)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def y_y2(y,length=200):
    # rearange order
    y = y.float().cpu()
    time = (torch.arange(length)*30)/3600
    # rearange order
    y2 = torch.zeros_like(y)
    y2[y==0] = 4
    y2[y==1] = 2
    y2[y==2] = 1
    y2[y==3] = 0
    y2[y==4] = 3

    # create REM line
    y2_R = y2.clone()
    t_R = time.clone()

    y2_R[y2!=3] = np.nan
    t_R[y2!=3] = np.nan
    return y2, y2_R, t_R

def plot_pre_ann_individual(mcdo, punet, a, path):

    # create time variable
    time = (torch.arange(200)*30)/3600


    for i in range(6):
        plt.figure(figsize=[2,1.5])
        y2, y2_R, t_R = y_y2(mcdo[i][300:500])
        plt.plot(time,y2,c='b')
        plt.plot(t_R,y2_R,c='r',linewidth=4)

        plt.yticks([0,1,2,3,4],['N3','N2','N1','R','W'],fontsize=6)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

        plt.ylim(-0.2,4.2)
        plt.savefig(os.path.join(path,'mcdo_'+str(i)+'.pdf'))
        plt.close()
        
        
        plt.figure(figsize=[2,1.5])
        y2, y2_R, t_R = y_y2(punet[i][300:500])
        plt.plot(time,y2,c='g')
        plt.plot(t_R,y2_R,c='r',linewidth=4)

        plt.yticks([0,1,2,3,4],['N3','N2','N1','R','W'],fontsize=6)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        plt.ylim(-0.2,4.2)
        plt.savefig(os.path.join(path,'punet_'+str(i)+'.pdf'))
        plt.close()
        
        plt.figure(figsize=[2,1.5])
        y2, y2_R, t_R = y_y2(a[:,i,300:500].cpu().reshape(200))
        plt.plot(time,y2,c='k')
        plt.plot(t_R,y2_R,c='r',linewidth=4)

        plt.yticks([0,1,2,3,4],['N3','N2','N1','R','W'],fontsize=6)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        plt.ylim(-0.2,4.2)
        plt.savefig(os.path.join(path,'rl_'+str(i)+'.pdf'))
        plt.close()
        
        plt.figure(figsize=[5,1.5])
        y2, y2_R, t_R = y_y2(a[:,i,100:600].cpu().reshape(500),length=500)
        time_ = (torch.arange(500)*30)/3600
        plt.plot(time_,y2,c='k')
        plt.plot(t_R,y2_R,c='r',linewidth=4)

        plt.yticks([0,1,2,3,4],['N3','N2','N1','R','W'],fontsize=6)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        plt.ylim(-0.2,4.2)
        plt.savefig(os.path.join(path,'rl_'+str(i)+'.png'))
        plt.close()
            


    
    
if __name__ ==  "__main__":
    loc = os.path.abspath(__file__)[:60]+'\\records'
    

    with open(os.path.join(loc, 'plot_data'), 'rb') as fp:
        [dl, rl, mcdo] = pickle.load(fp)
    
    device = "cuda"
    path = Path(os.path.abspath(__file__)).parent.parent.absolute()
    para_path = os.path.join(path,'para')
    model = ProbabilisticUnet(device, input_channels=7, num_classes=5, num_filters=[4,8,16], latent_dim=2, beta=10.0).to(device)
    model.load_state_dict(torch.load(os.path.join(para_path,'punet_ISRC.pth')))
    
    punet = []
    with torch.no_grad():
        for val_idx in range(6):
            rl_ = rl[:,random.randint(0,5),:].unsqueeze(1)
            segm = torch.repeat_interleave(rl_, 30*128,dim=2)
            model.forward(dl, segm, training = True)
            prior = model.posterior_latent_space.rsample()
            cur_pre = model.reconstruct(use_posterior_mean=False, calculate_posterior=False, z_posterior=prior)
            punet.append(cur_pre.cpu().argmax(1).reshape(1610))
            
    path = Path(os.path.abspath(__file__)).parent.absolute()
    path = os.path.join(path,'figures')
    plot_pre_ann_individual(mcdo, punet, rl, path)
    
    pass