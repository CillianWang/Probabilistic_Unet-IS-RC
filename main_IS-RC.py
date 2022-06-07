import os
from pathlib import Path
from os import path
import pathlib
import torch
import IS_RC_preprocessing.dataloaders as dataloader
import torch.optim as optim
import numpy


from models.Punet import ProbabilisticUnet
from models.usleep_lstm import UNet
import sys

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#####################################################
# arguments:
log_print = False
load = True
MCDO = True
# momentum = 0.9

args_windows = {
    'sys' : 'windows',
    'log' : log_print,
    'load': load,
    'nums_of_subjects': 69,
    'block': None,
    'batch': 1,
    'epoch': 20,
    'lr': 0.0003,
    'split': [0.7,0.2,0.1],
    'divide': 120 # take 420 min of data
}
args_linux = {
    'sys' : 'linux',
    'log' : log_print,
    'load': load,
    'nums_of_subjects': 69,
    # 279 for divide=360
    'block': None,
    'batch': 1,
    'epoch': 20, 
    'lr': 0.0003,
    'split': [0.7,0.2,0.1],
    'divide': 360 # take 420 min of data
}

# npz_file_path:
npz_path_windows = r'C:\Users\cilli\Codes\Punet\IS-RC\preprocessed'
npz_path_linux = '/home/Utime/Punet/IS-RC/preprocessed'

# check platform:
if path.exists(npz_path_windows): 
    npz_path, args = npz_path_windows, args_windows
    print('Running on Windows')
elif path.exists(npz_path_linux): 
    npz_path, args = npz_path_linux, args_linux
    print('Running on linux')
else: print('no file found')

# file path:
if args['log']:
    if args['sys']=='windows':
        sys.stdout = open(r'C:\Users\cilli\Codes\Punet\record\log.txt', 'w')
    elif args['sys']=='linux':
        sys.stdout = open('/home/Utime/Punet/record/log.txt', 'w')


if __name__ == "__main__":
    # dataloader:
    train_dataloader, test_dataloader = dataloader.loader(args, npz_path)
    path = Path(os.path.abspath(__file__)).parent.absolute()
    para_path = os.path.join(path,'para')
    args['name'] = 'usleep'
    args['training'] = False
    args['mcdo'] = True
        
    
    
    # model:
    if args['name']=='punet':
        model = ProbabilisticUnet(device, input_channels=7, num_classes=5, num_filters=[4,8,16], latent_dim=2, beta=10.0).to(device)
        model.load_state_dict(torch.load(os.path.join(para_path,'punet_ISRC.pth')))
        from train_punet_IS_RC import train
    
    elif args['name']=='usleep':
        model = UNet(n_channels=7, n_classes=5, classifier=True).to(device)
        model.load_state_dict(torch.load(os.path.join(para_path,'usleep.pth')))
        from train_usleep import train
        
    elif args['name']=='BNN':
        model = UNet(n_channels=7, n_classes=5, classifier=True).to(device)
        model.load_state_dict(torch.load(os.path.join(para_path,'usleep.pth')))
        from train_usleep import train
    

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    
    scheduler_RLRop = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    
    # if using usleep only:
    if args['name']=='usleep': train(data=train_dataloader, model=model, args=args, device=device, optimizer=optimizer, scheduler=scheduler, scheduler_RLRoP=scheduler_RLRop, eval_data=test_dataloader)
    
    # # if using punet:
    if args['name']=='punet': train(model, train_dataloader, test_dataloader, args, device, optimizer=optimizer, scheduler=scheduler)
 
    
    if args['log']:
        sys.stdout.close()


