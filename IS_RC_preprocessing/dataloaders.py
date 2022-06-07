# %% imports
# standard libs
import numpy as np
import glob
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import os
# %% Dataset
# get only one scoring for a given PSG
class IS_RC_Seperate(Dataset):
    def __init__(self,loc, train, transform=None):
        # save some local stuff
        self.loc = loc
        self.train = train
    
        # get all the sta files
        file_names = glob.glob(f"{loc}/*.npy")
        self.no_files = len(file_names)//2
        
        # reserve the last 10 recordings for testing
        self.val_skip = self.no_files - 10
        if  self.train:
            self.length = self.val_skip
        else:
            self.length = 10
        
        # multiply the length by six due to individual scorer loading
        self.length = 6*self.length
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # calculate file and scorer ids
        file_id = (idx//6)+1
        scorer_id = idx%6
        if self.train == False:
            file_id += self.val_skip
        
        # get the PSG data
        data = np.load(f"{self.loc}/data_{file_id}.npy")
        
        # get the label data
        labels = np.load(f"{self.loc}/labels_{file_id}.npy")
        labels_single_scorer = labels[scorer_id,:]
        
        return data,labels_single_scorer
  
# get all scorings for a given PSG
class IS_RC_All(Dataset):
    def __init__(self,loc, train, transform=None):
        # save some local stuff
        self.loc = loc
        self.train = train
    
        # get all the sta files
        file_names = glob.glob(f"{loc}/*.npy")
        self.no_files = len(file_names)//2
        
        # reserve the last 10 recordings for testing
        self.val_skip = self.no_files - 10
        if  self.train:
            self.length = self.val_skip
        else:
            self.length = 10
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # calculate file and scorer ids
        file_id = idx+1
        if self.train == False:
            file_id += self.val_skip
        
        # get the PSG data
        data = np.load(f"{self.loc}/data_{file_id}.npy")
        
        # get the label data
        labels = np.load(f"{self.loc}/labels_{file_id}.npy")
        
        return data,labels

# get only the scorings without a PSG
class IS_RC_Labels(Dataset):
    def __init__(self,loc, train, transform=None):
        # save some local stuff
        self.loc = loc
        self.train = train
    
        # get all the sta files
        file_names = glob.glob(f"{loc}/*.npy")
        self.no_files = len(file_names)//2
        
        # reserve the last 10 recordings for testing
        self.val_skip = self.no_files - 10
        if  self.train:
            self.length = self.val_skip
        else:
            self.length = 10
            
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # calculate file and scorer ids
        file_id = idx+1
        if self.train == False:
            file_id += self.val_skip
        
        # get the label data
        labels = np.load(f"{self.loc}/labels_{file_id}.npy")
        
        return labels

def loader(args, file_path):
    length = args['nums_of_subjects']
    train = IS_RC_Seperate(file_path, train=True)
    test = IS_RC_All(file_path, train=False)
    
    train_dataloader = DataLoader(train, batch_size=args['batch'], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=args['batch'], shuffle=True)

    return train_dataloader, test_dataloader

    

# %% testing
if __name__ == "__main__":
    loc = "D:/Datasets/IS-RC/preprocessed"
    
    dataset = IS_RC_Seperate(loc, train = True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False,drop_last=False, pin_memory=True)
    
    # %%
    for i,(data,labels) in enumerate(tqdm(dataloader)):
        1+1
    
    
    
    
    