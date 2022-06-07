# %% imports
# standard libs
import numpy as np
import mne
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# %% extract label data
def remove_trailing_sevens(scoring):
    while True:
        if scoring[-1] != 7:
            return scoring
        else:
            scoring = np.delete(scoring,-1)
            
def extract_labels(file,max_epochs):
    # get the file as collumns
    with open(file) as f:
        collumns = zip(*[line for line in f])
    
    # extract only the usefull collumns
    leading_sevens = 0
    scoring_here = []
    for i,collumn in enumerate(collumns):
        if i % 3 == 0:
            scoring = collumn[1:]
            scoring_as_array = remove_trailing_sevens(np.asarray(scoring,dtype=int))
            # get the leading sevens from scorer 1
            if i == 0:
                leading_sevens = np.min(np.where(scoring_as_array != 7))
        
            # offest scorers 3 and 4 by the amount of leading sevens
            if i == 2*3 or i == 3*3:
                offset = 0
            else:
                offset = leading_sevens
             
            # fit all scorings into the same amount
            scoring_as_array_full_length = np.zeros(max_epochs,dtype=int)+7
            scoring_as_array_full_length[0:len(scoring_as_array)-offset] = scoring_as_array[offset:]
            scoring_as_array_full_length[scoring_as_array_full_length==5] = 4
            scoring_here.append(scoring_as_array_full_length)
            
    # stack the scorings of the 6 different scorers
    scoring_here = np.vstack(scoring_here)
    epochs_before_start = leading_sevens
    return scoring_here, epochs_before_start

# %% extract edf data
def extract_edf(file,max_epochs,epochs_before_start):
    # read edf
    edf_data = mne.io.read_raw_edf(file,verbose=False)
    
    # extract raw data 
    raw_data = edf_data.get_data()
    channels = edf_data.ch_names
    no_samples = raw_data.shape[1]
    
    # extract all the channels of interest
    C3 = raw_data[[channels.index(i) for i in channels if 'C3' in i][0],:]
    C4 = raw_data[[channels.index(i) for i in channels if 'C4' in i][0],:]
    
    O1 = raw_data[[channels.index(i) for i in channels if 'O1' in i][0],:]
    O2 = raw_data[[channels.index(i) for i in channels if 'O2' in i][0],:]
    
    LOC = raw_data[[channels.index(i) for i in channels if 'LOC' in i][0],:]
    ROC = raw_data[[channels.index(i) for i in channels if 'ROC' in i][0],:]
    EMG = raw_data[[channels.index(i) for i in channels if 'EMG' in i][0],:]

    data = np.vstack((C3,C4,O1,O2,LOC,ROC,EMG))
    
    # log-normalize to 95th percentile
    normalization_factors = np.percentile(data,95,axis=1)
    normalization_factors_expanded = np.repeat(np.expand_dims(normalization_factors,1),no_samples,axis=1)
    data_scaled = np.sign(data) * np.log(np.abs(data)/normalization_factors_expanded + 1)
    
    # remove the epochs before start
    start = epochs_before_start*30*128
    data_windowed = np.zeros((7,max_epochs*30*128))
    data_windowed[:,:(no_samples-start)] = data_scaled[:,start:]
    
    return data_windowed
    

# %% if name main
if __name__ == "__main__":
    # %% parameters
    loc = "C:/Users/cilli/Codes/Punet/IS-RC/raw"
    max_epochs = 1610
    visualize = False

    # %% loop over all data
    Path(loc[:-3]+"preprocessed").mkdir(parents=True, exist_ok=True)
    file_names = natsorted(glob.glob(f"{loc}/*.edf"))
    
    i = 1
    for file in tqdm(file_names):
        # exlcuded file due to missing scorings
        if "AL_36_112108" in file:
            continue
        
        # extract label data
        STA_file = file.replace('edf','STA')
        scoring, epochs_before_start = extract_labels(STA_file,max_epochs)
        
        # extract edf data
        data = extract_edf(file,max_epochs,epochs_before_start)
        
        # save the data
        save_name = loc[:-3]+"preprocessed/"
        np.save(f"{save_name}data_{i}.npy",data)
        np.save(f"{save_name}labels_{i}.npy",scoring)
        i+=1
    
    # %% visualize an example
    if visualize == True:
        begin = (0)*128
        end = begin + 10*128
        
        time = np.arange(end-begin)/128
        
        plt.figure()
        for i in range(7):
            plt.plot(time,data[i,begin:end] - i*4,'k')
        plt.grid()
        plt.yticks([-24,-20,-16,-12,-8,-4,0],['EMG','ROC','LOC','O2','O1','C4','C3'])
        plt.xlabel('time [s]')
        plt.show()