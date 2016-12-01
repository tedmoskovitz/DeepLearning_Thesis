
# coding: utf-8

# In[1]:

import numpy as np
import scipy.io as spio 
import pdb

Tsp = spio.loadmat('/Users/TedMoskovitz/Thesis/fullfieldRGCdata_JN05/Tsp_expt4.mat')['Tsp']
Stim = spio.loadmat('/Users/TedMoskovitz/Thesis/fullfieldRGCdata_JN05/Stim_expt4.mat')['Stim']


# In[16]:

# pre-process data

# if classification problem
def get_flicker(num_classes=4): 
    cell_num = 9
    cell_ind = cell_num - 1

    dtSp = 1. # bin width in seconds for spike counts 
    stim_bin = 20 # number of stim points back that may have 'influenced' spikes in bin

    T = Stim.shape[0]
    spikes = np.asarray(np.histogram(Tsp[0, cell_ind], np.arange(dtSp/2, T+dtSp, dtSp))[0]).T

    binned_stims = np.zeros((spikes.shape[0]-stim_bin, stim_bin))
    for i in xrange(stim_bin, spikes.shape[0]):
       # if len(Stim[i+stim_bin : i]) >= stim_bin:
       binned_stims[i-stim_bin, :] = Stim[i-stim_bin : i].T

    clipped_Stim = binned_stims#binned_stims[stim_bin:-stim_bin,:]
    #clipped_Spk = spikes[stim_bin:-stim_bin]
    clipped_Spk = spikes[stim_bin:]
    
    if num_classes > 1:
        # binarize 
        #clipped_Spk[clipped_Spk > 1] = 1
        # convert to one-hot
        N = clipped_Spk.shape[0]
        #num_classes = 4
        oh_Spk = np.zeros((N, num_classes))
        #pdb.set_trace()
        for i in xrange(N):
            oh_Spk[i, clipped_Spk[i]] = 1
    else:
        oh_Spk = clipped_Spk

    num_train = 37000
    num_val = 5000
    num_test = 10500
    '''
    ones = np.where(oh_Spk[:,1] == 1.)[0]
    n_ones = len(ones)
    twos = np.where(oh_Spk[:,2] == 1.)[0]
    n_twos = len(twos)
    threes = np.where(oh_Spk[:,3] == 1.)[0]
    n_threes = len(threes)
    zeros = np.where(oh_Spk[:,0] == 1.)[0][:n_ones]
    n_zeros = len(zeros)

    ind_array = np.hstack((threes, twos, ones, zeros))
    '''
    tot = num_train + num_val + num_test
    ind_array = np.asarray(range(tot))
    np.random.shuffle(ind_array)
    
    train_mask = ind_array[:num_train]
    val_mask = ind_array[num_train : num_train + num_val]
    test_mask = ind_array[num_train + num_val : num_train + num_val + num_test]

    # mean subtraction
    clipped_Stim -= np.mean(clipped_Stim, axis=0)

    X_train = clipped_Stim[train_mask, :]
    y_train = oh_Spk[train_mask]

    X_val = clipped_Stim[val_mask, :]
    y_val = oh_Spk[val_mask]

    X_test = clipped_Stim[test_mask, :]
    y_test = oh_Spk[test_mask]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# if you want a continuous value
'''
def get_continuous():
    cell_num = 9
    cell_ind = cell_num - 1

    dtSp = 1. # bin width in seconds for spike counts 
    stim_bin = 20 # number of stim points back that may have 'influenced' spikes in bin

    T = Stim.shape[0]
    spikes = np.asarray(np.histogram(Tsp[0, cell_ind], np.arange(dtSp/2, T+dtSp, dtSp))[0]).T

    binned_stims = np.zeros((spikes.shape[0], stim_bin))
    for i in xrange(stim_bin, spikes.shape[0]+stim_bin):
        if len(Stim[i: i+stim_bin]) >= stim_bin:
            binned_stims[i, :] = Stim[i: i+stim_bin].T

    clipped_Stim = binned_stims[stim_bin:-stim_bin,:]
    clipped_Spk = spikes[stim_bin:-stim_bin]
    
    # mean subtraction
    clipped_Stim -= np.mean(clipped_Stim, axis=0)
    
    train_num = 50000
    val_num = 5000
    test_num = 20000
'''
    
    
    
  

#get_continuous()


# In[17]:

# testing
#X_train, y_train, X_val, y_val, X_test, y_test = get_flicker(num_classes=3)

#print float(len(np.where(y_train[:,1] == 1.)[0])) / y_train.shape[0]
#print float(len(np.where(y_test[:,1] == 1.)[0])) / y_test.shape[0]


# In[ ]:



