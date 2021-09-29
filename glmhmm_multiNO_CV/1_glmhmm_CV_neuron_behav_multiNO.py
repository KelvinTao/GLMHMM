import random
import numpy as np
import scipy.stats
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from pyGLMHMM1.GLMHMM_CV_adapt import GLMHMMEstimator
import scipy.io as sio
import os
from sklearn import preprocessing

##loading data
path='/DATA/taoxm/mouse/data/Esr222/0514';
#behavior: symb, activity: act
filePath =os.path.join(path,'activity_behav_0-4no_5-8ej.npz');
data=np.load(filePath);

##36642 points
stim = data['act']##neuron activity
#normalization
stim = preprocessing.scale(stim,axis=1, with_mean=True, with_std=True, copy=True)
symb = data['symb']##behavior symbol
##random sampling center point for classes: 0-4 (no behavior), 5-8 (mount2ejaculation)
#positive mounting intromission ejaculation
halfLen=200
#if halfLen==100:
#    randNums=[200,200,200,200,200];#for 200-segments

if halfLen==200:
    randNums=[30,30,30,50,100,100,100,100];#for 400-segments  ##no 0


random.seed(100)
sampleInds=[];
for bi in range(len(randNums)):
    i_p=np.argwhere(symb==(bi+1))
    sampleInds.extend(random.sample([i_p[i][0] for i in range(i_p.shape[0])],randNums[bi]))

##
# 100 ms interval 200 time stamps-20 s; 
## two mating process
output_symb=[];
output_stim=[];
for ni in range(len(sampleInds)):
    midInd=sampleInds[ni];
    output_symb.append(symb[(midInd-halfLen):(midInd+halfLen)].T)
    output_stim.append(stim[:,(midInd-halfLen):(midInd+halfLen)])

###parameters
random_state=1000#
num_samples = len(output_symb);
num_emissions = len(randNums)+1 # for all classes
###default
num_feedbacks = 1
num_filter_bins = 70
filter_offset = 1
num_steps = 1 ## emit fitting epoch
#num_feedbacks*num_filter_bins+filter_offset=channels or neurons
##fitting model
max_iter=1000
##build Cross_Validation models
out_dir='/DATA/taoxm/mouse/result/glmhmm_0514/CV5_'+str(2*halfLen)+'_0-4no5-8ej'
if not os.path.exists(out_dir):
  os.mkdir(out_dir)


folds=5
for num_states in range(2,7):
  output=[]
  for CV_i in range(1,folds+1): #from 1
    estimator = GLMHMMEstimator(random_state=None, max_iter=max_iter, num_samples = num_samples,
    num_states = num_states, num_emissions = num_emissions, num_feedbacks = num_feedbacks,
    num_filter_bins = num_filter_bins,filter_offset = filter_offset,
    num_steps = num_steps, ##The number of steps taken in the maximization step of the EM algorithm for calculating the emission matrix
    evaluate=False,get_error_bars=False, ##only one cycle for evaluate=True
    generate=False,analog_flag=False, ##analog symble
    L2_smooth=True,
    anneal_lambda=True,auto_anneal=True,
    cross_validate=True,CV_regularize=True,
    folds=folds,NO_test=CV_i,##
    CV_random=200)
    ##
    output.append(estimator.fit(output_stim, output_symb, []))
    ##save results
    fileName='states'+str(num_states)+'_'+str(folds)+'folds'
    np.save(os.path.join(out_dir,fileName),output)
    ##save model
    import pickle
    with open(os.path.join(out_dir,fileName+'_last_model.pk'), 'wb') as f:
      pickle.dump(estimator, f)


