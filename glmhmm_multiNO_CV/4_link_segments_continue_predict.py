import random
from sklearn.utils.validation import check_random_state
import pickle
import numpy as np
import scipy.stats
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from pyGLMHMM_continue.GLMHMM_CV_train_predict import GLMHMMEstimator
#from pyGLMHMM_continue.emitGenerate_X import _emit_generate
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
##random sampling center point for classes: 1-5
#halfLen=100##den_deconv, denoised deconv produces continue 0 longer than 200, leading training error
halfLen=200
if halfLen==100:
    randNums=[200,200,200,200,200,100];#for 200-segments


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


##get test set sample indexes
def get_test_indexes(CV_random,folds,NO_test,X):
    np.random.seed(CV_random)
    rp_data = np.random.permutation(len(X))
    part=int(np.round(len(rp_data)/folds))
    if NO_test==1:
        test_data_=rp_data[0:part]
        #train_data_=rp_data[part:]
    elif NO_test==folds:
        startI=part*(NO_test-1)
        test_data_=rp_data[startI:]
        #train_data_=rp_data[0:startI]
    else:
        startI=part*(NO_test-1)
        endI=part*NO_test
        test_data_=rp_data[startI:endI]
        #train_data_=np.hstack((rp_data[0:startI],rp_data[endI:]))
    return test_data_


###parameters
num_states = 2
random_state=1000#
num_samples = len(output_symb);
num_emissions = len(randNums)+1 # for all classes
###default
num_feedbacks = 1
num_filter_bins = 70
filter_offset = 1
num_steps = 1 ## emit fitting epoch
#num_feedbacks*num_filter_bins+filter_offset=channels or neurons
max_iter=1
CV_random=200

##build Non_Cross_Validation glmhmm models, no effect on prediction
##or loading trained model
#estimator = GLMHMMEstimator(random_state=random_state, max_iter=max_iter, num_samples = num_samples,
estimator = GLMHMMEstimator(random_state=None, max_iter=max_iter, num_samples = num_samples,
num_states = num_states, num_emissions = num_emissions, num_feedbacks = num_feedbacks,
num_filter_bins = num_filter_bins, num_steps = num_steps,
filter_offset = filter_offset,
CV_random=CV_random)
##initialization
estimator.init(output_stim)

##load model and data
out_dir='/DATA/taoxm/mouse/result/glmhmm_0514/CV5_'+str(2*halfLen)+'_0-4no5-8ej'
##load data
folds=5
fileName='states'+str(num_states)+'_'+str(folds)+'folds'+'.npy'
#out=np.load(os.path.join(out_dir,fileName),allow_pickle=True)[range(folds)]
fd=5
out=np.load(os.path.join(out_dir,fileName),allow_pickle=True)[range(fd)]
#acc_tests=[[]]*folds
#acc_tests_1value=[[]]*folds
acc_all=[]
acc_tests=[[]]*fd
acc_tests_1value=[[]]*fd
#for NO_test in range(1,folds+1):
for NO_test in range(1,fd+1):
  print(NO_test)
  test_inds=get_test_indexes(CV_random,folds,NO_test,output_symb)
  test_symb=[output_symb[test_inds[i]] for i in range(len(test_inds))]
  test_stim=[output_stim[test_inds[i]] for i in range(len(test_inds))]##many segments in a list
  ##prediction of test set
  estimator.emit_w_=out[NO_test-1][-1]['emit_w']
  estimator.trans_w_=out[NO_test-1][-1]['trans_w']
  [y_pred,y_lik_pred,z_pred,z_lik_pred]=estimator.predict(test_stim)
  ## Accuracy
  for i in range(len(test_symb)):
    acc_tests[NO_test-1].append(sum(test_symb[i]==y_pred[i])/len(y_pred[i]))
  acc_all.extend(acc_tests[NO_test-1])
  acc_tests_1value[NO_test-1]=np.mean(acc_tests[NO_test-1])


acc_mean=round(np.mean(acc_all),2)
acc_sd=round(np.std(acc_all),2)
####predict continue recording neuron activity
###choose the best training fold model to predict the all time segments
use_model_ind=np.argmax(acc_tests_1value)
##prediction of test set
estimator.emit_w_=out[use_model_ind][-1]['emit_w']
estimator.trans_w_=out[use_model_ind][-1]['trans_w']

##segment the whole recording:wq

segLen=halfLen*2
index_all=[]
for ni in range(0,len(symb)-segLen,segLen-1):
  indUse=range(segLen) if ni==0 else range(1,segLen) 
  index_all.extend([ni+i for i in indUse]) 

##prediction
pred_sum=len(index_all)
symb_pred=np.zeros(pred_sum)
symb_lik=np.zeros((num_emissions,pred_sum))
state_pred=np.zeros(pred_sum)
state_lik=np.zeros((num_states,pred_sum))
###
z0=np.nan #for first segment (randomization actually)
z0_lik=np.nan
for ni in range(0,len(symb)-segLen,segLen-1):
  print(ni)
  ##overlap 1 point
  segment=stim[:,ni:(ni+segLen)]
  ##predict one by one segment
  [y_pred,y_lik_pred,z_pred,z_lik_pred]=estimator.predict_one_segment(segment,z0,z0_lik)
  indUse=range(segLen) if ni==0 else range(1,segLen)
  indInAll=[ni+i for i in indUse]
  #index_all.extend(indAnAll)
  ##store prediction
  symb_pred[indInAll] = y_pred[indUse]
  symb_lik[:,indInAll] = y_lik_pred[:,indUse]
  state_pred[indInAll] = z_pred[indUse]
  state_lik[:,indInAll] = z_lik_pred[:,indUse]
  ## the first state of next seg uses the last of early seg as begin
  z0=z_pred[-1]
  z0_lik=z_lik_pred[:,-1]


symb_real=symb[index_all]
##save data
#pred_dir='/DATA/taoxm/mouse/result/glmhmm_0514/CV5_400/prediction'
pred_dir=out_dir+'/prediction'
out_dir2=os.path.join(pred_dir,'states'+str(num_states))
if not os.path.exists(out_dir2): os.mkdir(out_dir2)
np.savez(os.path.join(out_dir2,'states'+str(num_states)+'_'+str(folds)+'folds_max'+str(use_model_ind)+'_continue.pred'), \
symb_pred =symb_pred,symb_lik=symb_lik,state_pred=state_pred,state_lik=state_lik, symb_real=symb_real)

###
