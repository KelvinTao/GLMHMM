import random
from sklearn.utils.validation import check_random_state
import pickle
import numpy as np
import scipy.stats
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from pyGLMHMM1.GLMHMM_CV_train_predict import GLMHMMEstimator
from pyGLMHMM1.emitGenerate_X import _emit_generate
import scipy.io as sio
import os
from sklearn import preprocessing


##loading data
path='/DATA/taoxm/mouse/data/Esr222/0514';
##bstates: behav class; traces01: stimulus
filePath =os.path.join(path,'activity_behav_0-4no_5-8ej.npz');
data=np.load(filePath);
##bahav, dec_data, wav_ca

##36642 points
#stim = data['dec_data']##neuron activity
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

##build Non_Cross_Validation glmhmm models, no effect on prediction
##or loading trained model
estimator = GLMHMMEstimator(random_state=random_state, max_iter=max_iter, num_samples = num_samples,
num_states = num_states, num_emissions = num_emissions, num_feedbacks = num_feedbacks,
num_filter_bins = num_filter_bins, num_steps = num_steps,
filter_offset = filter_offset,
CV_random=200)
##initialization
estimator.init(output_stim)

##load model and data
out_dir='/DATA/taoxm/mouse/result/glmhmm_0514/CV5_'+str(2*halfLen)+'_0-4no5-8ej'
##load data
folds=5
fileName='states'+str(num_states)+'_'+str(folds)+'folds'+'.npy'
out=np.load(os.path.join(out_dir,fileName),allow_pickle=True)#[range(folds)]
len(out)

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


##get training and test data used by training
CV_random=200 ##loading model, estimator.CV_random
folds=5  ##estimator.folds
##
y_real_all=[]
y_pred_all=[]
y_real_sample=[]
y_pred_sample=[]
z_pred_sample=[]
z_lik_pred_sample=[]
acc_all=[]
for NO_test in range(1,folds+1):
#for NO_test in range(1,2+1):
  print(NO_test)
  test_inds=get_test_indexes(CV_random,folds,NO_test,output_symb)
  test_symb=[output_symb[test_inds[i]] for i in range(len(test_inds))]
  test_stim=[output_stim[test_inds[i]] for i in range(len(test_inds))]
  ##prediction of test set
  estimator.emit_w_=out[NO_test-1][-1]['emit_w']
  estimator.trans_w_=out[NO_test-1][-1]['trans_w']
  [y_pred,y_lik_pred,z_pred,z_lik_pred]=estimator.predict(test_stim)
  ##
  for i in range(len(test_symb)):
     y_real_sample.append(test_symb[i])
     y_pred_sample.append(y_pred[i])
     z_pred_sample.append(z_pred[i])
     z_lik_pred_sample.append(z_lik_pred[i])
     for j in range(len(test_symb[i])):
         y_real_all.append(test_symb[i][j])
         y_pred_all.append(y_pred[i][j])
  ##evaluation
  ## Accuracy and AUC
  for i in range(len(test_symb)):
    acc_all.append(sum(test_symb[i]==y_pred[i])/len(y_pred[i]))


y_real_all=np.array(y_real_all)
y_pred_all=np.array(y_pred_all)
acc_mean=round(np.mean(acc_all),2)
acc_sd=round(np.std(acc_all),2)

#z_pred
print(acc_mean)
print(acc_sd)
sum(y_real_all==y_pred_all)/len(y_pred_all)
np.savez(os.path.join(out_dir,'states'+str(num_states),fileName+'.pred'), y_real_all=y_real_all, y_pred_all=y_pred_all, \
y_real_sample=y_real_sample,y_pred_sample=y_pred_sample,\
acc_all=acc_all,z_pred_sample=z_pred_sample)#,z_lik_pred_sample=z_lik_pred_sample)

##
import matplotlib.pyplot as plt
z_lik=z_lik_pred_sample[1][0]
plt.figure()
plt.plot(range(len(z_lik)),z_lik)
plt.savefig(os.path.join(out_dir,'states'+str(num_states),fileName+'.sample.z_lik_pred.jpg'))
plt.close()



