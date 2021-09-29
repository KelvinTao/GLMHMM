#import random
import numpy as np
#import scipy.stats
#import scipy.ndimage.filters
import matplotlib.pyplot as plt
#from pyGLMHMM1.GLMHMM_CV_adapt import GLMHMMEstimator
#import scipy.io as sio
import os

##loading data
##build Cross_Validation models
out_dir='/DATA/taoxm/mouse/result/glmhmm_CV'
##
folds=5
ll_states=[]
states=[i for i in range(2,13)]
#states=[i for i in range(2,5)]
for num_states in states:
  ##load data
  fileName='output_states'+str(num_states)+'_'+str(folds)+'folds_part1.npy'
  out=np.load(os.path.join(out_dir,fileName),allow_pickle=True)
  ## get mean test_log_lik for each CV set 
  ll=[]
  for fi in [0]:
    ll.append(out[fi][-1]['loglik_CV']) # test_log_lik[good_ind]
  ll_states.append(np.mean(ll))


plt.figure()
plt.title("Hold-out log likelihood for states 2~12")
plt.plot(states, ll_states, "-ro")
plt.xticks(np.arange(1,14,1))
plt.xlabel("state_number")
plt.ylabel("log likelihood")
plt.savefig(os.path.join(out_dir,fileName+'.state_loglik.jpg'))
plt.show()

