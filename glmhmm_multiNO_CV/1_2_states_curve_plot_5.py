import numpy as np
import matplotlib.pyplot as plt
import os

##loading data
out_dir='/DATA/taoxm/mouse/result/glmhmm_0514/CV5_400_0-4no5-8ej'
##
folds=5
ll_states=[]
max_state=3
states=[i for i in range(2,max_state+1)]
for num_states in states:
  ##load data
  fileName='states'+str(num_states)+'_'+str(folds)+'folds.npy'
  out=np.load(os.path.join(out_dir,fileName),allow_pickle=True)
  ## get mean test_log_lik for each CV set 
  ll=[]
  for fi in range(folds):
    ll.append(out[fi][-1]['loglik_CV']) # test_log_lik[good_ind]
  ll_states.append(np.mean(ll))


#plt.figure(figsize=(a, b), dpi=dpi)
plt.figure(dpi=300)
plt.title("Cross-Validation log likelihood for states 2~%s"%str(max_state))
plt.plot(states, ll_states, "-ro")
plt.xticks(np.arange(1,8,1))
plt.xlabel("state_number")
plt.ylabel("log likelihood")
plt.savefig(os.path.join(out_dir,fileName+'.state_loglik.jpg'))
plt.show()
