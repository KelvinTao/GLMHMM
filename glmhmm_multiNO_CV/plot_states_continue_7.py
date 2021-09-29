##
import numpy as np
import os
import matplotlib.pyplot as plt
import time as tim

##
halfLen=200
folds=5
num_states =3
use_model_ind=0
##
out_dir='/DATA/taoxm/mouse/result/glmhmm_0514/CV5_400/prediction/states'+str(num_states)
fileName='states'+str(num_states)+'_'+str(folds)+'folds_max'+str(use_model_ind)+'_continue.pred.npz'

##load data
data=np.load(os.path.join(out_dir,fileName))
states=data['state_pred']
behav=data['symb_real']
pred=data['symb_pred']
accs=sum(behav==pred)/len(behav)

##
parts=10
length=len(states)
segLen=int(length/parts)
for i in range(parts):
    time=range(i*segLen,(i+1)*segLen)
    fig=plt.figure(figsize=[20,20])
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title("Predicted States for continued recordings part_%s"%str(i),fontsize=20)
    ax1.plot(time, states[time], 'b--', label='Predicted States (%s)'%str(num_states))
    ax1.plot(time, behav[time], 'k-', label='Real Behaviors')
    ax1.plot(time, pred[time], 'r--', label='Predicted Behaviors')
    ax1.set_title("Behaviors and Predicted Behaviors for the whole recording")
    ax1.text(max(time)-1000,6.5, 'Prediction Accuracy %.2f' % accs, ha='center', va= 'bottom',fontsize=20,color='k')##fontsize=8
    ax1.set_xlim([min(time),max(time)])
    ax1.set_ylim([-0.5,6.99])
    plt.legend(loc=2,fontsize=20)
    plt.tight_layout()
    pltFile=os.path.join(out_dir,fileName+'.state.pred.part'+str(i)+'.jpg')
    plt.savefig(pltFile, dpi=100)
    plt.close()



