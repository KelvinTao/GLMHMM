##
import numpy as np
import os
import matplotlib.pyplot as plt
import time as tim
import scipy.io as sio
##get intruder time points
data=sio.loadmat('/DATA/taoxm/mouse/data/Esr222/0514/Esr2220514_neuron.mat');
intrud=data['neuron'][0][0][2][1][0]
intr_points= np.round(intrud*10)
##
intr_use=[0]
for i in range(intr_points.shape[0]):
  for j in range(intr_points.shape[1]):
    intr_use.append(int(intr_points[i,j]))


##
halfLen=200
folds=5
num_states=2
use_model_ind=0
##
out_dir='/DATA/taoxm/mouse/result/glmhmm_0514/CV5_400/prediction/states'+str(num_states)
fileName='states'+str(num_states)+'_'+str(folds)+'folds_max'+str(use_model_ind)+'_continue.pred.npz'
##load data
data=np.load(os.path.join(out_dir,fileName))
##
states=data['state_pred']
behav=data['symb_real']
pred=data['symb_pred']

##save as matlab data
dataM={}
dataM['state_pred']=data['state_pred']
dataM['symb_real']=data['symb_real']
dataM['symb_pred']=data['symb_pred']
sio.savemat(os.path.join(out_dir,fileName+'.mat'),dataM)
##
accs=sum(behav==pred)/len(behav)
##add end index
intr_use.append(len(pred))
##plot
periods=['before_intruder1','intruder1','intruder1~intruder2','intruder2','intruder2~intruder3','intruder3','intruder3_later']
##statistics by periods--intruders
for i in range(len(periods)):
  time=range(intr_use[i],intr_use[i+1])
  print(len(time))
  ##
  st=states[time]
  bh=behav[time]
  #pred_bh=pred[time]
  
  print(bh[st==1])
  print(bh[st==0])


  ##
  fig=plt.figure(figsize=[20,20])
  ax1=fig.add_subplot(1,1,1)
  ax1.set_title("Predicted States for continued recordings "+periods[i],fontsize=20)
  ax1.plot(time, states[time], 'b--', label='Predicted States (%s)'%str(num_states))
  ax1.plot(time, behav[time], 'k-', label='Real Behaviors')
  ax1.plot(time, pred[time], 'r--', label='Predicted Behaviors')
  ax1.set_title("Behaviors and Predicted Behaviors for the whole recording")
  ax1.text(max(time)-400,6.5, 'Prediction Accuracy %.2f' % accs, ha='center', va= 'bottom',fontsize=20,color='k')##fontsize=8
  ax1.set_xlim([min(time),max(time)])
  ax1.set_ylim([-0.5,6.99])
  plt.legend(loc=2,fontsize=20)
  plt.tight_layout()
  pltFile=os.path.join(out_dir,fileName+'.state.pred.'+periods[i]+'.intruder_part'+str(i)+'.jpg')
  plt.savefig(pltFile, dpi=100)
  plt.close()


##statistics by states



##statistics by behavior types


##transition between behaviors of states
