##
import numpy as np
import os
import matplotlib.pyplot as plt

##
halfLen=200
out_dir='/DATA/taoxm/mouse/result/glmhmm_0514/CV5_'+str(2*halfLen)
num_states =2
#num_states =3
fileName='states'+str(num_states)+'_5folds.npy'
data=np.load(os.path.join(out_dir,'states'+str(num_states),fileName+'.pred.npz'))##, y_real_all=y_real_all, y_pred_all=y_pred_all, acc_all=acc_all)
pltPath=os.path.join(out_dir,'states'+str(num_states),'predictions_0.7up')
if not os.path.exists(pltPath):
    os.mkdir(pltPath)


##
states=data['z_pred_sample']
behav=data['y_real_sample']
pred=data['y_pred_sample']
accs=data['acc_all']

time=range(len(states[0]))
for i in range(len(states)):
    if accs[i]<0.7:
        continue
    print(i)
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.plot(time, states[i], 'b--', label='Predicted States (%s)'%str(num_states))
    ax1.set_title("Predicted States for segment_%s"%str(i))
    ax1.plot(time, behav[i], 'k-', label='Real Behaviors')
    ax1.plot(time, pred[i], 'r--', label='Predicted Behaviors')
    ax1.set_title("Behaviors and Predicted Behaviors for segment_%s"%str(i))
    ax1.text(int(len(time)/8*7),6.5, 'Prediction Accuracy %.2f' % accs[i], ha='center', va= 'bottom',fontsize=8,color='k')
    ax1.set_xlim([0,len(time)])
    ax1.set_ylim([-0.5,6.99])
    ##
    plt.legend(loc=2,fontsize=8)
    plt.tight_layout()
    ##
    pltFile=os.path.join(pltPath,fileName+'_'+str(i)+'.pred.jpg')
    plt.savefig(pltFile, dpi=300)
    plt.close()
    #break


