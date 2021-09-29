##
import numpy as np
import os

##
halfLen=200
out_dir='/DATA/taoxm/mouse/result/glmhmm_0514/CV5_'+str(2*halfLen)+'_0-4no5-8ej'
num_states = 2
fileName='states'+str(num_states)+'_5folds.npy'
data=np.load(os.path.join(out_dir,'states'+str(num_states),fileName+'.pred.npz'))##, y_real_all=y_real_all, y_pred_all=y_pred_all, acc_all=acc_all)
##
if halfLen==100:
    randNums=[200,200,200,200,200,100];#for 200-segments

if halfLen==200:
    randNums=[30,30,30,50,100,100,100,100];


num_emissions = len(randNums)+1 # for all classes
##
y_real_all=data['y_real_all']
y_pred_all=data['y_pred_all']
acc_all=data['acc_all']
acc_mean=round(np.mean(acc_all),2)
acc_sd=round(np.std(acc_all),2)
##plot accuracy
import matplotlib.pyplot as plt
# matplotlib.axes.Axes.hist() 方法的接口
plt.figure()
plt.title("Accuracy histogram of the 5-fold cross-validation")
#n, bins, patches = plt.hist(x=acc_all, bins='auto', color='#0504aa',density=True,rwidth=0.85)
acc_all=[i*10 for i in acc_all]
n, bins, patches = plt.hist(acc_all, 10,histtype='bar',facecolor='b',density=True, alpha=0.75,rwidth=0.97)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Accuracy')
plt.ylabel('Frequency of sample segments')
#plt.yticks([0.0,0.5,1.0,1.5,2.0],[0, 0.05,0.10,0.15,0.20])
#plt.yticks([0.0,0.5,1.0,1.5,2.0],[0, 0.05,0.10,0.15,0.20])
plt.xticks([0,2,4,6,8,10],[0.0,0.2,0.4,0.6,0.8,1.0])
plt.text(3, 0.18, 'mean=%2s, sd=%2s'%(acc_mean,acc_sd))
#maxfreq = n.max()
# 设置y轴的上限
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.savefig(os.path.join(out_dir,'states'+str(num_states),fileName+'.sample.acc.hist.jpg'))
plt.close()

##accuracy  for each class
acc = []
for i in range(num_emissions):
    print(i)
    index=np.where(y_real_all==i)
    print(len(index[0]))
    real=y_real_all[index]
    pred=y_pred_all[index]
    print(np.sum(real==pred))
    acc.append(round(np.sum(real==pred).astype(np.float)/len(index[0]),3))

print(acc)

np.savez(os.path.join(out_dir,'states'+str(num_states),fileName+'.pred.class_acc'), class_acc=acc)
##calculate AUC of each class
from sklearn.metrics import roc_curve,auc,recall_score,accuracy_score,precision_score,f1_score,roc_auc_score
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize
#from sklearn import metrics.roc_auc_score

aucs=[]
y_p = y_pred_all
test_s = y_real_all
# Binarize the output
test_s = label_binarize(test_s, classes=[i for i in range(num_emissions)])
y_p = label_binarize(y_p, classes=[i for i in range(num_emissions)])
# micro：多分类--accuracy
# weighted：不均衡数量的类来说，计算二分类metrics的平均
# macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
precision = precision_score(test_s, y_p, average='micro')
recall = recall_score(test_s, y_p, average='micro')
f1 = f1_score(test_s, y_p, average='micro')
acc = accuracy_score(test_s, y_p)
#auc=metrics.roc_auc_score(test_s, y_p)
print("Precision_score:",precision)
print("Recall_score:",recall)
print("F1_score:",f1)
print("Accuracy_score:",acc)
# roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
# 横坐标：假正率（False Positive Rate , FPR）
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_emissions):
    fpr[i], tpr[i], _ = roc_curve(test_s[:, i], y_p[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_s.ravel(), y_p.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
aucs.append(roc_auc["micro"])
# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_emissions)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_emissions):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])


# Finally average it and compute AUC
mean_tpr /= num_emissions
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_emissions), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve of 6 behavior')
plt.legend(loc="lower right")
plt.savefig(os.path.join(out_dir,'states'+str(num_states),fileName+'.ROC.jpg'))
plt.close()

np.savez(os.path.join(out_dir,'states'+str(num_states),fileName+'.pred.aucs'), aucs=aucs,roc_auc=roc_auc,fpr=fpr,tpr=tpr)



