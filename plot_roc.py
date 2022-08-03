import numpy as np
import pdb

from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pdb.set_trace()

ori_scores = np.squeeze(np.load('ori_scores.npy'), -1)
ori_labels = np.load('ori_labels.npy')

att_scores = np.squeeze(np.load('att_scores.npy'), -1)
att_labels = np.load('att_labels.npy')

sparse_scores = np.squeeze(np.load('sparse_scores.npy'), -1)
sparse_labels = np.load('sparse_labels.npy')

# original
ori_fpr, ori_tpr, ori_thersholds = roc_curve(ori_labels, ori_scores, pos_label=1)
ori_roc_auc = auc(ori_fpr, ori_tpr)

# attention
att_fpr, att_tpr, att_thersholds = roc_curve(att_labels, att_scores, pos_label=1)
att_roc_auc = auc(att_fpr, att_tpr)

# sparse
sparse_fpr, sparse_tpr, sparse_thersholds = roc_curve(sparse_labels, sparse_scores, pos_label=1)
sparse_roc_auc = auc(sparse_fpr, sparse_tpr)

plt.plot(ori_fpr, ori_tpr, '-.', color='b', label='w/o alignment: ROC (area = {0:.5f})'.format(ori_roc_auc), lw=2)
plt.plot(att_fpr, att_tpr, '--', color='r', label='attention alignment: ROC (area = {0:.5f})'.format(att_roc_auc), lw=2)
plt.plot(sparse_fpr, sparse_tpr, ':', color='g', label='sparse reconstruction alignment: ROC (area = {0:.5f})'.format(sparse_roc_auc), lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([0.9, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.legend(loc="lower right")

plt.savefig('roc.pdf')