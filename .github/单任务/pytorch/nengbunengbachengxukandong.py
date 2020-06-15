import os
import sys
import pickle
import matplotlib
matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt


statistics_path = '/home/fangjunyan/task5-多任务0.0/work_space/statistics/main/logmel_64frames_64melbins/taxonomy_level=coarse/holdout_fold=1/Cnn_9layers_AvgPooling/'
statistics_name = 'validate_statistics.pickle'
plt_micro_auprc = []
plt_micro_f1 = []
plt_macro_auprc = []
plt_x = []
statistic = ['micro_auprc', 'macro_auprc', 'micro_f1']
with open(statistics_path + statistics_name, 'rb') as file:
    a_dict1 = pickle.load(file)

for i in range(0, len(a_dict1)):
    plt_micro_auprc.append(a_dict1[i]['micro_auprc'])
    plt_x.append(a_dict1[i]['iteration'])
    plt_macro_auprc.append(a_dict1[i]['macro_auprc'])
    plt_micro_f1.append(a_dict1[i]['micro_f1'])
# print(a_dict1)
plt.figure(1)
plt.suptitle('statistics', fontsize='18')
plt.plot(plt_x, plt_micro_auprc, 'r-', label='micro_auprc')
plt.plot(plt_x, plt_macro_auprc, 'g-', label='macro_auprc')
plt.plot(plt_x, plt_micro_f1, 'b-', label='micro_f1')
plt.legend(loc='best')
plt.savefig('/home/fangjunyan/count/spacetimejiangwei1.jpg')

