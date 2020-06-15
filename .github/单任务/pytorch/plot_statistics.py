# reload a file to a variable
import matplotlib.pyplot as plt
import pickle
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

def plot_statistics(args):
    '''Draw a   statistics graph on validation pickle.

    Args:
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      taxonomy_level: 'fine' | 'coarse'
      model_type: string, e.g. 'Cnn_9layers_MaxPooling'
      iteration: int
      holdout_fold: '1', which means using validation data
                     'none', which means using model trained on all development data
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool
    '''
    # Arugments & parameters
    dataset___dir = args.dataset_dir
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    model_type = args.model_type
    iteration = args.iteration
    holdout_fold = args.holdout_fold
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    visualize = args.visualize
    filename = args.filename


    statistics_path=dataset___dir+'/statistics/main/logmel_64frames_64melbins'+\
                    '/taxonomy_level={}'.format(taxonomy_level)+\
                    '/holdout_fold={}'.format(holdout_fold)+\
                    '/Cnn_9layers_AvgPooling/'
    statistics_name='validate_statistics.pickle'
    plt_micro_auprc=[]
    plt_micro_f1=[]
    plt_macro_auprc=[]
    plt_x=[]
    statistic=['micro_auprc','macro_auprc','micro_f1']
    with open(statistics_path+statistics_name, 'rb') as file:
        a_dict1 =pickle.load(file)

    for i in range(0,len(a_dict1)):
        plt_micro_auprc.append(a_dict1[i]['micro_auprc'])
        plt_x.append(a_dict1[i]['iteration'])
        plt_macro_auprc.append(a_dict1[i]['macro_auprc'])
        plt_micro_f1.append(a_dict1[i]['micro_f1'])



    plt.figure(1)
    plt.suptitle('statistics', fontsize='18')
    plt.plot(plt_x, plt_micro_auprc, 'r-', label='micro_auprc')
    plt.plot(plt_x, plt_macro_auprc, 'g-', label='macro_auprc')
    plt.plot(plt_x, plt_micro_f1, 'b-', label='micro_f1')
    plt.legend(loc='best')
    plt.savefig('/home/liuzhuangzhuang/pycharm_P/'+time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))+'{}.jpg'.format(statistics_name))
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)

    args = parser.parse_args()

    plot_statistics()