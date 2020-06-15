import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import random
matplotlib.use('Agg')

from utilities import (create_folder, get_filename, create_logging, 
    load_scalar, get_labels, write_submission_csv)
from data_generator import DataGenerator, TestDataGenerator
from models import Cnn_5layers_AvgPooling, Cnn_9layers_MaxPooling, Cnn_9layers_AvgPooling, Cnn_13layers_AvgPooling, Pre_Cnn14, Pre_Cnn10,Duo_Cnn_9_drop,Duo_Cnn_621_drop,Duo_Cnn_9layers_AvgPooling,Duo_Cnn_621layers_AvgPooling,Space_Duo_Cnn_9_Avg,Space_Duo_Cnn_9_2max,Space_Duo_Cnn_9_2biaoqian
from losses import binary_cross_entropy ,binary_cross_entropy1
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu, forward
import config
def seed_everything(seed):#设定随机数2020
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




def train(args):
    '''Training. Model will be saved after several iterations. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      taxonomy_level: 'fine' | 'coarse'
      model_type: string, e.g. 'Cnn_9layers_MaxPooling'
      holdout_fold: '1' | 'None', where '1' indicates using validation and 
          'None' indicates using full data for training
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    model_type = args.model_type
    holdout_fold = args.holdout_fold
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename
    plt_x = []
    plt_y = []
    T_max=300
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    max_iteration = 10      # Number of mini-batches to evaluate on training data
    reduce_lr = True
    
    labels = get_labels(taxonomy_level)
    classes_num = len(labels)

    def mixup_data(x1, x2, y, alpha=1.0, use_cuda=True):  # 数据增强，看下那个博客
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)  # 随机生成一个（1,1）的张量
        else:
            lam = 1
        #
        batch_size = x1.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()  # 给定参数n，返回一个从0到n-1的随机整数序列
        else:
            index = torch.randperm(batch_size)  # 使用cpu还是gpu

        mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
        mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]  # 混合数据
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    train_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train.h5')
        
    validate_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'validate.h5')
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train.h5')
        
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_folder(checkpoints_dir)
    
    _temp_submission_path = os.path.join(workspace, '_temp_submissions', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type, '_submission.csv')
    create_folder(os.path.dirname(_temp_submission_path))
    
    validate_statistics_path = os.path.join(workspace, 'statistics', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type, 
        'validate_statistics.pickle')
    create_folder(os.path.dirname(validate_statistics_path))
    loss_path = os.path.join(workspace, 'loss',
                                            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                            'taxonomy_level={}'.format(taxonomy_level),
                                            'holdout_fold={}'.format(holdout_fold), model_type

                                            )
    create_folder(loss_path)
    
    annotation_path = os.path.join(dataset_dir, 'annotations.csv')
    
    yaml_path = os.path.join(dataset_dir, 'dcase-ust-taxonomy.yaml')
    
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type)

    create_logging(logs_dir, 'w')
    logging.info(args)

    if cuda:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    Model = eval(model_type)
    model = Model(classes_num)
    logging.info(" Space_Duo_Cnn_9_Avg  多一层 258*258 不共用FC，必须带时空标签 用loss 监测，使用去零one hot ")



    if cuda:
        model.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0., amsgrad=True)

    logging.info('model parm：{} '.format(sum(param.numel() for param in model.parameters())))
    #计算模型参数量

    # Data generator
    data_generator = DataGenerator(
        train_hdf5_path=train_hdf5_path, 
        validate_hdf5_path=validate_hdf5_path, 
        holdout_fold=holdout_fold, 
        scalar=scalar, 
        batch_size=batch_size)
    
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        taxonomy_level=taxonomy_level, 
        cuda=cuda, 
        verbose=False)
        
    # Statistics
    validate_statistics_container = StatisticsContainer(validate_statistics_path)
    
    train_bgn_time = time.time()
    iteration = 0
    best_inde={}
    best_inde['micro_auprc']=np.array([0.0])
    best_inde['micro_f1']=np.array([0.0])
    best_inde['macro_auprc']=np.array([0.0])
    best_inde['average_precision']=np.array([0.0])
    best_inde['sum'] =best_inde['micro_auprc']+best_inde['micro_f1']+best_inde['macro_auprc']
    last_loss1 = []
    last_loss2 = []
    last_loss = []
    best_map=0
    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():
        
        # Evaluate
        if iteration % 200 == 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}, {} level statistics:'.format(
                iteration, taxonomy_level))

            train_fin_time = time.time()

            # Evaluate on training data
            if mini_data:
                raise Exception('`mini_data` flag must be set to False to use '
                    'the official evaluation tool!')
            
            train_statistics = evaluator.evaluate(
                data_type='train', 
                max_iteration=None)
            if iteration > 5000:
                if best_map<np.mean(train_statistics['average_precision']):
                    best_map=np.mean(train_statistics['average_precision'])
                    logging.info('best_map= {}'.format(best_map))
                    # logging.info('iter= {}'.format(iteration))
                    checkpoint = {
                        'iteration': iteration,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'indicators': train_statistics}
                    checkpoint_path = os.path.join(
                        checkpoints_dir, 'best7.pth')
                    torch.save(checkpoint, checkpoint_path)
                    logging.info('best_models saved to {}'.format(checkpoint_path))



            # Evaluate on validation data
            if holdout_fold != 'none':
                validate_statistics = evaluator.evaluate(
                    data_type='validate', 
                    submission_path=_temp_submission_path, 
                    annotation_path=annotation_path, 
                    yaml_path=yaml_path, 
                    max_iteration=None)
                    
                validate_statistics_container.append_and_dump(
                    iteration, validate_statistics)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()



        # Reduce learning rate
        if reduce_lr and iteration % 200 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        batch_data2_dict = batch_data_dict.copy()
        n=[]

        for i,l in enumerate(batch_data2_dict['coarse_target']):
            k = 0
            for j in range(0, 8):
                if l[j] > 0.6:
                    l[j] = 1
                else:
                    l[j] = 0
                    k += 1
                if k == 8:
                    if taxonomy_level == 'coarse':
                        n.append(i)

        for i,l in enumerate(batch_data2_dict['fine_target']):
            k = 0
            for j in range(0, 29):
                if l[j] > 0.6:
                    l[j] = 1
                else:
                    l[j] = 0
                    k+=1
                if k==29:
                    if taxonomy_level == 'fine':
                        n.append(i)

        batch_data2_dict['fine_target'] = np.delete(batch_data2_dict['fine_target'], n, axis=0)
        batch_data2_dict['coarse_target'] = np.delete(batch_data2_dict['coarse_target'], n, axis=0)
        batch_data2_dict['audio_name'] = np.delete(batch_data2_dict['audio_name'], n, axis=0)
        batch_data2_dict['feature'] = np.delete(batch_data2_dict['feature'], n, axis=0)
        batch_data2_dict['spacetime'] = np.delete(batch_data2_dict['spacetime'], n, axis=0)
        if  batch_data2_dict['audio_name'].size==0:
            iteration += 1
            continue
        #使用 概率数据请注释下行，使用去零onehot数据不用注释
        batch_data_dict = batch_data2_dict

        # if iteration <8655:
        #      batch_data_dict = batch_data2_dict
        # elif iteration >=8655 and  iteration % 2 == 0:
        #     batch_data_dict = batch_data2_dict





        # Move data to GPU                                       ,'external_target','external_feature'
        for key in batch_data_dict.keys():
            if key in ['feature', 'fine_target', 'coarse_target', 'spacetime']:
                batch_data_dict[key] = move_data_to_gpu(
                    batch_data_dict[key], cuda)
        # Train
        model.train()
        # 使用mix_up  数据增强
        feature1, spacetime1, targets1_a, targets1_b, lam1 = mixup_data(batch_data_dict['feature'], batch_data_dict['spacetime'],batch_data_dict['fine_target'], alpha=1.0, use_cuda=True)
        feature2, spacetime2, targets2_a, targets2_b, lam2 = mixup_data(batch_data_dict['feature'], batch_data_dict['spacetime'], batch_data_dict['coarse_target'], alpha=1.0,use_cuda=True)
        batch_output1 = model.forward1(feature1, spacetime1)
        batch_output2 = model.forward2(feature2, spacetime2)
        lam1=int(lam1)
        lam2 = int(lam2)
        loss1=(lam1 * binary_cross_entropy(batch_output1, targets1_a)+(1-lam1) *binary_cross_entropy(batch_output1, targets1_b))
        loss2 = (lam2 * binary_cross_entropy(batch_output2, targets2_a) + (1 - lam2) * binary_cross_entropy(batch_output2, targets2_b))
        
        #不使用mix_up  数据增强，请使用以下代码
        # batch_target1 = batch_data_dict['fine_target']
        # batch_output1 = model.forward1(batch_data_dict['feature'], batch_data_dict['spacetime'])
        # batch_target2 = batch_data_dict['coarse_target']
        # batch_output2 = model.forward2(batch_data_dict['feature'], batch_data_dict['spacetime'])
        # loss1 = binary_cross_entropy(batch_output1, batch_target1)
        # loss2 = binary_cross_entropy(batch_output2, batch_target2)

        loss=loss1+loss2


        #使用loss监测请使用以下代码否者注释
        if iteration>4320:
            new_loss=loss.item()
            if len(last_loss)<5:
                last_loss.append(new_loss)
            else:
                cha=0
                for i in range(4):
                 cha +=abs(last_loss[i+1]-last_loss[i])
                if new_loss>last_loss[4] and cha>=(new_loss-last_loss[4]) >cha/2:
                    for i in range(4):
                        last_loss[i]=last_loss[i+1]
                    last_loss[4]=new_loss
                    logging.info(' drop iteration：{}'.format(iteration))
                    iteration += 1
                    continue
                elif new_loss>last_loss[4] and (new_loss-last_loss[4]) >cha/2.75:
                    for i in range(4):
                        last_loss[i]=last_loss[i+1]
                    last_loss[4]=new_loss
                    logging.info(' low weightiteration：{}'.format(iteration))
                    loss=loss/2

                else:
                    for i in range(4):
                        last_loss[i]=last_loss[i+1]
                    last_loss[4]=new_loss


        # # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration%50==0:
            plt_x.append(iteration)
            plt_y.append(loss)

        if iteration%13000 == 0 and iteration !=0:
            plt.figure(1)
            plt.suptitle('test result ', fontsize='18')
            plt.plot(plt_x, plt_y, 'r-', label='loss')
            plt.legend(loc='best')
            plt.savefig(loss_path+'/'+time.strftime('%m%d_%H%M%S',time.localtime(time.time()))+'loss.jpg')
            plt.savefig(loss_path + '/loss.jpg')


        # Stop learning
        if iteration == 13000:
            # logging.info("best_micro_auprc:{:.3f}".format(best_inde['micro_auprc']))
            # logging.info("best_micro_f1:{:.3f}".format(best_inde['micro_f1']))
            # logging.info("best_macro_auprc:{:.3f}".format(best_inde['macro_auprc']))
            # labels = get_labels(taxonomy_level)
            # for k, label in enumerate(labels):
            #     logging.info('    {:<40}{:.3f}'.format(label, best_inde['average_precision'][k]))
            break
        iteration += 1
            

        

def inference_validation(args):
    '''Inference and calculate metrics on validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      taxonomy_level: 'fine' | 'coarse'
      model_type: string, e.g. 'Cnn_9layers_MaxPooling'
      iteration: int
      holdout_fold: '1', which means using validation data
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool
    '''
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
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
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    labels = get_labels(taxonomy_level)
    classes_num = len(labels)

    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    train_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train.h5')
        
    validate_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'validate.h5')
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train.h5')
        
    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type, 
        '{}_iterations.pth'.format(iteration))
    
    submission_path = os.path.join(workspace, 'submissions', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type, 'submission.csv')
    create_folder(os.path.dirname(submission_path))
    
    annotation_path = os.path.join(dataset_dir, 'annotations.csv')
    
    yaml_path = os.path.join(dataset_dir, 'dcase-ust-taxonomy.yaml')
    
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)
        
    # Load scalar
    scalar = load_scalar(scalar_path)

    # Load model
    Model = eval(model_type)
    model = Model(classes_num)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if cuda:
        model.cuda()
        
    # Data generator
    data_generator = DataGenerator(
        train_hdf5_path=train_hdf5_path, 
        validate_hdf5_path=validate_hdf5_path, 
        holdout_fold=holdout_fold, 
        scalar=scalar, 
        batch_size=batch_size)

    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        taxonomy_level=taxonomy_level, 
        cuda=cuda, 
        verbose=True)
    
    # Evaluate on validation data
    evaluator.evaluate(
        data_type='validate', 
        submission_path=submission_path, 
        annotation_path=annotation_path, 
        yaml_path=yaml_path, 
        max_iteration=None)
    
    # Visualize
    if visualize:
        evaluator.visualize(data_type='validate')
    

def inference_evaluation(args):
    '''Inference on evaluation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      taxonomy_level: 'fine' | 'coarse'
      model_type: string, e.g. 'Cnn_9layers_MaxPooling'
      iteration: int
      holdout_fold: 'none', which means using model trained on all development data
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    model_type = args.model_type
    iteration = args.iteration
    holdout_fold = args.holdout_fold
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    labels = get_labels(taxonomy_level)
    classes_num = len(labels)
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    evaluate_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'evaluate.h5')
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train.h5')
        
    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type, 
        'best7.pth')
    
    submission_path = os.path.join(workspace, 'submissions', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type, 'best7_submission.csv')
    create_folder(os.path.dirname(submission_path))
    
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)
        
    # Load scalar
    scalar = load_scalar(scalar_path)

    # Load model
    Model = eval(model_type)
    model = Model(classes_num)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if cuda:
        model.cuda()
        
    # Data generator
    data_generator = TestDataGenerator(
        hdf5_path=evaluate_hdf5_path, 
        scalar=scalar, 
        batch_size=batch_size)
    
    # Forward
    output_dict = forward(
        model=model, 
        generate_func=data_generator.generate(), 
        cuda=cuda, 
        return_target=False)

    # Write submission
    if taxonomy_level == "fine":
        output = output_dict['output1']
    elif taxonomy_level == "coarse":
        output = output_dict['output2']
    write_submission_csv(
    audio_names=output_dict['audio_name'], 
    outputs=output,
    taxonomy_level=taxonomy_level, 
    submission_path=submission_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', 'none'], required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation.add_argument('--workspace', type=str, required=True)
    parser_inference_validation.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)
    parser_inference_validation.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_validation.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_validation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--visualize', action='store_true', default=False, help='Visualize log mel spectrogram of different sound classes.')
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    parser_inference_validation.add_argument('--pre_train', type=bool, default=False)

    parser_inference_evaluation = subparsers.add_parser('inference_evaluation')
    parser_inference_evaluation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_evaluation.add_argument('--workspace', type=str, required=True)
    parser_inference_evaluation.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)
    parser_inference_evaluation.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_evaluation.add_argument('--holdout_fold', type=str, choices=['none'], required=True)
    parser_inference_evaluation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_evaluation.add_argument('--batch_size', type=int, required=True)
    parser_inference_evaluation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_evaluation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    parser_inference_evaluation.add_argument('--pre_train', type=bool, default=False)
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        SEED = 2019
        seed_everything(SEED)
        train(args)

    elif args.mode == 'inference_validation':
        SEED = 2019
        seed_everything(SEED)
        inference_validation(args)

    elif args.mode == 'inference_evaluation':
        SEED = 2019
        seed_everything(SEED)
        inference_evaluation(args)

    else:
        raise Exception('Error argument!')