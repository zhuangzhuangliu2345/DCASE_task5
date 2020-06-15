import os
import sys
import random
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
import config
import matplotlib.pyplot as plt
from sklearn import metrics
matplotlib.use('Agg')
from models import Duo_Cnn_9_drop

def create_logging(log_dir, filemode):
    def create_folder(fd):
        if not os.path.exists(fd):
            os.makedirs(fd)
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging
def inference_exteral():
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

    workspace = '/home/liuzhuangzhuang/pycharm_P/task5-多任务0.0/work_space/'
    taxonomy_level = "fine"
    model_type = 'Duo_Cnn_9_drop'
    iteration = 10000
    holdout_fold =1
    batch_size = 32
    cuda = torch.cuda.is_available()

    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second

    labels = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum',
              'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation', 'Bus', 'Buzz',
              'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 'Child_speech_and_kid_speaking',
              'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell', 'Clapping', 'Computer_keyboard',
              'Crackle', 'Cricket', 'Crowd',
              'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans', 'Drawer_open_or_close',
              'Drip', 'Electric_guitar', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking',
              'Fill_(with_liquid)',
              'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong', 'Gurgling', 'Harmonica', 'Hi-hat',
              'Hiss', 'Keys_jangling', 'Knock', 'Male_singing', 'Male_speech_and_man_speaking',
              'Marimba_and_xylophone',
              'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Printer', 'Purr', 'Race_car_and_auto_racing',
              'Raindrop', 'Run', 'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)',
              'Skateboard', 'Slam',
              'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush',
              'Traffic_noise_and_roadway_noise', 'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet',
              'Waves_and_surf', 'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)']
    checkpoint_path = os.path.join(workspace, 'checkpoints','main',
                                   'logmel_{}frames_{}melbins'.format( frames_per_second, mel_bins),
                                   'taxonomy_level={}'.format(taxonomy_level),
                                   'holdout_fold={}'.format(holdout_fold), model_type,
                                   '{}_iterations.pth'.format(iteration))

    logs_dir = os.path.join(workspace, 'logs',  'exteral_train',
                            'logmel_{}frames_{}melbins'.format( frames_per_second, mel_bins),
                            'taxonomy_level={}'.format(taxonomy_level),
                            'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')

    external_hdf5_path='/home/liuzhuangzhuang/pycharm_P/task5-多任务0.0/work_space/features/logmel_64frames_64melbins/pre19_train.h5'
    logging.info('external_hdf5_path={}'.format(external_hdf5_path))
    logging.info('taxonomy_level={}'.format(taxonomy_level))
    logging.info('model_type={}'.format(model_type))
    logging.info('iteration={}'.format(iteration))




    # Load model
    Model = eval(model_type)
    model = Model()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    if cuda:
        model.cuda()



    def generate_exteral(external_hdf5_path):
        external_data_dict = {}

        with h5py.File(external_hdf5_path, 'r') as hf:
            external_data_dict['audio_name'] = np.array(
                [audio_name.decode() for audio_name in hf['audio_name'][:]])

            external_data_dict['feature'] = hf['feature'][:].astype(np.float32)

            if 'label_targets' in hf.keys():
                external_data_dict['label_targets'] = \
                    hf['label_targets'][:].astype(np.float32)


        external_data_indexes = np.arange(
            len(external_data_dict['audio_name']))
        with h5py.File(
                '/home/liuzhuangzhuang/pycharm_P/task5-多任务0.0/work_space/scalars/logmel_64frames_64melbins/pre19_train.h5',
                'r') as hf:
            mean = hf['mean'][:]
            std = hf['std'][:]

        scalar2 = {'mean': mean, 'std': std}

        def scale(x, mean, std):
            return (x - mean) / std
        def transform2(x):

            return scale(x, scalar2['mean'],scalar2['std'])
        random_state = np.random.RandomState(2020)
        random_state.shuffle(external_data_indexes)
        pointer2 = 0
        while True:
            # Reset pointer
            if pointer2 >= len(external_data_indexes):
                break

            batch_external_indexes = external_data_indexes[
                                     pointer2: pointer2 + batch_size]

            pointer2 += batch_size
            batch_data_dict = {}
            # logging.info(self.external_data_dict.keys())
            batch_external_feature = external_data_dict['feature'][batch_external_indexes]
            batch_external_feature = transform2(batch_external_feature)
            batch_data_dict['external_feature'] = batch_external_feature

            batch_data_dict['external_audio_name'] = \
                external_data_dict['audio_name'][batch_external_indexes]

            batch_data_dict['external_target'] = \
                external_data_dict['label_targets'][batch_external_indexes]

            yield batch_data_dict
    # Forward
    generate_exteral=generate_exteral(external_hdf5_path)

    def move_data_to_gpu(x, cuda):
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)
        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)
        else:
            raise Exception("Error!")

        if cuda:
            x = x.cuda()

        return x

    def append_to_dict(dict, key, value):
        if key in dict.keys():
            dict[key].append(value)
        else:
            dict[key] = [value]

    output_dict = {}

    # Evaluate on mini-batch
    for batch_data_dict in generate_exteral:

        # Predict
        batch_feature = move_data_to_gpu(batch_data_dict['external_feature'], cuda)

        with torch.no_grad():
            model.eval()
            batch_output3 = model.forward3(batch_feature)


        append_to_dict(dict=output_dict, key='audio_name',
                       value=batch_data_dict['external_audio_name'])


        append_to_dict(dict=output_dict, key='output3',
                       value=batch_output3.data.cpu().numpy())
        append_to_dict(dict=output_dict, key='external_target',
                       value=batch_data_dict['external_target'])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)


    output=output_dict['output3']
    target = output_dict['external_target']

    def get_binary_target(target):
        '''Get binarized target. The original target is between 0 and 1
        representing the average annotations of labelers. Set a threshold to
        binarize the target to either 0 or 1. We set a small threshold
        simulates XOR operation of labels.
        '''

        threshold = 0.001  # XOR of annotations
        return (np.sign(target - threshold) + 1) / 2

    target = get_binary_target(target)


    average_precision = metrics.average_precision_score(target, output, average=None)
    logging.info('{} ,{},external_average precision:'.format(model_type, taxonomy_level))
    gaolabels=[]
    for k, label in enumerate(labels):
        logging.info('    {:<40}{:.3f}'.format(label, average_precision[k]))
        if average_precision[k]>np.mean(average_precision):
            gaolabels.append(label)

    logging.info('    {:<40}{:.3f}'.format('Average', np.mean(average_precision)))
    logging.info('gaolabels:{}'.format(gaolabels))

if __name__ == '__main__':

    inference_exteral()

