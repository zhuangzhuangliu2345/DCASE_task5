import numpy as np
import h5py
import csv
import time
import logging
import os
import glob
import matplotlib.pyplot as plt
import logging

from utilities import scale
import config


class Base(object):
    def __init__(self):
        pass

    def load_hdf5(self, hdf5_path):
        '''Load hdf5 file. 
        
        Returns:
          {'audio_name': (audios_num,), 
           'feature': (audios_num, frames_num, mel_bins), 
           (if exist) 'fine_target': (audios_num, classes_num), 
           (if exist) 'coarse_target': (audios_num, classes_num)}
        '''
        data_dict = {}
        
        with h5py.File(hdf5_path, 'r') as hf:
            if 'audio_name' in hf.keys():
                data_dict['audio_name'] = np.array(
                    [audio_name.decode() for audio_name in hf['audio_name'][:]])
            if 'feature' in hf.keys():
                data_dict['feature'] = hf['feature'][:].astype(np.float32)
            if 'spacetime' in hf.keys():
                data_dict['spacetime'] = hf['spacetime'][:].astype(np.float32)
            if 'fine_target' in hf.keys():
                data_dict['fine_target'] = \
                    hf['fine_target'][:].astype(np.float32)

            if 'coarse_target' in hf.keys():
                data_dict['coarse_target'] = \
                    hf['coarse_target'][:].astype(np.float32)   
            if 'label_targets'  in hf.keys():
                data_dict['label_targets'] = \
                    hf['label_targets'][:].astype(np.float32)

        return data_dict

    def transform(self, x):
        return scale(x, self.scalar['mean'], self.scalar['std'])



    def transform2(self,x):

        return scale(x, self.scalar2['mean'],self.scalar2['std'])


class DataGenerator(Base):
    
    def __init__(self, train_hdf5_path, validate_hdf5_path, holdout_fold, 
        scalar, batch_size, seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          train_hdf5_path: string
          validate_hdf5_path: string
          holdout_fold: '1' | 'None', where '1' indicates using validation and 
              'None' indicates using full data for training
          scalar: object, containing mean and std value
          batch_size: int
          seed: int, random seed
        '''
        self.scalar = scalar
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)


        # Load training data
        load_time = time.time()
        # external_hdf5_path=os.path.join(os.path.dirname(train_hdf5_path),'pre19_train.h5')
        # # external_hdf5_path="/home/liuzhuangzhuang/pycharm_P/task5-多任务0.0/work_space/features/logmel_64frames_64melbins/pre_train.h5"
        # with h5py.File('/home/liuzhuangzhuang/pycharm_P/task5-多任务0.0/work_space/scalars/logmel_64frames_64melbins/pre19_train.h5','r') as hf:
        #     mean = hf['mean'][:]
        #     std = hf['std'][:]
        #
        # self.scalar2 = {'mean': mean, 'std': std}
        # self.external_data_dict = self.load_hdf5(external_hdf5_path)
        # self.external_data_indexes = np.arange(
        #     len(self.external_data_dict['audio_name']))
        # logging.info('External audio num: {}'.format(
        #     len( self.external_data_indexes)))
        # self.random_state.shuffle(self.external_data_indexes)
        # self.pointer2 = 0
        self.train_data_dict = self.load_hdf5(os.path.join(os.path.dirname(train_hdf5_path), 'train.h5'))
        self.validate_data_dict = self.load_hdf5(validate_hdf5_path)

        self.train_space_data_dict = self.load_hdf5(os.path.join(os.path.dirname(train_hdf5_path), 'trainspacetime.h5'))
        self.validate_space_data_dict = self.load_hdf5(os.path.join(os.path.dirname(validate_hdf5_path), 'validatespacetime.h5'))
        # Combine train and validate data for training
        if holdout_fold == 'none':
            self.train_data_dict, self.validate_data_dict, self.train_space_data_dict,self.validate_space_data_dict= \
                self.combine_train_validate_data(
                    self.train_data_dict, self.validate_data_dict,self.train_space_data_dict,self.validate_space_data_dict)
        
        self.train_audio_indexes = np.arange(
            len(self.train_data_dict['audio_name']))
            
        self.validate_audio_indexes = np.arange(
            len(self.validate_data_dict['audio_name']))

      
        logging.info('Load data time: {:.3f} s'.format(
            time.time() - load_time))
        logging.info('Training audio num: {}'.format(
            len(self.train_audio_indexes)))            
        logging.info('Validation audio num: {}'.format(
            len(self.validate_audio_indexes)))


        self.random_state.shuffle(self.train_audio_indexes)

        self.pointer = 0

        
    def combine_train_validate_data(self, train_data_dict, validate_data_dict,train_space_data_dict,validate_space_data_dict):
        '''Combining train and validate data to full train data. 
        '''
        new_train_data_dict = {}
        new_validate_data_dict = {}
        new_train_space_data_dict={}
        new_validate_space_data_dict={}
        for key in train_data_dict.keys():
            new_train_data_dict[key] = np.concatenate(
                (train_data_dict[key], validate_data_dict[key]), axis=0)
            new_validate_data_dict[key] = np.array([])
        for key in train_space_data_dict.keys():
            new_train_space_data_dict[key] = np.concatenate(
                (train_space_data_dict[key], validate_space_data_dict[key]), axis=0)
            new_validate_space_data_dict[key] = np.array([])

        return new_train_data_dict, new_validate_data_dict,new_train_space_data_dict,new_validate_space_data_dict
        
    def generate_train(self):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: 
            {'audio_name': (batch_size,), 
             'feature': (batch_size, frames_num, mel_bins), 
             (if exist) 'fine_target': (batch_size, classes_num), 
             (if exist) 'coarse_target': (batch_size, classes_num)}
        '''
        while True:
            # Reset pointer
            if self.pointer >= len(self.train_audio_indexes):
                self.pointer = 0
                self.random_state.shuffle(self.train_audio_indexes)

            # Get batch audio_indexes
            batch_audio_indexes = self.train_audio_indexes[
                self.pointer: self.pointer + self.batch_size]

            self.pointer += self.batch_size

            #
            # if self.pointer2 >= len(self.external_data_indexes):
            #     self.pointer2 = 0
            #     self.random_state.shuffle(self.external_data_indexes)
            #
            # batch_external_indexes = self.external_data_indexes[
            #                          self.pointer2: self.pointer2 + self.batch_size]
            #
            # self.pointer2 += self.batch_size
            # batch_external_feature = self.external_data_dict['feature'][batch_external_indexes]
            # batch_external_feature = self.transform2(batch_external_feature)
            # batch_data_dict['external_feature'] = batch_external_feature
            # batch_data_dict['external_audio_name'] = \
            #     self.external_data_dict['audio_name'][batch_external_indexes]
            #
            # batch_data_dict['external_target'] = \
            #     self.external_data_dict['label_targets'][batch_external_indexes]
            batch_data_dict = {}
            # logging.info(self.external_data_dict.keys())
            batch_data_dict['audio_name'] = \
                self.train_data_dict['audio_name'][batch_audio_indexes]

            
            batch_feature = self.train_data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature



            batch_data_dict['fine_target'] = \
                self.train_data_dict['fine_target'][batch_audio_indexes]                
                
            batch_data_dict['coarse_target'] = \
                self.train_data_dict['coarse_target'][batch_audio_indexes]


            batch_data_dict['spacetime'] = \
                self.train_space_data_dict['spacetime'][batch_audio_indexes]

            
            yield batch_data_dict
            
    def generate_validate(self, data_type, max_iteration=None):
        '''Generate mini-batch data for validation. 
        
        Args:
          data_type: 'train' | 'validate'
          max_iteration: None | int, use maximum iteration of partial data for
              fast evaluation
        
        Returns:
          batch_data_dict: 
            {'audio_name': (batch_size,), 
             'feature': (batch_size, frames_num, mel_bins), 
             (if exist) 'fine_target': (batch_size, classes_num), 
             (if exist) 'coarse_target': (batch_size, classes_num)}
        '''
        batch_size = self.batch_size
        
        if data_type == 'train':
            data_dict = self.train_data_dict
            space_data_dict = self.train_space_data_dict
            audio_indexes = np.array(self.train_audio_indexes)
        elif data_type == 'validate':
            data_dict = self.validate_data_dict
            space_data_dict = self.validate_space_data_dict
            audio_indexes = np.array(self.validate_audio_indexes)
        else:
            raise Exception('Incorrect argument!')
            
        iteration = 0
        pointer = 0
        
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= len(audio_indexes):
                break

            # Get batch audio_indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]                
            pointer += batch_size
            iteration += 1

            batch_data_dict = {}

            batch_data_dict['audio_name'] = \
                data_dict['audio_name'][batch_audio_indexes]
            
            batch_feature = data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature
            
            batch_data_dict['fine_target'] = \
                data_dict['fine_target'][batch_audio_indexes]                
                
            batch_data_dict['coarse_target'] = \
                data_dict['coarse_target'][batch_audio_indexes]
            batch_data_dict['spacetime'] = \
                space_data_dict['spacetime'][batch_audio_indexes]

            yield batch_data_dict


class TestDataGenerator(Base):
    def __init__(self, hdf5_path, scalar, batch_size, seed=1234):
        self.scalar = scalar
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.evalspacetime=self.load_hdf5(os.path.join(os.path.dirname(hdf5_path), 'evalspacetime.h5'))
        # Load training data
        self.data_dict = self.load_hdf5(hdf5_path)

        self.audio_indexes = np.arange(
            len(self.data_dict['audio_name']))
        
        logging.info('Audio num to be inferenced: {}'.format(
            len(self.audio_indexes)))
            
        self.pointer = 0

    def generate(self, max_iteration=None):

        batch_size = self.batch_size

        data_dict = self.data_dict
        audio_indexes = np.array(self.audio_indexes)

        iteration = 0
        pointer = 0
        
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= len(audio_indexes):
                break

            # Get batch audio_indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]                
            pointer += batch_size
            iteration += 1

            batch_data_dict = {}

            batch_data_dict['audio_name'] = \
                data_dict['audio_name'][batch_audio_indexes]
            
            batch_feature = data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature
            batch_data_dict['spacetime'] = \
                self.evalspacetime['spacetime'][batch_audio_indexes]
            yield batch_data_dict