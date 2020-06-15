import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)
def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0:: 2].transpose(0, -1) * mixup_lambda[0:: 2] + x[1:: 2].transpose(0, -1) * mixup_lambda[1:: 2]).transpose(0, -1)
    return out
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    
    
class Cnn_5layers_AvgPooling(nn.Module):
    
    def __init__(self, classes_num):
        super(Cnn_5layers_AvgPooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.fc)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 1))
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        
        output = torch.sigmoid(self.fc(x))
        
        return output
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
    
class Cnn_9layers_MaxPooling(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_9layers_MaxPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='max')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        
        output = torch.sigmoid(self.fc(x))
        
        return output
        
        
class Cnn_9layers_AvgPooling(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_9layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        #
        # output = torch.sigmoid(self.fc(x))


        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = torch.sigmoid(self.fc(x))
        return output
        
        
class Cnn_13layers_AvgPooling(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_13layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        
        output = torch.sigmoid(self.fc(x))
        
        return output


class Cnn_13layers_MaxPooling(nn.Module):
    def __init__(self, classes_num):
        super(Cnn_13layers_MaxPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='max')

        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)

        output = torch.sigmoid(self.fc(x))

        return output

class Pre_Cnn14(nn.Module):
    def __init__(self, classes_num):

        super(Pre_Cnn14, self).__init__()

        #
        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None
        #
        # # Spectrogram extractor
        # self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
        #                                          win_length=window_size, window=window, center=center,
        #                                          pad_mode=pad_mode,
        #                                          freeze_parameters=True)
        #
        # # Logmel feature extractor
        # self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
        #                                          n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
        #                                          top_db=top_db,
        #                                          freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)


        self.fc1 = nn.Linear(2048, 2048, bias=True)

        self.fc2 = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        x = input[:, None, :, :]
        # x = x.transpose(1, 3)#1,3轴转换
        # x = self.bn0(x)
        # x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc2(x))

        output_dict = clipwise_output

        return output_dict



class Pre_Cnn10(nn.Module):
    def __init__(self, classes_num):

        super(Pre_Cnn10, self).__init__()
        #
        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None
        #
        # # Spectrogram extractor
        # self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
        #                                          win_length=window_size, window=window, center=center,
        #                                          pad_mode=pad_mode,
        #                                          freeze_parameters=True)
        #
        # # Logmel feature extractor
        # self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
        #                                          n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
        #                                          top_db=top_db,
        #                                          freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc2 = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        x = input[:, None, :, :]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        output_dict = torch.sigmoid(self.fc2(x))



        return output_dict


class Duo_Cnn_9layers_AvgPooling(nn.Module):
    def __init__(self, classes_num):
        super(Duo_Cnn_9layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 29, bias=True)
        self.fc2 = nn.Linear(512, 8, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward1(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)

        # output = torch.sigmoid(self.fc(x))
        #
        # x = torch.mean(x, dim=3)
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2

        output = torch.sigmoid(self.fc1(x))
        return output

    def forward2(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)

        # output = torch.sigmoid(self.fc(x))

        # x = torch.mean(x, dim=3)
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2

        output = torch.sigmoid(self.fc2(x))
        return output

class Duo_Cnn10(nn.Module):
    def __init__(self, classes_num):

        super(Duo_Cnn10, self).__init__()
        #
        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None
        #
        # # Spectrogram extractor
        # self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
        #                                          win_length=window_size, window=window, center=center,
        #                                          pad_mode=pad_mode,
        #                                          freeze_parameters=True)
        #
        # # Logmel feature extractor
        # self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
        #                                          n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
        #                                          top_db=top_db,
        #                                          freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc2 = nn.Linear(512, 29, bias=True)
        self.fc3 = nn.Linear(512, 8, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)

    def forward1(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        x = input[:, None, :, :]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        output_dict = torch.sigmoid(self.fc2(x))



        return output_dict
    def forward2(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        x = input[:, None, :, :]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        output_dict = torch.sigmoid(self.fc3(x))



        return output_dict

class Duo_Cnn_9_drop(nn.Module):
    def __init__(self,classes_num):
        super(Duo_Cnn_9_drop, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 29, bias=True)
        self.fc2 = nn.Linear(512, 8, bias=True)
        self.fc3 = nn.Linear(512, 80, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)

    def forward1(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        #
        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = torch.sigmoid(self.fc1(x))
        return output

    def forward2(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        #
        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = torch.sigmoid(self.fc2(x))
        return output
    def forward3(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        #
        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = torch.sigmoid(self.fc3(x))
        return output

class Duo_Cnn_621_drop(nn.Module):
    def __init__(self, classes_num):
        super(Duo_Cnn_621_drop, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block6 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 29, bias=True)
        self.fc2 = nn.Linear(512, 8, bias=True)
        self.fc3 = nn.Linear(512, 80, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)

    def forward1(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        #
        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = torch.sigmoid(self.fc1(x))
        return output

    def forward2(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        #
        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = torch.sigmoid(self.fc2(x))
        return output
    def forward3(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        #
        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = torch.sigmoid(self.fc3(x))
        return output

class Duo_Cnn_621layers_AvgPooling(nn.Module):
    def __init__(self, classes_num):
        super(Duo_Cnn_621layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 29, bias=True)
        self.fc2 = nn.Linear(512, 8, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward1(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        #
        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = torch.sigmoid(self.fc1(x))
        return output

    def forward2(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        #
        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = torch.sigmoid(self.fc2(x))
        return output

class Space_Duo_Cnn_9_Avg(nn.Module):
    def __init__(self, classes_num):
        super(Space_Duo_Cnn_9_Avg, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(528, 29, bias=True)
        self.fc2 = nn.Linear(528, 8, bias=True)
        self.fc3 = nn.Linear(528, 528, bias=True)
        self.fc4 = nn.Linear(528, 528, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc4)
        init_layer(self.fc3)
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward1(self, input, spacetime):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)

        # output = torch.sigmoid(self.fc(x))
        #
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = torch.cat((x, spacetime), 1)
        x = F.relu_(self.fc3(x))
        output = torch.sigmoid(self.fc1(x))
        return output

    def forward2(self, input, spacetime):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)

        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = torch.cat((x, spacetime), 1)
        x = F.relu_(self.fc4(x))
        output = torch.sigmoid(self.fc2(x))
        return output

class Space_Duo_Cnn_9_2max(nn.Module):
    def __init__(self, classes_num):
        super(Space_Duo_Cnn_9_2max,self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(528, 29, bias=True)
        self.fc2 = nn.Linear(528, 8, bias=True)
        self.fc3 = nn.Linear(528, 528, bias=True)
        self.fc4 = nn.Linear(528, 528, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc4)
        init_layer(self.fc3)
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward1(self, input, spacetime):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)

        # output = torch.sigmoid(self.fc(x))
        #
        # x = torch.mean(x, dim=3)
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        x = torch.mean(x, dim=3)
        y = kmax_pooling(x, 2, 2)
        x1=y[:,:,0]
        x2=y[:,:,1]
        x3 = torch.mean(x, dim=2)
        x = x1 + x2+x3
        x = torch.cat((x, spacetime), 1)
        x = F.relu_(self.fc3(x))

        output = torch.sigmoid(self.fc1(x))
        return output

    def forward2(self, input, spacetime):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)

        # output = torch.sigmoid(self.fc(x))

        # x = torch.mean(x, dim=3)
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        x = torch.mean(x, dim=3)
        y = kmax_pooling(x, 2, 2)
        x1 = y[:, :, 0]
        x2 = y[:, :, 1]
        x3 = torch.mean(x, dim=2)
        x = x1 + x2+x3
        x = torch.cat((x, spacetime), 1)
        x = F.relu_(self.fc4(x))
        output = torch.sigmoid(self.fc2(x))
        return output


class Space_Duo_Cnn_9_2biaoqian(nn.Module):
    def __init__(self, classes_num):
        super(Space_Duo_Cnn_9_2biaoqian, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(528, 29, bias=True)
        self.fc2 = nn.Linear(528, 8, bias=True)
        self.fc3 = nn.Linear(528, 528, bias=True)
        self.fc4 = nn.Linear(528, 528, bias=True)
        # self.fc5 = nn.Linear(37, 29, bias=True)
        self.fc6 = nn.Linear(37, 8, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc4)
        init_layer(self.fc3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        # init_layer(self.fc5)
        init_layer(self.fc6)

    def forward1(self, input, spacetime):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)

        # output = torch.sigmoid(self.fc(x))
        #
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = torch.cat((x, spacetime), 1)
        x = F.relu_(self.fc3(x))
        output = torch.sigmoid(self.fc1(x))
        return output

    def forward2(self, input, spacetime):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        # x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        # (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)

        # output = torch.sigmoid(self.fc(x))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = torch.cat((x, spacetime), 1)
        x = F.relu_(self.fc4(x))
        x = self.fc2(x)
        output2 = self.forward1(input, spacetime)
        x = torch.cat((x, output2), 1)
        x = torch.sigmoid(self.fc6(x))

        return x

