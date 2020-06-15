
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.fftpack import fft,ifft

class conf:
    sampling_rate = 16000  # 1.6K
    duration = 10  # sec
    hop_length = 320  # to make time steps 128
    fmin = 50
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = 1024  # 40ms
def enframe(signal, nw, inc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length=len(signal) #信号总长度
    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    #zeros+=1e-8
    pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
#    win=np.tile(winfunc(nw),(nf,1))  #window窗函数，这里默认取1
#    return frames*win   #返回帧信号矩阵
    return frames


def devideExcitationAndVocal(y=None):
    l = len(y)
    if l != 512:
        print(l)
    y = y * np.hanning(l)  #############加汉宁窗
    ar = librosa.lpc(y, 12)
    ff = scipy.fft(ar, l) ** (-1)
    est_x = scipy.signal.lfilter(0 - ar[1:], 1, y)
    ERR = scipy.fft(y - est_x)
    G = np.mean((y - est_x) ** 2) ** 0.5;

    # return np.log(np.abs(ff[:l//2])+1e-8),np.log(np.abs(ERR[:l//2])+1e-8)
    return np.abs(ff[:l // 2]) * G, np.abs(ERR[:l // 2])
path = "/home/liuzhuangzhuang/pycharm_P/dataset_root/audio/"
name="00_006215.wav"
import time
import matplotlib.pyplot as plt
import librosa.display

t1 = time.time()
c = 0

y, sr = librosa.load(path+name , sr=16000)
y = scipy.signal.lfilter(([1, -0.97]), 1, y)
ys = enframe(y, 512, 256)
vocal, excitation = np.zeros([ys.shape[0], 256]), np.zeros([ys.shape[0], 256])
for i, y in enumerate(ys):
    try:
        ZY, FT = devideExcitationAndVocal(y)
    except:
        continue
    vocal[i, :] = ZY
    excitation[i, :] = FT
vocal, excitation = vocal.T, excitation.T
# vocal,excitation= vocal.real,excitation.real

# vocal,excitation=np.exp(vocal), np.exp(excitation)
vocal, excitation = np.abs(vocal) ** 2, np.abs(excitation) ** 2
# print(vocal.shape, excitation.shape)
vocal = librosa.feature.melspectrogram(sr=sr, S=vocal, n_mels=128)
excitation = librosa.feature.melspectrogram(sr=sr, S=excitation, n_mels=128)  # to mel scale

vocal = librosa.power_to_db(vocal, ref=np.max)
excitation = librosa.power_to_db(excitation, ref=np.max)



librosa.display.specshow(excitation, x_axis='time', y_axis='mel',
                         sr=conf.sampling_rate, hop_length=conf.hop_length,
                         fmin=conf.fmin, fmax=conf.fmax)
plt.colorbar(format='%+2.0f dB')
plt.title('excitation{}.jpg'.format(name))
plt.savefig('/home/liuzhuangzhuang/pycharm_P/dataset_root/logmel谱/excitation{}.jpg'.format(name))
plt.show()
librosa.display.specshow(vocal, x_axis='time', y_axis='mel',
                         sr=conf.sampling_rate, hop_length=conf.hop_length,
                         fmin=conf.fmin, fmax=conf.fmax)
plt.colorbar(format='%+2.0f dB')
plt.title('vocal{}.jpg'.format(name))
plt.savefig('/home/liuzhuangzhuang/pycharm_P/dataset_root/logmel谱/vocal{}.jpg'.format(name))
plt.show()

print("OK")