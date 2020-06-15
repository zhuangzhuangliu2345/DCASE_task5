import librosa
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import librosa.display
path="/home/liuzhuangzhuang/pycharm_P/dataset_root/audio/"
name="00_006215.wav"
pathname=path +name
class conf:
    sampling_rate = 44100   # 1.6K
    duration = 4  # sec
    hop_length = 320  # to make time steps 128
    fmin = 50
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = 1024  # 40ms
def audio_to_melspectrogram(conf, audio):
    print("0")
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=conf.sampling_rate,#采样率
                                                 n_mels=conf.n_mels,#
                                                 hop_length=conf.hop_length,#连续帧之间样本数
                                                 n_fft=conf.n_fft,#FFT窗口的长度
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    print("1")
    return spectrogram
def show_melspectrogram(conf, mel, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mel, x_axis='time', y_axis='mel',
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                             fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('{}.jpg'.format(name))
    plt.savefig('/home/liuzhuangzhuang/pycharm_P/dataset_root/logmel谱/冰淇凌车{}.jpg'.format(name))

    plt.show()
y,sr=librosa.load(pathname, sr=None)
mel=audio_to_melspectrogram(conf, y)
show_melspectrogram(conf, mel, title='Log-frequency power spectrogram')
