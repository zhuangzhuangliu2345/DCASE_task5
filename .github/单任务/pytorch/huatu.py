import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.05, 10, 1000)
y = np.cos(x)

plt.plot(x, y, ls="-", lw=2, label="plot figure")

plt.legend()

plt.show()

batch_data2_dict = batch_data_dict
        n = []
        if taxonomy_level == 'coarse':
            for i, l in enumerate(batch_data2_dict['coarse_target']):
                k = 0
                for j in range(0, 8):
                    if l[j] > 0.6:
                        l[j] = 1
                    else:
                        l[j] = 0
                        k += 1
                    if k == 8:
                        n.append(i)
        if taxonomy_level == 'fine':
            for i, l in enumerate(batch_data2_dict['fine_target']):
                k = 0
                for j in range(0, 29):
                    if l[j] > 0.6:
                        l[j] = 1
                    else:
                        l[j] = 0
                        k += 1
                    if k == 29:
                        n.append(i)

        batch_data2_dict['fine_target'] = np.delete(batch_data2_dict['fine_target'], n, axis=0)
        batch_data2_dict['coarse_target'] = np.delete(batch_data2_dict['coarse_target'], n, axis=0)
        batch_data2_dict['audio_name'] = np.delete(batch_data2_dict['audio_name'], n, axis=0)
        batch_data2_dict['feature'] = np.delete(batch_data2_dict['feature'], n, axis=0)
        if batch_data2_dict['audio_name'].size == 0:
            iteration += 1
            continue
        batch_data_dict = batch_data2_dict
