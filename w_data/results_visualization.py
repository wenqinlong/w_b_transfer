import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from myutils import data_norm

para = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle']
data_point = ['data' + str(i) for i in range(603)]
freq = np.arange(30, 80.25, 0.25)  # shape: (201,)


def fw_visual(real, pred, frequency, path):
    for i in para+data_point:
        real[i] = data_norm(real[i], i, reverse=True)
        pred[i] = data_norm(pred[i], i, reverse=True)
    for j in range(2):
        real_LL = real.iloc[j, 6:207]
        real_LR = real.iloc[j, 207:408]
        real_RR = real.iloc[j, 408:609]      # Length: 201, dtype: float64
        # print(real_RR)

        pred_LL = pred.iloc[j, 6:207]  # print(pred_LL.shape)   (201,)
        pred_LR = pred.iloc[j, 207:408]
        pred_RR = pred.iloc[j, 408:609]  # Length: 201, dtype: float64
        # print(pred.iloc[j, 0:6]==real.iloc[j, 0:6])     # True, it means the data is correct.

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        pred_ = [pred_LL, pred_LR, pred_RR]
        real_ = [real_LL, real_LR, real_RR]
        ref = ['LL', 'LR', 'RR']

        for k in range(3):
            ax[k].plot(frequency, pred_[k], 'r--', label='Predicted {}'.format(ref[k]))    # don't forget the plt.legend()
            ax[k].plot(frequency, real_[k], 'b--', label='Simulated {}'.format(ref[k]))
            ax[k].legend()
            ax[k].set_xlabel('Frequency (THz)')
            ax[k].set_ylabel('Reflectance')
            ax[k].set_ylim(0,1)

        plt.savefig(path+'./pred_simu_{}'.format(j), dpi=150)
        plt.show()


if __name__ == '__main__':
    cwd = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/w_data/test_results/'
    real_data = pd.read_csv(cwd+'real/real_data.csv', header=None, names=['id']+para+data_point)
    pred_data = pd.read_csv(cwd+'pred/pred_data.csv', header=None, names=['id']+para+data_point)
    fw_visual(real_data, pred_data, freq, cwd)
