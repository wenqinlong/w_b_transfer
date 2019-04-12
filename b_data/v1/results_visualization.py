import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def denorm(x):
    # de-normalization
    for i in range(5):
        x.iloc[:, i] = 0.5 * (x.iloc[:, i] + 1) * (x.iloc[:, i].max() - x.iloc[:, i].min()) + x.iloc[:, i].min()

    x.iloc[:, 5:206] = x.iloc[:, 5:206] * 3.15
    x.iloc[:, 206:407] = 0.5 * x.iloc[:, 206:407] - 1
    # print(x.iloc[:, 5:206])
    # x.to_csv(path + './data/raw_data/data_dnorm.csv')
    return x


def fw_visual(simu, pred, frequency, path, epoch):

    for j in range(20):
        simu_phase = simu.iloc[j, 5:205]
        simu_ref = simu.iloc[j, 206:406]

        pred_phase = pred.iloc[j, 5:205]
        pred_ref = pred.iloc[j, 206:406]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        pred_ = [pred_phase, pred_ref]
        simu_ = [simu_phase, simu_ref]
        rp = ['phase', 'reflectance']

        for k in range(2):
            ax[k].plot(frequency, pred_[k], 'r--', label='Predicted {}'.format(rp[k]))    # don't forget the plt.legend()
            ax[k].plot(frequency, simu_[k], 'b--', label='Simulated {}'.format(rp[k]))
            ax[k].legend()
            ax[k].set_xlabel('Wavelength (nm)')
            ax[k].set_ylabel(rp[k])
            # if k == 0:
            #     ax[k].set_ylim(-3.15, 3.15)
            # else:
            #     ax[k].set_ylim(-1, 0)

        plt.savefig(path+'epoch_{}/pred_simu_{}'.format(epoch, j), dpi=150)
        plt.show()


if __name__ == '__main__':
    cwd = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/b_data/test_results/'
    wavelength = np.linspace(1000, 2000, 200)  # shape: (200,)
    para = ['para' + str(i) for i in range(5)]
    data_point = ['data' + str(i) for i in range(402)]

    EPOCH = 1500

    simu_data = pd.read_csv(cwd+'epoch_{}/simu_data.csv'.format(EPOCH), header=None, names=para+data_point)
    pred_data = pd.read_csv(cwd+'epoch_{}/pred_data.csv'.format(EPOCH), header=None, names=para+data_point)
    denorm_simu_data = denorm(simu_data)
    denorm_pred_data = denorm(pred_data)
    fw_visual(denorm_simu_data, denorm_pred_data, wavelength, cwd, EPOCH)
