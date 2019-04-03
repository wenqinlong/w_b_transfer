import pandas as pd
import matplotlib.pyplot as plt
import random
from myutils import data_norm

para = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle']
data_point = ['data' + str(i) for i in range(603)]


def fw_visual(real, pred):
    for i in para+data_point:
        real[i] = data_norm(real[i], i, reverse=True)
        pred[i] = data_norm(pred[i], i, reverse=True)
    print(real.head())
    for j in range(2):
        LL = real.iloc[j, 6:207]
        LR = real.iloc[j, 207:408]
        RR = real.iloc[j, 408:609]
        print(LL)



if __name__=='__main__':
    cwd = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/w_data/test_results/'
    real_data = pd.read_csv(cwd+'real/real_data.csv', header=None, names=['id']+para+data_point)
    pred_data = pd.read_csv(cwd+'pred/pred_data.csv', header=None, names=['id']+para+data_point)
    fw_visual(real_data, pred_data)
