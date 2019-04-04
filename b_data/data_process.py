import pandas as pd
import os
from random import sample
import numpy as np

os.makedirs('./data/test_2000', exist_ok=True)
os.makedirs('./data/train_', exist_ok=True)
os.makedirs('./data/test_norm_2000', exist_ok=True)
os.makedirs('./data/train_', exist_ok=True)

cwd = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/b_data/'

para = pd.read_csv(
    cwd+'data/raw_data/4parameter_Period_600nm_300K.txt',
    delim_whitespace=True, header=None,
    names=['p'+ str(i) for i in range(4)])    # [8392 rows x 1 columns]
phase = pd.read_csv(
    cwd+'data/raw_data/4phase_Period_600nm_300K.txt',                       # [1678600 rows x 2 columns]
    delim_whitespace=True, header=None,names=['wavelength', 'phase'])
reflectance = pd.read_csv(                                                  # [1678600 rows x 2 columns]
    cwd+'data/raw_data/4R_Period_600nm_300K.txt',
    delim_whitespace=True, header=None,
    names=['wavelength', 'reflectance'])

phase_ref = pd.concat([phase,reflectance], axis=1)     # [1678600 rows x 4 columns]
phase_ref = phase_ref.loc[:,~phase_ref.columns.duplicated()]    # [1678600 rows x 3 columns]

wavelength = phase.iloc[0:200,0]
data = np.empty((0, 404))
for i in range(8393):
    pa = para.iloc[i, :]
    ph = phase_ref.iloc[i*200:(i+1)*200, 1]
    ref = phase_ref.iloc[i*200:(i+1)*200, 2]
    concat_data = np.concatenate([pa.values.reshape([1,4]), ph.values.reshape([1,200]), ref.values.reshape([1,200])], axis=1)
    data = np.concatenate([data, concat_data], axis=0)
np.savetxt('./data/raw_data/data.csv', data, delimiter=',')


# for i in range(8392):
#     data = pd.concat[]

