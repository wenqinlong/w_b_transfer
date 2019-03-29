import pandas as pd
from random import sample
from copy import deepcopy
import os
from myutils import data_norm

os.makedirs('./data/test', exist_ok=True)
os.makedirs('./data/train', exist_ok=True)
os.makedirs('./data/test_norm', exist_ok=True)
os.makedirs('./data/train_norm', exist_ok=True)

cwd = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/w_data/'
para = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle']
data_point = ['data' + str(i) for i in range(201)]

LL = pd.read_csv(cwd + 'data/raw_data/LL_data.txt', delim_whitespace=True, header=None,
                 names=para+data_point)                               # print(data_load_LL)    # shape (30513, 206)
LR = pd.read_csv(cwd + 'data/raw_data/LR_data.txt', delim_whitespace=True, header=None,
                 names=para+data_point)
RR = pd.read_csv(cwd + 'data/raw_data/LR_data.txt', delim_whitespace=True, header=None,
                 names=para+data_point)

# prepare random test data and training data
test_loc = sample(range(30513), 5000)  # print(len(test_loc))    # 5000

test_LL = LL.loc[test_loc, :]
test_LR = LR.loc[test_loc, :]
test_RR = RR.loc[test_loc, :]

train_LL = LL.drop(test_loc, axis=0)
train_LR = LR.drop(test_loc, axis=0)
train_RR = RR.drop(test_loc, axis=0)

test_LL.to_csv(cwd + 'data/test/test_LL.csv')
test_LR.to_csv(cwd + 'data/test/test_LR.csv')
test_RR.to_csv(cwd + 'data/test/test_RR.csv')

train_LL.to_csv(cwd + 'data/train/train_LL.csv')
train_LR.to_csv(cwd + 'data/train/train_LR.csv')
train_RR.to_csv(cwd + 'data/train/train_RR.csv')

# deepcopy data for normalization
test_LL_norm = deepcopy(test_LL)
test_LR_norm = deepcopy(test_LR)
test_RR_norm = deepcopy(test_RR)
train_LL_norm = deepcopy(train_LL)
train_LR_norm = deepcopy(train_LR)
train_RR_norm = deepcopy(train_RR)

for i in para+data_point:
    test_LL_norm[i] = data_norm(test_LL_norm[i], i)
    test_LR_norm[i] = data_norm(test_LR_norm[i], i)
    test_RR_norm[i] = data_norm(test_RR_norm[i], i)

    train_LL_norm[i] = data_norm(train_LL_norm[i], i)
    train_LR_norm[i] = data_norm(train_LR_norm[i], i)
    train_RR_norm[i] = data_norm(train_RR_norm[i], i)

test_LL_norm.to_csv(cwd + 'data/test_norm/test_LL_norm.csv')
test_LR_norm.to_csv(cwd + 'data/test_norm/test_LR_norm.csv')
test_RR_norm.to_csv(cwd + 'data/test_norm/test_RR_norm.csv')

train_LL_norm.to_csv(cwd + 'data/train_norm/train_LL_norm.csv')
train_LR_norm.to_csv(cwd + 'data/train_norm/train_LR_norm.csv')
train_RR_norm.to_csv(cwd + 'data/train_norm/train_RR_norm.csv')
