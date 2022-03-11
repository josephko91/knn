import numpy as np
import pandas as pd


def data_processing():
    data = pd.read_csv('heart_disease.csv', low_memory=False, sep=',', na_values='?').values

    N = data.shape[0]

    np.random.shuffle(data)
    # prepare data

    ntr = int(np.round(N * 0.8))
    nval = int(np.round(N * 0.15))
    ntest = N - ntr - nval

    # spliting training, validation, and test
    x_train = np.append([np.ones(ntr)], data[:ntr].T[:-1], axis=0).T
    y_train = data[:ntr].T[-1].T
    x_val = np.append([np.ones(nval)], data[ntr:ntr + nval].T[:-1], axis=0).T
    y_val = data[ntr:ntr + nval].T[-1].T
    x_test = np.append([np.ones(ntest)], data[-ntest:].T[:-1], axis=0).T
    y_test = data[-ntest:].T[-1].T
    
    # make labels List[int]
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    return x_train, y_train, x_val, y_val, x_test, y_test
