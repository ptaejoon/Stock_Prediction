#import keras
import pandas as pd
from pandas import DataFrame
import numpy as np
import gensim
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import math


def changeLSTMsetX( Xdata):
    X = []
    for i in range(len(Xdata)):
        x = Xdata[i:(i + 10)]
        if (i + 10) < len(Xdata):
            X.append(x)
        else:
            break
    return np.array(X)

GAN_trainX_STOCK = pickle.load(open('udf_trainX_STOCK.sav', 'rb'))
GAN_testX_STOCK = pickle.load(open('udf_testX_STOCK.sav', 'rb'))
GAN_trainY = pickle.load(open('udf_trainY.sav','rb'))
GAN_testY = pickle.load(open('udf_testY.sav','rb'))
print(GAN_trainX_STOCK.shape)

#GAN_trainX_STOCK_a =GAN_trainX_STOCK[:,0,:]
#GAN_trainX_STOCK_b =GAN_trainX_STOCK[-1,:,:]
#GAN_trainX_STOCK = np.concatenate([GAN_trainX_STOCK_a,GAN_trainX_STOCK_b])
print(GAN_trainX_STOCK.shape)
#(present - 0.7) / (1.3 - 0.7) * 2 - 1
for idx in range(4,GAN_trainX_STOCK.shape[2],5):
    # (present - 1.1) / (6.4 - 1.1) * 2 - 1
    GAN_trainX_STOCK[:,:,idx] = (GAN_trainX_STOCK[:,:,idx] - 1.1) / 5.3 * 2 - 1
for idx in range(0,GAN_trainX_STOCK.shape[2]):
    if idx % 5 != 4:
        GAN_trainX_STOCK[:,:,idx] = (GAN_trainX_STOCK[:,:,idx] - 0.7) / 0.6 * 2 - 1
GAN_trainSTOCK= GAN_trainX_STOCK.copy()
GAN_trainY = (GAN_trainY-0.7)/0.6*2 - 1
for idx in reversed(range(795)):
    if idx % 5 == 4:
        GAN_trainSTOCK = np.delete(GAN_trainSTOCK, idx, axis=2)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
for idx in range(4,GAN_testX_STOCK.shape[2],5):
    # (present - 1.1) / (6.4 - 1.1) * 2 - 1
    GAN_testX_STOCK[:,:,idx] = (GAN_testX_STOCK[:,:,idx] - 1.1) / 5.3 * 2 - 1
for idx in range(0,GAN_testX_STOCK.shape[2]):
    if idx % 5 != 4:
        GAN_testX_STOCK[:,:,idx] = (GAN_testX_STOCK[:,:,idx] - 0.7) / 0.6 * 2 - 1
GAN_testSTOCK= GAN_testX_STOCK.copy()
GAN_testY = (GAN_testY-0.7)/0.6*2 - 1
for idx in reversed(range(795)):
    if idx % 5 == 4:
        GAN_testSTOCK = np.delete(GAN_testSTOCK, idx, axis=2)

print(GAN_testSTOCK.shape)
print(GAN_testX_STOCK.shape)
print(GAN_testY.shape)
print(GAN_testX_STOCK)
pickle.dump(GAN_trainSTOCK, open('v4_trainSTOCK.sav', 'wb'))
pickle.dump(GAN_trainX_STOCK, open('v4_trainX_STOCK.sav', 'wb'))
pickle.dump(GAN_trainY, open('v4_trainY.sav', 'wb'))
pickle.dump(GAN_testSTOCK, open('v4_testSTOCK.sav', 'wb'))
pickle.dump(GAN_testX_STOCK, open('v4_testX_STOCK.sav', 'wb'))
pickle.dump(GAN_testY, open('v4_testY.sav', 'wb'))

