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

GAN_trainY = pickle.load(open('2020_2_trainY.sav', 'rb'))
GAN_trainSTOCK = pickle.load(open('2020_2_trainSTOCK.sav', 'rb'))
GAN_trainX_STOCK = pickle.load(open('2020_2_STOCK', 'rb'))
print(GAN_trainX_STOCK.shape)

for row in range(1,GAN_trainX_STOCK.shape[0]):
    for idx in range(4,GAN_trainX_STOCK.shape[1],5):
        if GAN_trainX_STOCK[row-1][idx] == 0:
            GAN_trainX_STOCK[row][idx] = 1.3
            continue
        trade_scale = GAN_trainX_STOCK[row][idx]/GAN_trainX_STOCK[row-1][idx]
        if trade_scale < 0.5:
            GAN_trainX_STOCK[row][idx] = 1.1
        elif trade_scale >= 1:
            GAN_trainX_STOCK[row][idx] = 2 * math.log(trade_scale,2)+1.3
            if GAN_trainX_STOCK[row][idx] > 6.4:
                GAN_trainX_STOCK[row][idx]=6.4
        else:
            GAN_trainX_STOCK[row][idx] = 1.3
for row in reversed(range(1,GAN_trainX_STOCK.shape[0])):
    for idx in range(0,GAN_trainX_STOCK.shape[1]):
        if idx % 5 != 4:
            close = idx - idx % 5
            GAN_trainX_STOCK[row][idx] = GAN_trainX_STOCK[row][idx] / GAN_trainX_STOCK[row-1][close]
            if GAN_trainX_STOCK[row][idx] < 0.7:
                GAN_trainX_STOCK[row][idx] = 0.7
            if GAN_trainX_STOCK[row][idx] > 1.3:
                GAN_trainX_STOCK[row][idx] = 1.3


GAN_trainX_STOCK = np.delete(GAN_trainX_STOCK, (0), axis=0)
GAN_trainSTOCK = GAN_trainX_STOCK.copy()
GAN_trainY = GAN_trainX_STOCK[10:].copy()
GAN_trainX_STOCK = changeLSTMsetX(GAN_trainX_STOCK)
PV = pickle.load(open('2020_2_PV.sav','rb'))
PV = np.delete(PV,(0),axis=0)
PV = changeLSTMsetX(PV)

for idx in reversed(range(795)):
    if idx % 5 == 4:
        GAN_trainSTOCK = np.delete(GAN_trainSTOCK, idx, axis=1)
for idx in reversed(range(GAN_trainY.shape[1])):
    if idx % 5 == 4:
        GAN_trainY = np.delete(GAN_trainY, idx, axis=1)
GAN_trainSTOCK = changeLSTMsetX(GAN_trainSTOCK)
pickle.dump(PV,open('udf_2020_2_PV.sav','wb'))
pickle.dump(GAN_trainSTOCK, open('udf_2020_2_trainSTOCK.sav', 'wb'))
pickle.dump(GAN_trainX_STOCK, open('udf_2020_2_trainX_STOCK.sav', 'wb'))
pickle.dump(GAN_trainY, open('udf_2020_2_trainY.sav', 'wb'))

