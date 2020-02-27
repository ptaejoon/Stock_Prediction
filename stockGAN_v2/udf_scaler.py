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

GAN_trainY = pickle.load(open('trainY.sav', 'rb'))
GAN_trainX_STOCK = pickle.load(open('trainX_STOCK.sav', 'rb'))
GAN_trainX_PV = pickle.load(open('trainX_PV.sav', 'rb'))
GAN_testY = pickle.load(open('testY.sav', 'rb'))
GAN_testX_STOCK = pickle.load(open('testX_STOCK.sav', 'rb'))
GAN_testX_PV = pickle.load(open('testX_PV.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))
print(GAN_trainX_STOCK.shape)

GAN_trainX_STOCK= scaler.inverse_transform(GAN_trainX_STOCK.reshape(-1,795))
GAN_trainX_STOCK =GAN_trainX_STOCK.reshape(-1,10,795)
GAN_trainX_STOCK_a =GAN_trainX_STOCK[:,0,:]
GAN_trainX_STOCK_b =GAN_trainX_STOCK[-1,:,:]
GAN_trainX_STOCK = np.concatenate([GAN_trainX_STOCK_a,GAN_trainX_STOCK_b])
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

GAN_trainX_STOCK = np.delete(GAN_trainX_STOCK, (0), axis=0)
GAN_trainSTOCK = GAN_trainX_STOCK.copy()
GAN_trainY = GAN_trainX_STOCK[10:].copy()
GAN_trainX_STOCK = changeLSTMsetX(GAN_trainX_STOCK)
for idx in reversed(range(795)):
    if idx % 5 == 4:
        GAN_trainSTOCK = np.delete(GAN_trainSTOCK, idx, axis=1)
for idx in reversed(range(GAN_trainY.shape[1])):
    if idx % 5 == 4:
        GAN_trainY = np.delete(GAN_trainY, idx, axis=1)
GAN_trainSTOCK = changeLSTMsetX(GAN_trainSTOCK)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

GAN_testX_STOCK = scaler.inverse_transform(GAN_testX_STOCK.reshape(-1, 795))
GAN_testX_STOCK =GAN_testX_STOCK.reshape(-1,10,795)
GAN_testX_STOCK_a =GAN_testX_STOCK[:,0,:]
GAN_testX_STOCK_b =GAN_testX_STOCK[-1,:,:]
GAN_testX_STOCK = np.concatenate([GAN_testX_STOCK_a,GAN_testX_STOCK_b])

for row in range(1, GAN_testX_STOCK.shape[0]):
    for idx in range(4, GAN_testX_STOCK.shape[1], 5):
        if GAN_testX_STOCK[row - 1][idx] == 0:
            GAN_testX_STOCK[row][idx] = 1.3
            continue
        trade_scale = GAN_testX_STOCK[row][idx] / GAN_testX_STOCK[row - 1][idx]
        if trade_scale < 0.5:
            GAN_testX_STOCK[row][idx] = 1.1
        elif trade_scale >= 1:
            GAN_testX_STOCK[row][idx] = 2 * math.log(trade_scale, 2) + 1.3
            if GAN_testX_STOCK[row][idx] > 6.4:
                GAN_testX_STOCK[row][idx] = 6.4
        else:
            GAN_testX_STOCK[row][idx] = 1.3
for row in range(1, GAN_testX_STOCK.shape[0]):
    for idx in range(0, GAN_testX_STOCK.shape[1]):
        if idx % 5 != 4:
            close = idx - idx % 5
            GAN_testX_STOCK[row][idx] = GAN_testX_STOCK[row][idx] / GAN_testX_STOCK[row - 1][close]

GAN_testX_STOCK = np.delete(GAN_testX_STOCK, (0), axis=0)
GAN_testSTOCK = GAN_testX_STOCK.copy()
GAN_testY = GAN_testX_STOCK[10:].copy()
GAN_testX_STOCK = changeLSTMsetX(GAN_testX_STOCK)
for idx in reversed(range(795)):
    if idx % 5 == 4:
        GAN_testSTOCK = np.delete(GAN_testSTOCK, idx, axis=1)
for idx in reversed(range(GAN_testY.shape[1])):
    if idx % 5 == 4:
        GAN_testY = np.delete(GAN_testY, idx, axis=1)
GAN_testSTOCK = changeLSTMsetX(GAN_testSTOCK)

print(GAN_testSTOCK.shape)
print(GAN_testX_STOCK.shape)
print(GAN_testY.shape)
print(GAN_testX_STOCK)
#pickle.dump(GAN_trainSTOCK, open('udf_trainSTOCK.sav', 'wb'))
#pickle.dump(GAN_trainX_STOCK, open('udf_trainX_STOCK.sav', 'wb'))
#pickle.dump(GAN_trainY, open('udf_trainY.sav', 'wb'))
#pickle.dump(GAN_testSTOCK, open('udf_testSTOCK.sav', 'wb'))
#pickle.dump(GAN_testX_STOCK, open('udf_testX_STOCK.sav', 'wb'))
#pickle.dump(GAN_testY, open('udf_testY.sav', 'wb'))

