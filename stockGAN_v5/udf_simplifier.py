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
GAN_valX_STOCK = pickle.load(open('udf_2020_2_trainX_STOCK.sav','rb'))
GAN_valY = pickle.load(open('udf_2020_2_trainY.sav','rb'))
print(GAN_valX_STOCK.shape)
print(GAN_valY.shape)
#GAN_trainX_STOCK_a =GAN_trainX_STOCK[:,0,:]
#GAN_trainX_STOCK_b =GAN_trainX_STOCK[-1,:,:]
#GAN_trainX_STOCK = np.concatenate([GAN_trainX_STOCK_a,GAN_trainX_STOCK_b])
#close open max min trade
GAN_trainSTOCK= GAN_trainX_STOCK.copy()
for idx in reversed(range(795)):
    if idx % 5 != 0 and idx % 5 != 2:
        GAN_trainSTOCK = np.delete(GAN_trainSTOCK, idx, axis = 2)
for idx in reversed(range(636)):
    if idx % 2 == 1:
        GAN_trainY = np.delete(GAN_trainY, idx, axis=1)
GAN_testSTOCK = GAN_testX_STOCK.copy()
for idx in reversed(range(795)):
    if idx % 5 != 0 and idx % 5 != 2:
        GAN_testSTOCK = np.delete(GAN_testSTOCK, idx, axis = 2)
for idx in reversed(range(636)):
    if idx % 2 == 1:
        GAN_testY = np.delete(GAN_testY, idx, axis=1)
GAN_valSTOCK = GAN_valX_STOCK.copy()
for idx in reversed(range(795)):
    if idx % 5 != 0 and idx % 5 != 2:
        GAN_valSTOCK = np.delete(GAN_valSTOCK, idx, axis = 2)
for idx in reversed(range(636)):
    if idx % 2 == 1:
        GAN_valY = np.delete(GAN_valY, idx, axis=1)

print(GAN_testSTOCK.shape)
print(GAN_testX_STOCK.shape)
print(GAN_testY.shape)
print(GAN_testX_STOCK)
print(GAN_valX_STOCK.shape)
print(GAN_valSTOCK.shape)
print(GAN_valY.shape)

pickle.dump(GAN_trainSTOCK, open('v5_trainSTOCK.sav', 'wb'))
pickle.dump(GAN_trainX_STOCK, open('v5_trainX_STOCK.sav', 'wb'))
pickle.dump(GAN_trainY, open('v5_trainY.sav', 'wb'))
pickle.dump(GAN_testSTOCK, open('v5_testSTOCK.sav', 'wb'))
pickle.dump(GAN_testX_STOCK, open('v5_testX_STOCK.sav', 'wb'))
pickle.dump(GAN_testY, open('v5_testY.sav', 'wb'))
pickle.dump(GAN_valSTOCK, open('v5_valSTOCK.sav', 'wb'))
pickle.dump(GAN_valX_STOCK, open('v5_valX_STOCK.sav', 'wb'))
pickle.dump(GAN_valY, open('v5_valY.sav', 'wb'))
