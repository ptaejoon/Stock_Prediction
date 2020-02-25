import numpy as np
import pymysql
import gensim
import datetime
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# from khaiii import KhaiiiApi
# from tokenizer import tokenize
"""
def adv_loss(y):


def gen_loss(y_true,y_pred):
    return adv_loss(y_pred)+p_loss(y_true,y_pred)+dlp_loss(y_true,y_pred)

def dis_loss(y_true,y_pred):
    return adv_loss
"""

def numpy_one_day_data(self, newsDB, stockDB, days, pv):
    end = days
    start = days + datetime.timedelta(days=-1)
    if days.year % 2 == 0:
        news_sql = "select news from even_article where writetime >= %s and writetime < %s"
    else:
        news_sql = "select news from odd_article where writetime >= %s and writetime < %s"
    # news_sql = "select news from article where writetime >= %s and writetime < %s"
    stock_sql = """select CLOSE_PRICE, OPEN_PRICE, MAX_PRICE, MIN_PRICE, TRADE_AMOUNT
        from stock where TRADE_TIME >= %s and TRADE_TIME < %s and
        corp_name in (
        select distinct corp_name from corp_stock 
        where TRADE_TIME < '2010-01-05 00:00:00')
        order by TRADE_TIME, CORP_NAME """
    #newsCur = newsDB.cursor()
    stockCur = stockDB.cursor()
    stockCur.execute(stock_sql, (start, end))
    stockData = stockCur.fetchall()
    count = 1
    returnVec = np.zeros(self.pv_size)
    while len(stockData) < 1:
        # newsCur.execute(news_sql, (start, end))
        # article_same_day = newsCur.fetchall()
        # tempVec = np.zeros(self.pv_size)
        # for article in article_same_day:
        #     article_token = tokenize(article[0])
        #     vec = pv.infer_vector(article_token)
        #     # vec = pv.infer_vector(article)
        #     tempVec = tempVec + vec
        # if len(article_same_day) is not 0:
        #     tempVec = tempVec / len(article_same_day)
        # returnVec = returnVec + tempVec
        # count += 1
        # # print(str(end)+' does not have stock data')
        end = end + datetime.timedelta(days=1)
        start = start + datetime.timedelta(days=1)
        stockCur.execute(stock_sql, (start, end))
        stockData = stockCur.fetchall()

    #newsCur.execute(news_sql, (start, end))
    #article_same_day = newsCur.fetchall()
    #tempVec = np.zeros(self.pv_size)
    # for article in article_same_day:
    #     vec = pv.infer_vector(article)
    #     tempVec = tempVec + vec
    # if len(article_same_day) is not 0:
    #     tempVec = tempVec / len(article_same_day)
    #returnVec = tempVec + returnVec
    #returnVec = returnVec / count
    stockData = np.array(stockData)
    stockData = stockData.flatten(order='C')
    # returnVec = np.concatenate((stockData, returnVec),axis=None)
    print(str(end) + ' produced input')
    end = end + datetime.timedelta(days=1)
    return stockData, end

def changeLSTMsetX(self, Xdata):
    X = []
    for i in range(len(Xdata)):
        x = Xdata[i:(i + 10)]
        if (i + self.gen_timestep) < len(Xdata):
            X.append(x)
        else:
            break
    return np.array(X)

def build_input(self):
    article = {"host": '127.0.0.1', "port": 3306,
                    "user": 'sinunu', "password": '1q2w3e4r', "db": 'mydb', 'charset': 'utf8'}
    index = {"host": '127.0.0.1', "port": 3306,
                  "user": 'sinunu', "password": '1q2w3e4r', "db": 'mydb', 'charset': 'utf8'}

    articleDBconnect = pymysql.connect(
        host=article["host"],
        port=article['port'],
        user=article['user'],
        password=article['password'],
        db=article['db'],
        charset=article['charset'],
        # cursorclass=pymysql.cursors.DictCursor
    )
    stockDBconnect = pymysql.connect(
        host=index["host"],
        port=index['port'],
        user=index['user'],
        password=index['password'],
        db=index['db'],
        charset=index['charset'],
        # cursorclass=pymysql.cursors.DictCursor
    )
    pv_model = gensim.models.Doc2Vec.load('vocab_20.model')
    # days = datetime.datetime(2011, 1, 2, 17, 0, 0)
    days = datetime.datetime(2010, 1, 2, 15, 30, 0)
    np_pv = np.empty(792)
    np_stock = np.empty(795)
    trainY = np.empty(795)
    trainSTOCK = np.empty(795 * 9)
    testY = np.empty(795)
    testSTOCK = np.empty(795 * 9)
    scaler = MinMaxScaler()
    while days.year < 2019:
        # for i in range(50):
        stock, days = self.numpy_one_day_data(articleDBconnect, stockDBconnect, days, pv_model)
        #np_pv = np.vstack([np_pv, oneday])
        np_stock = np.vstack([np_stock, stock])
        # trainX = np.vstack([trainX, oneday])
        if len(stock) is not 0:
            trainY = np.vstack([trainY, stock])
    np_stock = np.delete(np_stock, (0), axis=0)
    #np_pv = np.delete(np_pv, (0), axis=0)
    np_stock = scaler.fit_transform(np_stock)
    #trainX = np.concatenate((np_stock, np_pv), axis=1)
    trainY = np.delete(trainY, (0), axis=0)
    for i in range(len(np_stock) - 10):
        timestep_days = np.array(trainY[i: i + 9], copy=True)
        trainSTOCK = np.vstack([trainSTOCK, timestep_days.flatten()])
    trainSTOCK = np.delete(trainSTOCK, (0), axis=0)
    # print(trainY[0])
    trainY = trainY[10:]
    np_stock = changeLSTMsetX(np_stock)
    trainY = scaler.transform(trainY)
    trainX_STOCK = np_stock
    # TEST SET BUILDING
    print("------------ START BUILDING TEST SET ---------")
    np_pv = np.empty(792)
    np_stock = np.empty(795)
    while days.year < 2020:
        # for iter in range(15):
        stock, days = self.numpy_one_day_data(articleDBconnect, stockDBconnect, days, pv_model)
        # testX = np.vstack([testX, oneday])
        # np_pv = np.vstack([np_pv, oneday])
        np_stock = np.vstack([np_stock, stock])
        if len(stock) is not 0:
            testY = np.vstack([testY, stock])
    np_stock = np.delete(np_stock, (0), axis=0)
    #np_pv = np.delete(np_pv, (0), axis=0)
    np_stock = scaler.transform(np_stock)
    #testX = np.concatenate((np_stock, np_pv), axis=1)
    testY = np.delete(testY, (0), axis=0)
    for i in range(len(np_stock) - 10 ):
        timestep_days = np.array(testY[i:(i + 9)], copy=True)
        testSTOCK = np.vstack([testSTOCK, timestep_days.flatten()])
    testSTOCK = np.delete(testSTOCK, (0), axis=0)
    testY = testY[10:]
    testY = scaler.transform(testY)
    np_stock = changeLSTMsetX(np_stock)
    testX_STOCK = np_stock
    print("Data Setting DONE")

    print("Data Saving")
    pickle.dump(trainY, open('trainY.sav', 'wb'))
    pickle.dump(trainX_STOCK, open('trainX_STOCK.sav', 'wb'))
    pickle.dump(testY, open('testY.sav', 'wb'))
    pickle.dump(testX_STOCK, open('testX_STOCK.sav', 'wb'))
    pickle.dump(self.scaler, open('scaler.sav', 'wb'))
    print("Data Saving Done")