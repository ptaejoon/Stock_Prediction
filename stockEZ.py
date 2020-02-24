import keras
from keras import layers
import pandas as pd
from pandas import DataFrame
import numpy as np
import pymysql
import gensim
import datetime
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
#from khaiii import KhaiiiApi
#from tokenizer import tokenize
"""
def adv_loss(y):
    

def gen_loss(y_true,y_pred):
    return adv_loss(y_pred)+p_loss(y_true,y_pred)+dlp_loss(y_true,y_pred)

def dis_loss(y_true,y_pred):
    return adv_loss
"""
def root_mean_squared_error(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))

class easyModel():
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.pv_size = 792
        self.stock_size = 795
        self.gen_timestep = 10
        self.corp_num = 159
        self.gen_output = self.corp_num*1
        self.gen_feature = self.pv_size+self.stock_size
        self.article = {"host": '127.0.0.1', "port": 3306,
                  "user": 'sinunu', "password": '1q2w3e4r', "db": 'mydb', 'charset': 'utf8'}
        self.index = {"host": '127.0.0.1', "port": 3306,
                  "user": 'sinunu', "password": '1q2w3e4r', "db": 'mydb', 'charset': 'utf8'}
        
        self.generator = self.build_generator(LayerName='LSTM')
        print("Start Building Data")
        if os.path.isfile('trainX.sav') is True:
            print("Data Already Exists. Start Loading")
            self.trainX = pickle.load(open('trainX.sav','rb'))
            self.trainY = pickle.load(open('trainY.sav','rb'))
            self.trainSTOCK = pickle.load(open('trainSTOCK.sav','rb'))
            self.testX = pickle.load(open('testX.sav','rb'))
            self.testY = pickle.load(open('testY.sav','rb'))
            self.testSTOCK = pickle.load(open('testSTOCK.sav','rb'))
            self.scaler = pickle.load(open('scaler.sav','rb'))
            trainX_transformed = self.scaler.inverse_transform(self.trainX[:,:,:self.stock_size].reshape(-1,self.stock_size))
            testX_transformed = self.scaler.inverse_transform(self.testX[:,:,:self.stock_size].reshape(-1,self.stock_size))
            self.trainY = self.scaler.inverse_transform(self.trainY)
            self.testY = self.scaler.inverse_transform(self.testY)
            #print(trainX_transformed.shape)
            scaling_data = np.vstack([trainX_transformed,testX_transformed,self.testY,self.trainY])
            print(scaling_data.shape)
            self.scaler = MinMaxScaler()
            self.scaler.fit(scaling_data)
            self.scaler.data_min_ = [0] * self.stock_size
            self.scaler.data_max_ = self.scaler.data_max_*1.3
            #print(self.scaler.data_min_)
            #print(self.scaler.data_max_)
            print(self.scaler.data_min_)

            self.trainX[:,:,:self.stock_size] = self.scaler.transform(trainX_transformed).reshape(-1,self.gen_timestep,self.stock_size)
            self.testX[:,:,:self.stock_size] = self.scaler.transform(testX_transformed).reshape(-1,self.gen_timestep,self.stock_size)
            self.trainY = self.scaler.transform(self.trainY)
            self.testY = self.scaler.transform(self.testY)
            #print(self.scaler.data_min_)
            for idx in reversed(range(self.stock_size)):
                if idx % 5 is not 0:
                    self.trainY = np.delete(self.trainY, idx, axis=1)
            print(self.trainY.shape)
            for idx in reversed(range(self.stock_size)):
                if idx % 5 is not 0 :
                    self.testY = np.delete(self.testY, idx, axis=1)
            print(self.testY.shape)

        else:
            print("Data Doesn't Exists. Start Building")
            self.trainX,self.trainY,self.trainSTOCK,self.testX,self.testY,self.testSTOCK = self.build_input()
        print("Training Data Processing Finished")
        self.generator.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.00001, 0.5),
                               metrics=['accuracy'])
        #self.generator.compile(loss = root_mean_squared_error,optimizer=keras.optimizers.Adam(0.001,0.5),metrics=['accuracy'])
        self.generator.summary()

    def build_generator(self, LayerName):
        model = keras.Sequential()
        if LayerName == 'LSTM':
            model.add(keras.layers.LSTM(self.gen_output,batch_input_shape=(self.batch_size,self.gen_timestep,self.gen_feature)))# input_shape=(self.gen_timestep, self.gen_feature)))#return_sequences=True))
        elif LayerName == 'GRU':
            model.add(keras.layers.GRU(self.gen_output, input_shape=(self.gen_timestep, self.gen_feature)))#return_sequences=True))
        #model.add(keras.layers.ReLU())
        #model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.gen_output))
        model.add(keras.layers.ReLU())
        model.summary()
        merged_input = keras.Input(shape=(self.gen_timestep,self.gen_feature))
        return keras.Model(merged_input, model(merged_input),name='generator')

    def numpy_one_day_data(self,newsDB,stockDB,days,pv):
        end = days
        start = days + datetime.timedelta(days=-1)
        if days.year % 2 == 0:
            news_sql = "select news from even_article where writetime >= %s and writetime < %s"
        else :
            news_sql = "select news from odd_article where writetime >= %s and writetime < %s"
        #news_sql = "select news from article where writetime >= %s and writetime < %s"
        stock_sql = """select CLOSE_PRICE, OPEN_PRICE, MAX_PRICE, MIN_PRICE, TRADE_AMOUNT
            from corp_stock where TRADE_TIME >= %s and TRADE_TIME < %s and
            corp_name in (
            select distinct corp_name from corp_stock 
            where TRADE_TIME < '2010-01-05 00:00:00')
            order by TRADE_TIME, CORP_NAME """
        newsCur = newsDB.cursor()
        stockCur = stockDB.cursor()
        stockCur.execute(stock_sql,(start,end))
        stockData = stockCur.fetchall()
        count = 1
        returnVec = np.zeros(self.pv_size)
        while len(stockData) < 1:
            newsCur.execute(news_sql, (start, end))
            article_same_day = newsCur.fetchall()
            tempVec = np.zeros(self.pv_size)
            for article in article_same_day:
                article_token = tokenize(article[0])
                vec = pv.infer_vector(article_token)
                #vec = pv.infer_vector(article)
                tempVec = tempVec + vec
            if len(article_same_day) is not 0:
                tempVec = tempVec / len(article_same_day)
            returnVec = returnVec + tempVec
            count += 1
            #print(str(end)+' does not have stock data')
            end = end + datetime.timedelta(days=1)
            start = start + datetime.timedelta(days=1)
            stockCur.execute(stock_sql,(start,end))
            stockData = stockCur.fetchall()

        newsCur.execute(news_sql,(start,end))
        article_same_day = newsCur.fetchall()
        tempVec = np.zeros(self.pv_size)
        for article in article_same_day:
            vec = pv.infer_vector(article)
            tempVec = tempVec+vec
        if len(article_same_day) is not 0:
            tempVec = tempVec / len(article_same_day)
        returnVec = tempVec + returnVec
        returnVec = returnVec / count
        stockData = np.array(stockData)
        stockData = stockData.flatten(order='C')
        #returnVec = np.concatenate((stockData, returnVec),axis=None)
        print(str(end)+' produced input')
        end = end + datetime.timedelta(days=1)
        return returnVec,stockData,end
    def changeLSTMsetX(self,Xdata):
        X = []
        for i in range(len(Xdata)):
            x = Xdata[i:(i+self.gen_timestep)]
            if (i + self.gen_timestep) < len(Xdata):
                X.append(x)
            else:
                break
        return np.array(X)

    def build_input(self):
        articleDBconnect = pymysql.connect(
            host=self.article["host"],
            port=self.article['port'],
            user=self.article['user'],
            password=self.article['password'],
            db=self.article['db'],
            charset=self.article['charset'],
            # cursorclass=pymysql.cursors.DictCursor
        )
        stockDBconnect = pymysql.connect(
            host=self.index["host"],
            port=self.index['port'],
            user=self.index['user'],
            password=self.index['password'],
            db=self.index['db'],
            charset=self.index['charset'],
            # cursorclass=pymysql.cursors.DictCursor
        )
        pv_model = gensim.models.Doc2Vec.load('vocab_20.model')
        #days = datetime.datetime(2011, 1, 2, 17, 0, 0)
        days = datetime.datetime(2010,1, 2, 15, 30, 0)
        np_pv = np.empty(self.pv_size)
        np_stock = np.empty(self.stock_size)
        trainY = np.empty(self.stock_size)
        trainSTOCK = np.empty(self.stock_size*(self.gen_timestep - 1))
        testY = np.empty(self.stock_size)
        testSTOCK = np.empty(self.stock_size*(self.gen_timestep - 1))
        while days.year < 2019:
        #for i in range(50):
            oneday, stock,days = self.numpy_one_day_data(articleDBconnect, stockDBconnect, days, pv_model)
            np_pv = np.vstack([np_pv,oneday])
            np_stock = np.vstack([np_stock,stock])
            #trainX = np.vstack([trainX, oneday])
            if len(stock) is not 0:
                trainY = np.vstack([trainY,stock])
        np_stock = np.delete(np_stock,(0),axis=0)
        np_pv = np.delete(np_pv,(0),axis=0)
        np_stock = self.scaler.fit_transform(np_stock)
        trainX = np.concatenate((np_stock,np_pv),axis=1)
        trainY = np.delete(trainY, (0), axis=0)
        for i in range(len(trainX) - self.gen_timestep):
            timestep_days = np.array(trainY[i: i + (self.gen_timestep-1)], copy = True)
            trainSTOCK = np.vstack([trainSTOCK, timestep_days.flatten()])
        trainSTOCK = np.delete(trainSTOCK,(0),axis=0)
        #print(trainY[0])
        trainY = trainY[self.gen_timestep:]
        trainX = self.changeLSTMsetX(trainX)
        trainY = self.scaler.transform(trainY)
        # TEST SET BUILDING
        print("------------ START BUILDING TEST SET ---------")
        np_pv = np.empty(self.pv_size)
        np_stock = np.empty(self.stock_size)
        while days.year < 2020:
        #for iter in range(15):
            oneday, stock,days = self.numpy_one_day_data(articleDBconnect, stockDBconnect, days, pv_model)
            #testX = np.vstack([testX, oneday])
            np_pv = np.vstack([np_pv, oneday])
            np_stock = np.vstack([np_stock, stock])
            if len(stock) is not 0:
                testY = np.vstack([testY,stock])
        np_stock = np.delete(np_stock, (0), axis=0)
        np_pv = np.delete(np_pv, (0), axis=0)
        np_stock = self.scaler.transform(np_stock)
        testX = np.concatenate((np_stock, np_pv), axis=1)
        testY = np.delete(testY, (0), axis=0)
        for i in range(len(testX) - self.gen_timestep):
            timestep_days = np.array(testY[i:(i+self.gen_timestep-1)],copy=True)
            testSTOCK = np.vstack([testSTOCK,timestep_days.flatten()])
        testSTOCK = np.delete(testSTOCK, (0), axis=0)
        testY = testY[self.gen_timestep:]
        testY = self.scaler.transform(testY)
        testX = self.changeLSTMsetX(testX)
        print("Data Setting DONE")

        print("Data Saving")
        pickle.dump(trainX,open('trainX.sav','wb'))
        pickle.dump(trainY,open('trainY.sav','wb'))
        pickle.dump(trainSTOCK,open('trainSTOCK.sav','wb'))
        pickle.dump(testX,open('testX.sav','wb'))
        pickle.dump(testY,open('testY.sav','wb'))
        pickle.dump(testSTOCK,open('testSTOCK.sav','wb'))
        pickle.dump(self.scaler,open('scaler.sav','wb'))
        print("Data Saving Done")
        return trainX,trainY,trainSTOCK,testX,testY,testSTOCK

    def train(self,epoch):
        self.generator.fit(self.trainX,self.trainY,epochs=epoch,batch_size=self.batch_size)


    # def predict(self, days):  # y hat
    #     articleDBconnect = pymysql.connect(
    #         host=self.article["host"],
    #         port=self.article['port'],
    #         user=self.article['user'],
    #         password=self.article['password'],
    #         db=self.article['db'],
    #         charset=self.article['charset'],
    #         # cursorclass=pymysql.cursors.DictCursor
    #     )
    #     stockDBconnect = pymysql.connect(
    #         host=self.index["host"],
    #         port=self.index['port'],
    #         user=self.index['user'],
    #         password=self.index['password'],
    #         db=self.index['db'],
    #         charset=self.index['charset'],
    #         # cursorclass=pymysql.cursors.DictCursor
    #     )
    #     pv_model = gensim.models.Doc2Vec.load('vocab_20.model')
    #     days = days - datetime.timedelta(days=self.gen_timestep)
    #     print(days)
    #     np_pv = np.empty(self.pv_size)
    #     np_stock = np.empty(self.stock_size)
    #
    #     for i in range(self.gen_timestep):
    #         oneday, stock, days = self.numpy_one_day_data(articleDBconnect, stockDBconnect, days, pv_model)
    #         np_pv = np.vstack([np_pv, oneday])
    #         np_stock = np.vstack([np_stock, stock])
    #
    #     np_stock = np.delete(np_stock, (0), axis=0)
    #     np_pv = np.delete(np_pv, (0), axis=0)
    #     print(np_pv.shape)
    #     np_stock = self.scaler.fit_transform(np_stock)
    #     trainX = np.concatenate((np_stock, np_pv), axis=1)
    #     print(trainX.shape)
    #     # print(trainY[0])
    #     trainX = trainX.reshape((-1,self.gen_timestep,self.gen_output))#self.changeLSTMsetX(trainX)
    #     print(trainX.shape)
    #     predict_result = self.generator.predict(trainX)
    #     return self.generator.predict(predict_result)
    #
    # def predict_testSet(self):
    #     testSet = self.testX
    #     predict_result = self.generator.predict(testSet).reshape(-1,self.gen_output)
    #     return self.scaler.inverse_transform(predict_result)
    #
    # def predict_trainSet(self):
    #     trainSet = self.trainX
    #     predict_result = self.generator.predict(trainSet).reshape(-1,self.gen_output)
    #     return self.scaler.inverse_transform(predict_result)
    #
    # def load(self,num):
    #     self.generator.load_weights('train/%s-%d.h5' % ("gen",num))
    #     self.discriminator.load_weights('train/%s-%d.h5' % ("dis",num))
    #     #self.combined.load_weights('train/%s-%d.h5' % ("GAN",num))
    # def save(self,num):
    #     #self.combined.save_weights('%s-%d.h5' % ("GAN",num))
    #     self.generator.save_weights("%s-%d.h5" % ("gen",num))
    #     self.discriminator.save_weights("%s-%d.h5" % ("dis",num))

    

if __name__ == '__main__':
    md = easyModel(batch_size=50)
    md.train(300)
    result = md.generator.predict(md.testX)
    answer = md.testY
    for idx in range(md.corp_num):
        result=np.insert(result,idx*5+1,0,axis=1)
        result=np.insert(result,idx*5+2,0,axis=1)
        result=np.insert(result,idx*5+3,0,axis=1)
        result=np.insert(result,idx*5+4,0,axis=1)
        answer=np.insert(answer, idx * 5 + 1, 0, axis=1)
        answer=np.insert(answer, idx * 5 + 2, 0, axis=1)
        answer=np.insert(answer, idx * 5 + 3, 0, axis=1)
        answer=np.insert(answer, idx * 5 + 4, 0, axis=1)
    result = md.scaler.inverse_transform(result)
    answer = md.scaler.inverse_transform(answer)
    print(result)
    print(answer)
    for idx in reversed(range(md.stock_size)):
        if idx % 5 is not 0:
            answer = np.delete(answer, idx, axis=1)
            result = np.delete(result,idx,axis=1)
    print(result)
    print(answer)
    print(np.sum(np.abs(result-answer)))

