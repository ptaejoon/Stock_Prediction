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
import tensorflow as tf
from tensorflow import keras

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

class GAN():
    def __init__(self,batch_size):
        #self.khaiii = KhaiiiApi()
        # LSTM Input : (795 + 792)*1
        # LSTM Output : (1000)
        # LSTM Output with time series : (1000 (Feature) * 10 (times))
        # discriminator Input : 1000 * 10
        # discriminator Output : 1 (0 or 1)
        self.batch_size = batch_size
        self.pv_size = 792
        self.stock_size = 795
        self.gen_output = 795
        self.gen_timestep = 10
        #self.scaler = MinMaxScaler()
        self.gen_feature = self.pv_size+self.stock_size
        self.article = {"host": '127.0.0.1', "port": 3306,
                  "user": 'sinunu', "password": '1q2w3e4r', "db": 'mydb', 'charset': 'utf8'}
        self.index = {"host": '127.0.0.1', "port": 3306,
                  "user": 'sinunu', "password": '1q2w3e4r', "db": 'mydb', 'charset': 'utf8'}
        self.dis_input = self.gen_output * self.gen_timestep
        self.dis_output = 1
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator(LayerName='LSTM')
        print("Start Building Data")
        if os.path.isfile('trainX.sav') is True:
            print("Data Already Exists. Start Loading")
            self.GAN_trainX = pickle.load(open('trainX.sav','rb'))
            self.GAN_trainY = pickle.load(open('trainY.sav','rb'))
            self.GAN_trainSTOCK = pickle.load(open('trainSTOCK.sav','rb'))
            self.GAN_testX = pickle.load(open('testX.sav','rb'))
            self.GAN_testY = pickle.load(open('testY.sav','rb'))
            self.GAN_testSTOCK = pickle.load(open('testSTOCK.sav','rb'))
            self.scaler = pickle.load(open('scaler.sav','rb'))
            trainX_transformed = self.scaler.inverse_transform(self.GAN_trainX[:,:,:self.gen_output].reshape(-1,self.gen_output))
            testX_transformed = self.scaler.inverse_transform(self.GAN_testX[:,:,:self.gen_output].reshape(-1,self.gen_output))
            self.GAN_trainY = self.scaler.inverse_transform(self.GAN_trainY)
            self.GAN_testY = self.scaler.inverse_transform(self.GAN_testY)
            #print(trainX_transformed.shape)
            scaling_data = np.vstack([trainX_transformed,testX_transformed,self.GAN_testY,self.GAN_trainY])
            self.scaler = MinMaxScaler(feature_range=(0,1))
            self.scaler.fit(scaling_data)
            self.scaler.data_min_ = [0] * self.gen_output
            self.scaler.data_max_ = self.scaler.data_max_*1.3
            #print(self.scaler.data_min_)
            #print(self.scaler.data_max_)
            #print(self.scaler.data_min_)
            self.GAN_trainX[:,:,:self.gen_output] = self.scaler.transform(trainX_transformed).reshape(-1,self.gen_timestep,self.gen_output)
            self.GAN_testX[:,:,:self.gen_output] = self.scaler.transform(testX_transformed).reshape(-1,self.gen_timestep,self.gen_output)

            self.GAN_trainY = self.scaler.transform(self.GAN_trainY)
            self.GAN_testY = self.scaler.transform(self.GAN_testY)
            self.GAN_trainSTOCK = self.GAN_trainX[:,:,:self.gen_output].reshape((-1,self.gen_timestep,self.gen_output)).copy()
            self.GAN_testSTOCK = self.GAN_testX[:,:,:self.gen_output].reshape((-1,self.gen_timestep,self.gen_output)).copy()

            self.GAN_trainSTOCK = self.GAN_trainSTOCK.reshape((-1, ((self.gen_timestep) * self.gen_output)))
            self.GAN_testSTOCK = self.GAN_testSTOCK.reshape((-1, ((self.gen_timestep) * self.gen_output)))

            #print(self.scaler.data_min_)

        else:
            print("Data Doesn't Exists. Start Building")
            self.GAN_trainX,self.GAN_trainY,self.GAN_trainSTOCK,self.GAN_testX,self.GAN_testY,self.GAN_testSTOCK = self.build_input()
        print("Training Data Processing Finished")
        #self.generator.compile(loss=root_mean_squared_error,
                #optimizer=keras.optimizers.Adam(0.00002, 0.5),metrics=['accuracy'])
        self.discriminator.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(0.0000003, 0.5),metrics=['accuracy'])
        self.discriminator.trainable = False

        combined_input = keras.Input(shape=((self.gen_timestep),self.gen_feature),name='stock_news_input')
        #combined = 기사 + 주식
        gen_stock = self.generator(inputs=combined_input)
        #gen_stock : 795 차원의 10일 뒤 결과

        past_stock = keras.Input(shape=(self.stock_size,self.gen_timestep), name='past_stock')
        #past_stock : 과거 9일간의 데이터
        combined_stock = keras.layers.concatenate(inputs=[past_stock, gen_stock],axis=2,name='combined_stock')
        #combined_stock : 총 10일간의 데이터 past + gen_stock

        valid = self.discriminator(inputs=combined_stock)
        self.combined = keras.models.Model(inputs=[combined_input, past_stock], outputs=valid)

        self.combined.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0000001, 0.5))
        self.combined.summary()
        if os.path.isfile('GAN.h5') is True:
            self.load()

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
        model.add(keras.layers.Reshape((self.gen_output,1)))
        model.summary()
        merged_input = keras.Input(shape=(self.gen_timestep,self.gen_feature))
        return keras.models.Model(merged_input, model(merged_input),name='generator')

    def build_discriminator(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(32, kernel_size=4,strides=2,input_shape=(self.stock_size,self.gen_timestep+1)))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Conv1D(64, kernel_size=4,strides=2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Conv1D(128, kernel_size=4,strides=2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        model.summary()
        estimated_sequence = keras.Input(shape=(self.stock_size,self.gen_timestep+1))
        return keras.models.Model(estimated_sequence, model(estimated_sequence),name='disciriminator')

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
                #article_token = tokenize(article[0])
                #vec = pv.infer_vector(article_token)
                vec = pv.infer_vector(article)
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

    def train(self,time):
        # Rescale Data
        batch_size = self.batch_size
        batch = int(self.GAN_trainX.shape[0]/batch_size)
        #print(self.GAN_trainSTOCK.shape)
        #print(self.GAN_trainSTOCK.shape)
#        self.GAN_trainSTOCK = self.scaler.transform(self.GAN_trainSTOCK)
        half_batch = int(batch_size / 2)
        self.GAN_trainSTOCK = self.GAN_trainSTOCK.reshape((-1,self.gen_timestep,self.gen_output))
        for times in range(time):
            print("epoch :"+str(times))
            for epoch in range(batch):

                gen_input = self.GAN_trainX[epoch*batch_size:(epoch+1)*batch_size]
                gen_answer = self.GAN_trainY[epoch*batch_size:(epoch+1)*batch_size]
                gen_answer = gen_answer.reshape((-1, 1, self.gen_output))
                gen_stock = self.GAN_trainSTOCK[epoch*batch_size:(epoch+1)*batch_size]
                gen_stock_output = self.generator.predict(gen_input)
                gen_stock_output = gen_stock_output.reshape((-1,1,self.gen_output))
                gen_dis_real_input = np.hstack([gen_stock,gen_answer])#np.append(gen_stock,gen_answer,axis=1)
                gen_dis_fake_input = np.hstack([gen_stock,gen_stock_output])#np.append(gen_stock,gen_stock_output,axis=1)
                gen_dis_fake_input = gen_dis_fake_input.reshape((-1,self.gen_output,self.gen_timestep+1))
                gen_dis_real_input = gen_dis_real_input.reshape((-1,self.gen_output,self.gen_timestep+1))
                gen_stock = gen_stock.reshape((-1,self.gen_output,self.gen_timestep))
                d_loss_real = self.discriminator.train_on_batch(gen_dis_real_input,np.ones((batch_size,1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_dis_fake_input,np.zeros((batch_size,1)))
                d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)
                valid = np.array([1]*batch_size)
                g_loss = self.combined.train_on_batch([gen_input,gen_stock],valid)#gen_answer)
                print(str(epoch) + ' [D real loss : ' + str(d_loss_real[0]) + ', D fake loss: '+str(d_loss_fake[0])+' D real acc : ' + str(d_loss_real[1])+' D fake acc : '+str(d_loss_fake[1])+' D total acc :' +
                     str(100 * d_loss[1]) + '] [ G loss : ' + str(g_loss))

            predict_result = self.generator.predict(self.GAN_trainX).reshape(-1, self.gen_output)

            if times % 10 == 0:
                self.save(times)
            print(self.scaler.inverse_transform(predict_result))
            print(self.scaler.inverse_transform(self.GAN_trainY))
            #print(self.predict_testSet())
            #print(self.scaler.inverse_transform(self.GAN_testY))
            print(np.sum(np.abs((self.scaler.inverse_transform(predict_result) - self.scaler.inverse_transform(self.GAN_trainY)))))

    def predict(self, days):  # y hat
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
        days = days - datetime.timedelta(days=self.gen_timestep)
        print(days)
        np_pv = np.empty(self.pv_size)
        np_stock = np.empty(self.stock_size)

        for i in range(self.gen_timestep):
            oneday, stock, days = self.numpy_one_day_data(articleDBconnect, stockDBconnect, days, pv_model)
            np_pv = np.vstack([np_pv, oneday])
            np_stock = np.vstack([np_stock, stock])

        np_stock = np.delete(np_stock, (0), axis=0)
        np_pv = np.delete(np_pv, (0), axis=0)
        print(np_pv.shape)
        np_stock = self.scaler.fit_transform(np_stock)
        trainX = np.concatenate((np_stock, np_pv), axis=1)
        print(trainX.shape)
        # print(trainY[0])
        trainX = trainX.reshape((-1,self.gen_timestep,self.gen_output))#self.changeLSTMsetX(trainX)
        print(trainX.shape)
        predict_result = self.generator.predict(trainX)
        return self.generator.predict(predict_result)

    def predict_testSet(self):
        testSet = self.GAN_testX
        predict_result = self.generator.predict(testSet).reshape(-1,self.gen_output)
        return self.scaler.inverse_transform(predict_result)

    def predict_trainSet(self):
        trainSet = self.GAN_trainX
        predict_result = self.generator.predict(trainSet).reshape(-1,self.gen_output)
        return self.scaler.inverse_transform(predict_result)

    def save(self,epoch):
        self.combined.save_weights('train/%s-%d.h5' % ("GAN", epoch))
        self.generator.save_weights('train/%s-%d.h5' % ("gen", epoch))
        self.discriminator.save_weights('train/%s-%d.h5' % ("dis", epoch))
        # self.discriminator.trainable = False
        # #keras.models.save_model(self.combined,'GAN.h5')
        # #self.combined.save_weights('GANWeight.h5')
        # self.combined.save('GAN.h5')
        # self.discriminator.trainable = True
        # #keras.models.save_model(self.generator, 'gen.h5')
        # self.generator.save('gen.h5')
        # #self.generator.save_weights('genWeight.h5')
        # #keras.models.save_model(self.discriminator, 'dis.h5')
        # self.discriminator.save('dis.h5')
        # #self.discriminator.save_weights('disWeight.h5')
        print("Save Done")

    def load(self,epoch):
        self.combined.load_weights('train/%s-%d.h5' % ("GAN", epoch))
        self.generator.load_weights('train/%s-%d.h5' % ("gen", epoch))
        self.discriminator.load_weights('%train/%s-%d.h5' % ("dis", epoch))
        # self.discriminator = tf.keras.models.load_model('dis.h5')
        # #self.discriminator.load_weights('disWeight.h5')
        # self.generator = tf.keras.models.load_model('gen.h5')
        # #self.generator.load_weights('genWeight.h5')
        # #self.discriminator.compile(loss='binary_crossentropy',
        # #                           optimizer=keras.optimizers.Adam(0.000001, 0.5), metrics=['accuracy'])
        # self.discriminator.trainable = False
        # self.combined = tf.keras.models.load_model('GAN.h5')
        # #self.combined.load_weights('GANWeight.h5')
        # #self.combined.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.000001, 0.5))

        print("loading Model done")
    # def save(self):
    #     self.discriminator.trainable = False
    #     tf.keras.models.save_model()




if __name__ == '__main__':
    #print(gan.scaler.inverse_transform(gan.GAN_testY))
    if os.path.isfile('train/GAN-0.h5') is False:
        gan = GAN(batch_size=20)
        gan.train(50)
    else :
        gan = GAN(batch_size=40)
        gan.load(50)
        gan.train(5)
        #gan = pickle.load(open('GAN.sav', 'rb'))
    #days = datetime.datetime(2020, 1, 2, 15, 30, 0)
    #price2019 = gan.predict_testSet()
    #print(price2019)

