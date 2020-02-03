import keras
from keras import layers
import pandas as pd
from pandas import DataFrame
import numpy as np
import pymysql
import gensim
import datetime

class GAN():
    def __init__(self):
        # LSTM Input : (795 + 792)*1
        # LSTM Output : (1000)
        # LSTM Output with time series : (1000 (Feature) * 10 (times))
        # discriminator Input : 1000 * 10
        # discriminator Output : 1 (0 or 1)
        self.pv_size = 792
        self.stock_size = 795
        self.gen_output = 795
        self.gen_timestep = 10
        self.gen_feature = self.pv_size+self.stock_size
        self.article = {"host": '127.0.0.1', "port": 3306,
                  "user": 'root', "password": 'sogangsp', "db": 'mydb', 'charset': 'utf8'}
        self.index = {"host": 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com', "port": 3306,
                  "user": 'admin', "password": 'sogangsp', "db": 'mydb', 'charset': 'utf8'}
        self.dis_input = self.gen_output * self.gen_timestep
        self.dis_output = 1

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator(LayerName='LSTM')
        print("Start Building Data")
        self.GAN_trainX,self.GAN_trainY,self.GAN_trainSTOCK,self.GAN_testX,self.GAN_testY,self.GAN_testSTOCK = self.build_input()
        print("Training Data Processing Finished")
        self.generator.compile(loss='mean_squared_error', optimizer='sgd')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))
        self.discriminator.trainable = False


        combined_input = keras.Input(shape=((self.gen_timestep),self.gen_feature),name='stock_news_input')
        #combined = 기사 + 주식
        gen_stock = self.generator(inputs=combined_input)
        #gen_stock : 795 차원의 10일 뒤 결과

        past_stock = keras.Input(shape=((self.gen_timestep-1),self.stock_size), name='past_stock')
        #past_stock : 과거 9일간의 데이터
        combined_stock = keras.layers.concatenate([past_stock, gen_stock],axis=1,name='combined_stock')
        #combined_stock : 총 10일간의 데이터 past + gen_stock

        valid = self.discriminator(inputs=combined_stock)
        self.combined = keras.Model(inputs=[combined_input,past_stock], outputs=valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))

    def build_generator(self, LayerName):
        model = keras.Sequential()
        if LayerName == 'LSTM':
            model.add(keras.layers.LSTM(self.gen_output, input_shape=(self.gen_timestep, self.gen_feature)))
        elif LayerName == 'GRU':
            model.add(keras.layers.GRU(self.gen_output, input_shape=(self.gen_timestep, self.gen_feature)))
        model.add(keras.layers.Reshape((1,self.gen_output)))
        model.summary()
        merged_input = keras.Input(shape=(self.gen_timestep,self.gen_feature))
        return keras.Model(merged_input, model(merged_input))

    def build_discriminator(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(32, kernel_size=5,strides=5,input_shape=(self.gen_timestep,self.stock_size),data_format='channels_first'))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Conv1D(32, kernel_size=3))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Conv1D(32, kernel_size=3))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(1))
        model.summary()
        estimated_sequence = keras.Input(shape=(self.gen_timestep,self.stock_size))
        return keras.Model(estimated_sequence, model(estimated_sequence))

    def numpy_one_day_data(self,newsDB,stockDB,days,pv):
        end = days
        start = days + datetime.timedelta(days=-1)
        news_sql = "select news from article where writetime >= %s and writetime < %s"
        stock_sql = """select CLOSE_PRICE, OPEN_PRICE, MAX_PRICE, MIN_PRICE, TRADE_AMOUNT
            from CORP_STOCK where TRADE_TIME >= %s and TRADE_TIME < %s and
            corp_name in (
            select distinct corp_name from CORP_STOCK 
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
                vec = pv.infer_vector(article)
                tempVec = tempVec + vec
            tempVec = tempVec / len(article_same_day)
            returnVec = returnVec + tempVec
            count += 1
            print(str(end)+' does not have stock data')
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
        tempVec = tempVec / len(article_same_day)
        returnVec = tempVec + returnVec
        returnVec = returnVec / count
        stockData = np.array(stockData)
        stockData = stockData.flatten(order='C')
        returnVec = np.concatenate((stockData, returnVec),axis=None)
        print(str(end)+' produced input')
        end = end + datetime.timedelta(days=1)
        return returnVec,stockData,end

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
        pv_model = gensim.models.Doc2Vec.load('paragraph_vector.model')
        days = datetime.datetime(2011, 1, 2, 17, 0, 0)
        trainX = np.empty(self.pv_size+self.stock_size)
        trainY = np.empty(self.stock_size)
        trainSTOCK = np.empty(self.stock_size*(self.gen_timestep - 1))
        testX = np.empty(self.pv_size+self.stock_size)
        testY = np.empty(self.stock_size)
        testSTOCK = np.empty(self.stock_size*(self.gen_timestep - 1))
        """while days.year < 2020:
            oneday,days = self.numpy_one_day_data(articleDBconnect,stockDBconnect,days,pv_model)
            trainX = np.vstack([input,oneday])
            print(input.shape)
        trainY = trainX[10:]"""
        for iter in range(50):
            oneday, stock,days = self.numpy_one_day_data(articleDBconnect, stockDBconnect, days, pv_model)
            trainX = np.vstack([trainX, oneday])
            #print(stock)
            #print(oneday)
            if len(stock) is not 0:
                trainY = np.vstack([trainY,stock])
        for i in range(len(trainX) - self.gen_timestep):
            timestep_days = trainY[i: i + (self.gen_timestep-1)]
            #print(timestep_days.shape)
            trainSTOCK = np.vstack([trainSTOCK, timestep_days.flatten()])
        trainY = trainY[9:]
        trainX = trainX[:40]


        for iter in range(15):
            oneday, stock,days = self.numpy_one_day_data(articleDBconnect, stockDBconnect, days, pv_model)
            testX = np.vstack([testX, oneday])
            if len(stock) is not 0:
                testY = np.vstack([testY,stock])
        for i in range(len(trainX) - self.gen_timestep):
            timestep_days = np.copy(trainY[i:(i+self.gen_timestep-1)])
            testSTOCK = np.vstack([testSTOCK,timestep_days.flatten()])
        testY = testY[9:]
        testX = testX[:5]

        print("Data Setting DONE")
        return trainX,trainY,trainSTOCK,testX,testY,testSTOCK

    def train(self, epochs, batch_size=128):
        # Rescale Data
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            print("epoch : " + str(epoch))

            gen_input = self.GAN_trainX[epoch*batch_size:(epoch+1)*batch_size]
            gen_input = gen_input.reshape((int(gen_input.shape[0]/self.gen_timestep),self.gen_timestep,self.gen_feature))
            print(gen_input.shape)
            print("input")
            print(gen_input)
            gen_answer = self.GAN_trainY[epoch*batch_size:(epoch+1)*batch_size]
            gen_answer = gen_answer.reshape((int(gen_answer.shape[0] / self.gen_timestep), self.gen_timestep,self.gen_output))
            gen_stock = self.GAN_trainSTOCK[epoch*batch_size:(epoch+1)*batch_size]
            gen_stock_output = self.generator.predict(gen_input)

            print(gen_stock.shape)
            print(gen_answer.shape)
            print(gen_stock_output.shape)

            print("9일치 stock")
            print(gen_stock)
            print("answer")
            print(gen_answer)
            print("prediction")
            print(gen_stock_output)

            gen_stock_output = gen_stock_output.reshape(batch_size,)
            gen_stock = gen_answer.reshape(batch_size,((self.gen_timestep-1)*self.stock_size))

            gen_dis_real_input = np.hstack([gen_stock,gen_answer])#np.append(gen_stock,gen_answer,axis=1)
            gen_dis_fake_input = np.hstack([gen_stock,gen_stock_output])#np.append(gen_stock,gen_stock_output,axis=1)
            d_loss_real = self.discriminator.train_on_batch(gen_dis_real_input,np.ones((batch_size),1))
            d_loss_fake = self.discriminator.train_on_batch(gen_dis_fake_input,np.zeros((batch_size,1)))
            #half and batch size의 문제
            d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)

            g_loss = self.combined.train_on_batch([gen_input,gen_stock],gen_answer)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))


    def predict(self, today):  # y hat
        return self.generator.predict(today)

gan = GAN()
gan.train(epochs=100, batch_size=20)
gan.predict()
