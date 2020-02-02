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
        self.gen_input = self.pv_size + self.stock_size
        self.gen_output = 795
        self.gen_timestep = 10
        self.gen_feature = 795
        self.article = {"host": '127.0.0.1', "port": 3306,
                  "user": 'root', "password": 'sogangsp', "db": 'mydb', 'charset': 'utf8'}
        self.index = {"host": 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com', "port": 3306,
                  "user": 'admin', "password": 'sogangsp', "db": 'mydb', 'charset': 'utf8'}
        self.dis_input = self.gen_output * self.gen_timestep
        self.dis_output = 1

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator(LayerName='LSTM')

        self.GAN_input = self.build_input()

        self.generator.compile(loss='mean_squared_error', optimizer='sgd')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))
        self.discriminator.trainable = False

        combined_input = keras.Input(shape=(self.gen_input,))
        gen_stock = self.generator(combined_input)
        past_stock = keras.Input(shape=(self.gen_feature * (self.gen_timestep - 1)), name='past_stock')
        combined_stock = keras.layers.concatenate([past_stock, gen_stock])
        valid = self.discriminator(combined_stock)
        self.combined = keras.Model(combined_stock, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))

    def build_generator(self, LayerName):
        model = keras.Sequential()
        if LayerName == 'LSTM':
            model.add(keras.layers.LSTM(self.gen_output, input_shape=(self.gen_timestep, self.gen_feature)))
        elif LayerName == 'GRU':
            model.add(keras.layers.GRU(self.gen_output, input_shape=(self.gen_timestep, self.gen_feature)))
        model.summary()
        merged_input = keras.Input(shape=(self.gen_timestep,self.gen_feature))
        return keras.Model(merged_input, model(merged_input))

    def build_discriminator(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(32, kernel_size=5, input_shape=(self.gen_timestep,self.gen_feature), strides=5,data_format='channels_first'))
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
        estimated_sequence = keras.Input(shape=(self.gen_timestep,self.gen_feature))
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
        return returnVec,end

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
        input = np.zeros(self.pv_size+self.stock_size)
        while days.year < 2020:
            oneday,days = self.numpy_one_day_data(articleDBconnect,stockDBconnect,days,pv_model)
            input = np.vstack([input,oneday])
            print(input.shape)
        return oneday

    def splitTrainTest(self):
        print("1")
        train = []
        test = []
        return train,test

    def train(self, epochs, batch_size=128, save_interval=50):
        # Load Data
        # Rescale Data
        trian,test = self.splitTrainTest()
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            print("temp")
        # ------------
        # train Discriminator
        # select half as generator's value
        # select half as real value
        # gen_result = self.generator.predict()
        # past_stock
        # dis_loss_real = self.discriminator.train_on_batch(past_stock+present_stock,np.ones((half_batch,1))
        # dis_loss_gen = self.discriminator.train_on_batch(past_stock+gen_result,np.zeros((half_batch,1))
        # dis_loss = 0.5 * np.add(dis_loss_real, dis_loss_gen)
        # ------------

        # ----------------
        # load stock data
        # load paragraph vector
        # gen_loss = self.combined.train_on_batch(,)
        # train Generator
        # self.generator.predict_on_batch(self,)
        # ---------------

    def predict(self, today):  # y hat
        return self.generator.predict(today)

gan = GAN()