import keras
from keras import layers
from pandas import DataFrame
import pymysql


class GAN():
    def __init__(self, paragraph_vector, stock_index):
        # LSTM Input : (1000+1024)*1
        # LSTM Output : (1000)
        # LSTM Output with time series : (1000 (Feature) * 10 (times))
        # discriminator Input : 1000 * 10
        # discriminator Output : 1 (0 or 1)
        self.gen_input = 2024
        self.gen_output = 1000
        self.gen_timestep = 10
        self.gen_feature = 1000

        self.dis_input = self.gen_output * self.gen_timestep
        self.dis_output = 1

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator(LayerName='LSTM')

        self.GAN_input = self.build_input(pv=paragraph_vector, index=stock_index)

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
        merged_input = keras.Input(shape=(self.gen_input,))
        return keras.Model(merged_input, model(merged_input))

    def build_discriminator(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(32, kernel_size=5, input_shape=(self.gen_timestep, self.gen_feature), strides=5))
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
        estimated_sequence = keras.Input(shape=(self.dis_input,))
        return keras.Model(estimated_sequence, model(estimated_sequence))

    def build_input(self, pv, index):
        input = pv.set_index('date').join(index.set_index('date'))
        return input

    def train(self, epochs, batch_size=128, save_interval=50):
        # Load Data
        # Rescale Data

        half_batch = int(batch_size / 2)
        for epoch in range(epochs):

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

"""
processedDB = {"host": 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com', "port": 3306, "user": 'admin',
               "password": "sogangsp", "db": "mydb", 'charset': 'utf8'}
rawDB = {"host": 'article-raw-data.cnseysfqrlcj.ap-northeast-2.rds.amazonaws.com', "port": 3306, "user": 'admin',
         "password": "sogangsp", "db": "mydb", 'charset': 'utf8'}

evenDBconnect = pymysql.connect(
    host=processedDB["host"],
    port=processedDB['port'],
    user=processedDB['user'],
    password=processedDB['password'],
    db=processedDB['db'],
    charset=processedDB['charset']
)

oddDBconnect = pymysql.connect(
    host=rawDB["host"],
    port=rawDB['port'],
    user=rawDB['user'],
    password=rawDB['password'],
    db=rawDB['db'],
    charset=rawDB['charset']
)
evenCursor = evenDBconnect.cursor()
oddCursor = oddDBconnect.cursor()


def companySelection():
    company_list_sql = select distinct corp_name from CORP_STOCK where TRADE_TIME < '2010-01-05 00:00:00.000'
    
"""