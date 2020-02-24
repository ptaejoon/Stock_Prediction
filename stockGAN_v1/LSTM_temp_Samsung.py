#!/usr/bin/env python
# coding: utf-8

# In[21]:


# mysql로부터 data import 
import pandas as pd
import pymysql.cursors
import numpy as np

# 주식 차트를 위한 import 
import datetime
import mpl_finance 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#주가예측을 위한 import   - 방법 keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


#주가 예측을 위한 import - 방법 tf

import tensorflow as tf


# In[22]:


# 삼성전자 1종목 10년치 data 출력 
connection = pymysql.connect(host = 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com'
                             ,user ='admin'
                             ,password = 'sogangsp'
                             ,db = 'mydb'
                             ,charset = 'utf8'
                             ,autocommit = True)

cursor = connection.cursor()
sql = 'select * from CORP_STOCK where CORP_NAME = "삼성전자" order by TRADE_TIME '
cursor.execute(sql)

data = cursor.fetchall()
connection.close()

data_frame = pd.DataFrame(data,columns=["CORP_NAME","TRADE_TIME","CLOSE_PRICE","DIFF","OPEN_PRICE","MAX_PRICE","MIN_PRICE","TRADE_AMOUNT"])


# In[23]:


samsung = data_frame.drop(labels="CORP_NAME",axis=1)


# In[24]:


samsung_inx_day= samsung.set_index("TRADE_TIME")


# In[25]:


samsung_inx_day.head()


# In[26]:


day_list = []
for day in samsung_inx_day.index:
    day_list.append(day.strftime('%Y-%m'))
    
day_list_cnt = range(len(samsung_inx_day))
day_list_cnt


# In[27]:


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(111)
#ax.xaxis.set_major_locator(ticker.FixedLocator(day_list_cnt))
#ax.xaxis.set_major_formatter(ticker.FixedFormatter(day_list))

mpl_finance.candlestick2_ohlc(ax, samsung_inx_day['OPEN_PRICE']
                             ,samsung_inx_day['MAX_PRICE']
                             ,samsung_inx_day['MIN_PRICE']
                             ,samsung_inx_day['CLOSE_PRICE']
                             ,width = 0.5 , colorup = 'r' , colordown = 'b')


# In[28]:


# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터를 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)

tf.set_random_seed(777)


# In[29]:


# Standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()

# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# Min-Max scaling

def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원
 

def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


# In[30]:


input_data_column_cnt = 5  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 1 # 결과데이터의 컬럼 개수

seq_length = 28        # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20   # 각 셀의 (hidden)출력 크기
forget_bias = 1.0          # 망각편향(기본값 1.0)
num_stacked_layers = 1     # stacked LSTM layers 개수
keep_prob = 1.0            # dropout할 때 keep할 비율
 
epoch_num = 1000           # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01       # 학습률


# In[31]:


samsung_stock=samsung_inx_day.copy()


# In[32]:


samsung_stock.drop(labels = 'DIFF',axis = 1,inplace = True)


# In[33]:


samsung_stock_data = samsung_stock.values[:].astype(np.float)


# In[34]:


samsung_stock_data


# In[35]:


# 가격데이터 정규화 처리
price = samsung_stock_data[:,:-1]
norm_price = min_max_scaling(price)
norm_price


# In[36]:


# 거래량데이터 정규화 처리
volume = samsung_stock_data[:,-1:]
norm_volume = min_max_scaling(volume)
norm_volume


# In[37]:


x =np.concatenate((norm_price ,norm_volume),axis=1) 
print("x.shape: ", x.shape)
print("x[0]: ", x[0])    # x의 첫 값
print("x[-1]: ", x[-1])  # x의 마지막 값

y = x[:, [-2]] # 타켓은 주식 종가이다
print("y[0]: ",y[0])     # y의 첫 값
print("y[-1]: ",y[-1])   # y의 마지막 값


# In[38]:


dataX = [] # 입력으로 사용될 Sequence Data
dataY = [] # 출력(타켓)으로 사용


# In[39]:


for i in range(0, len(y) - seq_length):
    _x = x[i : i+seq_length]
    _y = y[i + seq_length] # 다음 나타날 주가(정답)
    if i is 0:
        print(_x, "->", _y) # 첫번째 행만 출력해 봄
    dataX.append(_x) # dataX 리스트에 추가
    dataY.append(_y) # dataY 리스트에 추가


# In[40]:


# 학습용/테스트용 데이터 생성
# 전체 70%를 학습용 데이터로 사용
train_size = int(len(dataY) * 0.7)

# 나머지(30%)를 테스트용 데이터로 사용
test_size = len(dataY) - train_size

# 데이터를 잘라 학습용 데이터 생성
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

# 데이터를 잘라 테스트용 데이터 생성
testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])


# In[41]:


# 텐서플로우 플레이스홀더 생성
# 입력 X, 출력 Y를 생성한다
X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, 1])
print("Y: ", Y)

 

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)

predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions)


# In[42]:


# 모델(LSTM 네트워크) 생성

def lstm_cell():

    # LSTM셀을 생성
    # num_units: 각 Cell 출력 크기
    # forget_bias:  to the biases of the forget gate 
    #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim, 
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


# In[43]:


# num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()


# In[44]:


# RNN Cell(여기서는 LSTM셀임)들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)


# In[45]:


# [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.

# 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다

hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

 

 

# 손실함수로 평균제곱오차를 사용한다

loss = tf.reduce_sum(tf.square(hypothesis - Y))

# 최적화함수로 AdamOptimizer를 사용한다

optimizer = tf.train.AdamOptimizer(learning_rate)

# optimizer = tf.train.RMSPropOptimizer(learning_rate) # LSTM과 궁합 별로임

 

train = optimizer.minimize(loss)

 

# RMSE(Root Mean Square Error)

# 제곱오차의 평균을 구하고 다시 제곱근을 구하면 평균 오차가 나온다

# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions))) # 아래 코드와 같다

rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

 

 

train_error_summary = [] # 학습용 데이터의 오류를 중간 중간 기록한다

test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다

test_predict = ''        # 테스트용데이터로 예측한 결과

 

sess = tf.Session()

sess.run(tf.global_variables_initializer())

 

# 학습한다

start_time = datetime.datetime.now() # 시작시간을 기록한다

print('학습을 시작합니다...')

for epoch in range(epoch_num):

    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})

    if ((epoch+1) % 100 == 0) or (epoch == epoch_num-1): # 100번째마다 또는 마지막 epoch인 경우

        # 학습용데이터로 rmse오차를 구한다

        train_predict = sess.run(hypothesis, feed_dict={X: trainX})

        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})

        train_error_summary.append(train_error)

 

        # 테스트용데이터로 rmse오차를 구한다

        test_predict = sess.run(hypothesis, feed_dict={X: testX})

        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})

        test_error_summary.append(test_error)

        

        # 현재 오류를 출력한다

        print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch+1, train_error, test_error, test_error-train_error))

        

end_time = datetime.datetime.now() # 종료시간을 기록한다

elapsed_time = end_time - start_time # 경과시간을 구한다

print('elapsed_time:',elapsed_time)

print('elapsed_time per epoch:',elapsed_time/epoch_num)

 

 

# 하이퍼파라미터 출력

print('input_data_column_cnt:', input_data_column_cnt, end='')

print(',output_data_column_cnt:', output_data_column_cnt, end='')

 

print(',seq_length:', seq_length, end='')

print(',rnn_cell_hidden_dim:', rnn_cell_hidden_dim, end='')

print(',forget_bias:', forget_bias, end='')

print(',num_stacked_layers:', num_stacked_layers, end='')

print(',keep_prob:', keep_prob, end='')

 

print(',epoch_num:', epoch_num, end='')

print(',learning_rate:', learning_rate, end='')

 

print(',train_error:', train_error_summary[-1], end='')

print(',test_error:', test_error_summary[-1], end='')

print(',min_test_error:', np.min(test_error_summary))

 

# 결과 그래프 출력

plt.figure(1)

plt.plot(train_error_summary, 'gold')

plt.plot(test_error_summary, 'b')

plt.xlabel('Epoch(x100)')

plt.ylabel('Root Mean Square Error')

 

plt.figure(2)

plt.plot(testY, 'r')

plt.plot(test_predict, 'b')

plt.xlabel('Time Period')

plt.ylabel('Stock Price')

plt.show()

 

 

# sequence length만큼의 가장 최근 데이터를 슬라이싱한다

recent_data = np.array([x[len(x)-seq_length : ]])

print("recent_data.shape:", recent_data.shape)

print("recent_data:", recent_data)

 

# 내일 종가를 예측해본다

test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

 

print("test_predict", test_predict[0])

test_predict = reverse_min_max_scaling(price,test_predict) # 금액데이터 역정규화한다

print("Tomorrow's stock price", test_predict[0]) # 예측한 주가를 출력한다

 


# In[46]:


tf.reset_default_graph()


# In[254]:





# In[ ]:




