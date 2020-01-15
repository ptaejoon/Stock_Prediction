import keras
from keras import layers
def mergeIndexandArticle(index,article,outputSize,inputSize):
    #merge stock index vecot and article vector
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(outputSize, input_dim=inputSize, activation='relu'))
    return model

def RecurrentModel(model,addingLayer,outputSize,timestep,feature):
    if addingLayer == 'LSTM':
        model.add(keras.layers.LSTM(outputSize,input_shape=(timestep,feature),return_sequences=False))
    elif addingLayer == 'GRU':
        model.add(keras.layers.GRU(outputSize,input_shape=(timestep,feature)))
    return model

inputArticle = []
inputIndex = []
LSTMmodel = mergeIndexandArticle(inputIndex,inputArticle,1000,100)
LSTMmodel = RecurrentModel(LSTMmodel,'LSTM',5,10,1000)

def build_discriminator():
    model = keras.Sequential()
    model.add()
    model.add()
    model.add()
