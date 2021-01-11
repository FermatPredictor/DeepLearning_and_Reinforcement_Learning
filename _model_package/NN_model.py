# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist #手寫數字資料集
from keras.utils import np_utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD #stochastic gradient descent


class NN_model():
    """
    keras version: 2.2.4
    tensorflow version: 2.4.0
    
    Use the command to see your vesion:
    print(keras.__version__)
    print(tensorflow.__version__)    
    
    建立一個神經網路實現分類問題。
    NN需要將input拉成一維的資料，
    out用one-hot encoding呈現
    """
    
    def __init__(self, input_size:int, output_size:int):
        model = Sequential() #建立空的神經網路

        model.add(Dense(500, input_dim=input_size, activation='sigmoid'))
        model.add(Dense(500, activation='sigmoid'))        
        model.add(Dense(output_size, activation='softmax'))
        
        """
        組裝神經網路
        loss: 損失函數
        optimizers: 訓練方式(lr是learning rate)
        metrics: 評分標準
        """
        model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
        model.summary() #檢示神經網路的架構
        self.model = model
    

    def fit(self, x_train, y_train, batch_size=100, epochs=20):
        """
        訓練神經網路
        batch_size: 一次訓練幾筆資料(每幾筆資料調一次參數)
        epochs: 資料共訓練幾次
        """
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        
    def predict_classes(self, x_test):
        predict = self.model.predict(x_test)
        return np.argmax(predict, axis=-1)
    
    def save(self, path):
        """ 將訓練好的模型存為HDF5格式, 副檔名.h5 """
        self.model.save(path)
        
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

        
if __name__=='__main__':
    #caller example

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    plt.imshow(x_train[9487])
    print(y_train[9487])
    
    x_train = x_train.reshape(60000,784)
    x_test = x_test.reshape(10000,784)
    y_train = np_utils.to_categorical(y_train,10) #one-hot encoding
    y_test = np_utils.to_categorical(y_test,10) #one-hot encoding
    
    model = NN_model(784,10)
    model.fit(x_train, y_train)
    
    predict = model.predict_classes(x_test)
    score = model.evaluate(x_test, y_test)
    print('測試資料的loss', score[0])
    print('測試資料的正確率', score[1])
    
    if score[1] > 0.9:
        model.save('./NN_handwrite_model.h5')

