# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist #手寫數字資料集
from keras.utils import np_utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


class CNN_model():
    """
    keras version: 2.2.4
    tensorflow version: 2.4.0
    
    Use the command to see your vesion:
    print(keras.__version__)
    print(tensorflow.__version__)    
    
    建立一個神經網路實現分類問題。
    與NN相比，CNN是圖形辨識的高手，
    注意input的維度是三維(ex: (28,28,1), (28,28,3))
    output用one-hot encoding呈現
    """
    
    def __init__(self, input_shape:tuple, output_size:int):
        model = Sequential() #建立空的神經網路
        
        """
        padding='same' 意思是做完convolution後矩陣不要縮水
        另，CNN的技巧是愈後面的層數神經元愈多。
        做完CNN和Pooling層後，會拉平送回正常的NN
        """
        model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))    
        model.add(Dense(output_size, activation='softmax'))

        
        """
        組裝神經網路
        loss: 損失函數
        optimizers: 訓練方式(lr是learning rate)
        metrics: 評分標準
        """
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.summary() #檢示神經網路的架構
        self.model = model
    

    def fit(self, x_train, y_train, batch_size=100, epochs=5):
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
    
    x_train = x_train.reshape(60000,28,28,1)/255 #資料標準化映射至0~1之間，有助訓練
    x_test = x_test.reshape(10000,28,28,1)/255
    y_train = np_utils.to_categorical(y_train,10) #one-hot encoding
    y_test = np_utils.to_categorical(y_test,10) #one-hot encoding
    
    model = CNN_model((28,28,1),10)
    model.fit(x_train, y_train)
    
    predict = model.predict_classes(x_test)
    score = model.evaluate(x_test, y_test)
    print('測試資料的loss', score[0])
    print('測試資料的正確率', score[1])
    
    if score[1] > 0.98:
        model.save('./CNN_handwrite_model.h5')

