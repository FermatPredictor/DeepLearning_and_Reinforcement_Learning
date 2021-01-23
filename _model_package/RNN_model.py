# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist #手寫數字資料集
from tensorflow.keras.datasets import imdb #影評資料集
from keras.utils import np_utils
from keras.preprocessing import sequence

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM # Long-Short-term memory layer，有長期記憶的RNN模型
from tensorflow.keras.optimizers import SGD #stochastic gradient descent


class RNN_model():
    """
    keras version: 2.2.4
    tensorflow version: 2.4.0
    
    Use the command to see your vesion:
    print(keras.__version__)
    print(tensorflow.__version__)    
    
    用RNN實現分類問題，
    out用one-hot encoding呈現
    """
    
    def __init__(self, input_size:int, output_size:int):
        
        model = Sequential() #建立空的神經網路
        """ 
        Embedding: 原本每個單字用一個1~10000的數字來表示，
        (以one-hot encoding的角度來看就是一萬維向量)
        Embedding 技術可以將它降維，並且語意相近的單字向量也會接近。
        Embedding 可以將離散的數據變的連續
        """
        model.add(Embedding(input_size,128))
        model.add(LSTM(150))
        model.add(Dropout(0.2)) # Dropout層防止過度擬合，斷開比例:0.2
        model.add(Dense(output_size, activation='softmax'))
        
        """
        組裝神經網路
        loss: 損失函數
        optimizers: 訓練方式(lr是learning rate)
        metrics: 評分標準
        """
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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


def vector_to_sentence(word_vec):
    """
    函數功能: 將imdb資料集的單字向量如[1, 14, 22, 16, 43, 530, 973, 2, 2, 65, ...]，
    透過字典映射轉為原本的英文句子(不過標點符號有缺失)
    """
    word_dict = imdb.get_word_index(path='imdb_word_index.json')
    D = {v+3:k for k,v in word_dict.items()}
    return ' '.join(list(map(lambda s:D.get(s,'[None]'),word_vec)))
        
if __name__=='__main__':
    """
    caller example:
    本例用影評資料集, 預測一個影評是「正評」還是負評
    """
    
    # num_words表示取最常出現頻率前幾個字
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    
    #print(vector_to_sentence(x_train[0]))
    #print(vector_to_sentence(x_train[1]))

    
    """
    資料前處理: pad_sequences做「截長補短」，取影評的「後面」一百個字，過短的字補0
    (測試發現取影評的「後面」一百個字好像比取「前面」一百個字效果好)
    """
    x_train = sequence.pad_sequences(x_train, maxlen=100)
    x_test = sequence.pad_sequences(x_test, maxlen=100)
    y_train = np_utils.to_categorical(y_train,2) #one-hot encoding
    y_test = np_utils.to_categorical(y_test,2) #one-hot encoding
    
    model = RNN_model(10000,2)
    model.fit(x_train, y_train)
    
    predict = model.predict_classes(x_test)
    score = model.evaluate(x_test, y_test)
    print('測試資料的loss', score[0])
    print('測試資料的正確率', score[1])
    
    if score[1] > 0.9:
        model.save('./RNN_movie_emotion_judge_model.h5')

