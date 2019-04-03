import numpy as np
# 引入很多包
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

def load_data():
   #网络下载失败
    (x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz') # 网络下载数据
   #读取本地数据
   # path = './mnist.npz'
   # f = np.load(path)
   #x_train, y_train = f['x_train'], f['y_train']
   #x_test, y_test = f['x_test'], f['y_test']
   #f.close()
    number = 10000
    x_train = x_train[0: number]
    y_train = y_train[0: number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_train.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    # x_test = np.random.normal(x_test)
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)
	
#下载	
(x_train, y_train), (x_test, y_test) = load_data()

# 引入模型 训练并且测试 输出测试结果
model = Sequential()
model.add(Dense(input_dim = 28 * 28, units = 633, activation = 'sigmoid'))
model.add(Dense(units = 633, activation = 'sigmoid'))
model.add(Dense(units = 633,activation = 'sigmoid'))
model.add(Dense(units = 10, activation = 'softmax'))

model.compile(loss = 'mse', optimizer = SGD(lr = 0.001), metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 100, epochs = 20)

score = model.evaluate(x_train, y_train)
print('\n Train Acc: ', score[1])

score = model.evaluate(x_test, y_test)
print('\n Test Acc: ', score[1])