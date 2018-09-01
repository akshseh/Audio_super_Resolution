import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Input, LSTM, RepeatVector
# from keras.datasets import mnist
from matplotlib import pyplot as plt
import h5py
import os
import pickle
from scipy.io.wavfile import write as wavwrite
from keras.models import model_from_json


script_dir = os.path.dirname(__file__)
rel_path = "final_train_data.h5"
abs_file_path = os.path.join(script_dir, rel_path)


f = h5py.File(abs_file_path,'r')

x_train = f[f.keys()[0]]
y_train = f[f.keys()[1]]

rel_path = "final_test_data.h5"
abs_file_path = os.path.join(script_dir, rel_path)

f = h5py.File(abs_file_path,'r')

X_test = f[f.keys()[0]]
y_test = f[f.keys()[1]]

X_train = x_train

# print "LENX: ",len(X_train)
# print "LENY: ",len(y_train[0])


Y_train = []
for i in y_train:
	Y_train.append([i])

Y_test = []
for i in y_test:
	Y_test.append([i])

# # print X_train[0]
# # wavwrite('test2xseclr.wav',16000,X_test[:10000])

# print np.array([X_test[0]]).shape
# dimlayer2 = ()

# model = Sequential()
# model.add(Dense(4,input_shape=(24576,1)))
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(Activation('sigmoid'))
# model.add(Dense(1))

# # '''
# # inputs = Input(shape=(24576, 1))
# # encoded = LSTM(4)(inputs)

# # decoded = RepeatVector(24576)(encoded)
# # decoded = LSTM(1, return_sequences=True)(decoded)

# # model = Model(inputs, decoded)
# # #model = Model(inputs, encoded)
# # '''

# model.compile(loss='mean_squared_error', optimizer='rmsprop')

# model.fit(X_train, y_train, nb_epoch=4, batch_size=16, shuffle="batch")
# # modelfilename = open("myfirstmodel","w")
# # pickle.dump(model,modelfilename)
# # modelfilename.close()

# # serialize model to JSON
# model_json = model.to_json()
# with open("autoenc_model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("autoenc_model.h5")
# print("Saved model to disk")
 
# # later...
 
# load json and create model
json_file = open('autoenc_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("autoenc_model.h5")
print("Loaded model from disk")

model.compile(loss='mean_squared_error', optimizer='rmsprop')

score = model.evaluate(X_test, y_test, batch_size=16)
mypredz = model.predict(np.array(X_test))
# print mypredz[0]
# print Y_test[0]
# print len(mypredz[0])
# print len(Y_train[0][0])
# print len(mypredz[0])==len(Y_train[0])
mypredz_0 = np.array(mypredz[0])*0.01
mypredz_1 = np.array(mypredz[0])*1
mypredz_2 = np.array(mypredz[0])*100

# wavwrite('point1_test2xsec_18nov.wav',16000,mypredz_0)
# wavwrite('1_test2xsec_18nov.wav',16000,mypredz_1)
wavwrite('10_test2xsec_18nov.wav',16000,mypredz_2)
# wavwrite('test2ysec_18nov.wav',16000,Y_test[0][0])

# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# print X_train.shape
# print len(X_train)
# # print X_train[0]
# # print X_train[5665][10]
# # plt.imshow(X_train[0])
# # print "hi"
# X_train = X_train.reshape(X_train.shape[0],1,28,28)
# X_test = X_test.reshape(X_test.shape[0],1,28,28)


# # print X_train[0]
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# print Y_train.shape

# print Y_train[0]
# Y_train = np_utils.to_categorical(Y_train, 10)
# Y_test = np_utils.to_categorical(Y_test, 10)

# print Y_train[0]

