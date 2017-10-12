import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
import h5py
import os
import pickle
from scipy.io.wavfile import write as wavwrite


script_dir = os.path.dirname(__file__)
rel_path = "vctk-speaker1-train.4.16000.24576.4096.h5"
abs_file_path = os.path.join(script_dir, rel_path)

# trainingfilename = "speaker1/vctk-speaker1-train.4.16000.8192.4096.h5"
# testingfilename = "speaker1/vctk-speaker1-test.4.16000.8192.4096.h5"

f = h5py.File(abs_file_path,'r')

x_train = f[f.keys()[0]]
y_train = f[f.keys()[1]]

rel_path = "vctk-speaker1-val.4.16000.24576.4096.h5"
abs_file_path = os.path.join(script_dir, rel_path)

f = h5py.File(abs_file_path,'r')

X_test = f[f.keys()[0]]
y_test = f[f.keys()[1]]

X_train = x_train

# for i in x_train:
# 	for j in i:
# 		X_train.append([])
# 		for k in j:
# 			print "l",
# 			X_train[-1].append(k)

Y_train = []
for i in y_train:
	Y_train.append([i])

Y_test = []
for i in y_test:
	Y_test.append([i])

print X_train[0]
wavwrite('test2xseclr.wav',16000,X_train[0])

print np.array([X_test[0]]).shape
dimlayer2 = ()

model = Sequential()
model.add(Dense(4,input_shape=(24576,1)))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=4, batch_size=16, shuffle="batch")
modelfilename = "myfirstmodel"
# pickle.dump(model,modelfilename)
# score = model.evaluate(X_test, y_test, batch_size=16)
mypredz = model.predict(np.array([X_train[0]]))
print mypredz[0]
print Y_test[0]
print len(mypredz[0])
print len(Y_train[0][0])
print len(mypredz[0])==len(Y_train[0])

wavwrite('test2xsec.wav',16000,mypredz[0])
wavwrite('test2ysec.wav',16000,Y_train[0][0])

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

