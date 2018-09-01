import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
# from keras.layers import Input, LSTM, RepeatVector
# from keras.datasets import mnist
from matplotlib import pyplot as plt
import h5py
import os
import pickle
from scipy.io.wavfile import write as wavwrite
from hmmlearn import hmm
import math
import random
import cmath
pi2 = cmath.pi * 2.0

def DFT(fnList):
    N = len(fnList)
    FmList = []
    for m in range(N):
        Fm = 0.0
        for n in range(N):
            Fm += fnList[n] * cmath.exp(- 1j * pi2 * m * n / N)
        FmList.append(Fm / N)
    return FmList

def InverseDFT(FmList):
    N = len(FmList)
    fnList = []
    for n in range(N):
        fn = 0.0
        for m in range(N):
            fn += FmList[m] * cmath.exp(1j * pi2 * m * n / N)
        fnList.append(fn)
    return fnList


script_dir = os.path.dirname(__file__)
rel_path = "final_train_data.h5"
abs_file_path = os.path.join(script_dir, rel_path)

# trainingfilename = "speaker1/vctk-speaker1-train.4.16000.8192.4096.h5"
# testingfilename = "speaker1/vctk-speaker1-test.4.16000.8192.4096.h5"

f = h5py.File(abs_file_path,'r')

x_train = f[f.keys()[0]]
y_train = f[f.keys()[1]]

# f.close()
# 102400 samples (2320 ms) is gathered from audio port with 44100 sampling rate. Sample values are between 0.0 and 1.0

int samplingRate = 44100;
int numberOfSamples = 102400;
# float samples[numberOfSamples] = ListenMic_Function(numberOfSamples,samplingRate);
# Window size or FFT Size is 1024 samples (23.2 ms)
int N = 1024;
#Number of windows is 100
int noOfWindows = numberOfSamples / N;

N = 360 # degrees (Number of samples)
a = float(random.randint(1, 100))
f = float(random.randint(1, 100))
p = float(random.randint(0, 360))
fnList = []
for n in range(N):
    t = float(n) / N * pi2
    fn = a * math.sin(f * t + p / 360 * pi2)
    fnList.append(fn)
FmList = DFT(fnList)


rel_path = "final_test_data.h5"
abs_file_path = os.path.join(script_dir, rel_path)

f = h5py.File(abs_file_path,'r')

X_test = f[f.keys()[0]]
y_test = f[f.keys()[1]]

# f.close()

X_train = x_train



Y_train = []
for i in y_train:
	Y_train.append([i])

Y_test = []
for i in y_test:
	Y_test.append([i])




# print X_train[0]
wavwrite('test2xseclr.wav',16000,X_train[0])

# print np.array([X_test[0]]).shape
# dimlayer2 = ()

# print np.shape(x_train)
# print np.shape(x_train[0])

hmmobj = hmm.GaussianHMM(3,"full")
# hmmobj.fit(np.array(x_train[1:4]).reshape(len(x_train[1:4]),len(x_train[0])))
hmmobj.fit(y_train[0])
fmodel = open("hmm_model","w")
pickle.dump(hmmobj,fmodel)
fmodel.close()


fmodel = open("hmm_model","r")
hmmobj = pickle.load(fmodel)
fmodel.close()
outp = hmmobj.predict(x_train[0])
# outp = hmmobj.predict(A[0][:len(A)/2])
# x_new = []
# for i in range(0,len(outp)):
# 	x_new.append(x_train[len(x_train)/2+i])
# 	x_new.append(x_train[len(x_train)/2+i]*percentages[outp[i]])

# outp = np.zeros(len(outp))

for i in range(1,len(outp)):
	if x_train[0][i]==0:
		x_train[0][i]=x_train[0][i-1]*outp[i]


# print np.shape(outp)
lol =x_train[0].reshape(len(x_train[0]))
new_audio = np.transpose([outp,lol]).reshape(2*len(outp))
# print "naaa: ",np.shape(new_audio)
# print outp
# print lol
# print B
# print A
# print outp
wavwrite('hmmoutpt_16k.wav',16000,x_train[0])
wavwrite('hmmoutpt_32k.wav',32000,new_audio)
wavwrite('hmmoutpt_8k.wav',8000,new_audio)

average = (np.array(x_train[0])*100+np.array(y_train[0]))/101
# a = numpy.random

# wavwrite('hmmoutpt_16k.wav',16000,outp)
wavwrite('present.wav',16000,average)
