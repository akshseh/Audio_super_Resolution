#import seaborn
import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import RMSprop
import tensorflow as tf
import h5py

# used the rnn-lstm implementation at https://gist.github.com/naotokui/12df40fa0ea315de53391ddc3e9dc0b9

f = h5py.File('./vctk-speaker1-train.4.16000.8192.4096.h5', 'r')

lowres = f[f.keys()[0]]
highres = f[f.keys()[1]]

sr = 8000
#lowres, _ = librosa.load(audio_filename, sr=sr, mono=True)
#print lowres.shape

min_lowres = np.min(lowres)
max_lowres = np.max(lowres)

# normalize
lowres = (lowres - min_lowres) / (max_lowres - min_lowres)
print lowres.dtype, min_lowres, max_lowres

maxlen     = 128 
nb_output = 256 
latent_dim = 128 

inputs = Input(shape=(maxlen, nb_output))
x = LSTM(latent_dim, return_sequences=True)(inputs)
x = Dropout(0.4)(x)
x = LSTM(latent_dim)(x)
x = Dropout(0.4)(x)
output = Dense(nb_output, activation='softmax')(x)
model = Model(inputs, output)

#optimizer = Adam(lr=0.005)
optimizer = RMSprop(lr=0.01) 
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

from tqdm import tqdm

step = 5
next_sample = []
samples = []
for j in tqdm(range(0, lowres.shape[0] - maxlen, step)):
    seq = lowres[j: j + maxlen + 1]  
    seq_matrix = np.zeros((maxlen, nb_output), dtype=bool) 
    for i,s in enumerate(seq):
        sample_ = int(s * (nb_output - 1)) # 0-255
        if i < maxlen:
            seq_matrix[i, sample_] = True
        else:
            seq_vec = np.zeros(nb_output, dtype=bool)
            seq_vec[sample_] = True
            next_sample.append(seq_vec)
    samples.append(seq_matrix)
samples = np.array(samples, dtype=bool)
next_sample = np.array(next_sample, dtype=bool)

def sample(preds, temperature=1.0, min_value=0, max_value=1):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    v = np.argmax(probas)/float(probas.shape[1])
    return v * (max_value - min_value) + min_value
     
for start in range(5000,220000,10000):
    seq = lowres[start: maxlen]  
    seq_matrix = np.zeros((maxlen, nb_output), dtype=bool) 
    for i,s in enumerate(seq):
        sample_ = int(s * (nb_output - 1))
        seq_matrix[i, sample_] = True

    for i in tqdm(range(5000)):
        z = model.predict(seq_matrix.reshape((1,maxlen,nb_output)))
        s = sample(z[0], 1.0)
        seq = np.append(seq, s)

        sample_ = int(s * (nb_output - 1))    
        seq_vec = np.zeros(nb_output, dtype=bool)
        seq_vec[sample_] = True

        seq_matrix = np.vstack((seq_matrix, seq_vec))  # added generated note info 
        seq_matrix = seq_matrix[1:]
        
    # scale back 
    seq = seq * (max_lowres - min_lowres) + min_lowres

#plt.figure(figsize=(30,5))
#plt.plot(seq.transpose())
#plt.show()
