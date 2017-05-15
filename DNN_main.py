import os
import sys
import time
import numpy as np
from os.path import basename, splitext, join, dirname


import json
import librosa
from librosa.feature import melspectrogram
from librosa import load
from multiprocessing import Pool
from functools import partial
import math


#keras import
import keras
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
from time import ctime

np.random.seed(8763)

opt_nn = 1
n_cores=8
catagoryNum = 2;
batchAppendNum = 6;
nBatchSize = 1024


hiddenUnitNum = 500

nEpoch=50

project_dir = "/home/mtl/Documents/audio_separation"




feats_dir = join(project_dir, "feats")
result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models")
file_data_list = join(corpus_dir, "input/data_list")


# read wav list from data_list.txt
fid = open(file_data_list, "r")
wav_lists = []

#initiate y with number of catagory
#y[1]= catagory 1...
y_lists=[]


dataNum=0
while True:
    dataline = fid.readline().strip('\n')
    if not dataline:
        break

    component = dataline.split(' ')
    dataline = component[0];
    print("file name: {}".format(dataline))
    wav_lists.append(dataline)

    y_lists.append([])
    for i in range(catagoryNum):
        y_lists[dataNum].append(component[i+1])

    dataNum+=1

splitTrainTest = math.floor(dataNum*2/3)

x_train = wav_lists[0 : splitTrainTest]
x_test = wav_lists[splitTrainTest : dataNum]
y_train = y_lists[0 : splitTrainTest]
y_test = y_lists[splitTrainTest : dataNum]

def extract_realAndImg(ff):

    freqBand, length = ff.shape
    ff_out = np.zeros((2 * freqBand, length))
    ff_out[0 : freqBand, : ] = ff[ : , : ].real
    ff_out[freqBand : 2*freqBand, :] = ff[ : , :].imag
    return ff_out


# load one pair of song at a time
# isComplex: Whether need to peparate to real amd imaginary part

def dataToBatch(x_list, y_list, appendNum, isComplex):
    
     
    #load input and output music
    s = x_list.split('.')
    x = np.load(join(feats_dir, "input/" + s[0] + ".npy"))
    y = []
    for i in range(catagoryNum):
        y.append([])
        s = y_list[i].split('.')
        y[i] = np.load(join(feats_dir,"cat"+str(i+1)+'/'+ s[0] + ".npy"))
    #Make size of cat2(addNoise) equal to main audio
    freqNum, inputLength = x.shape    
    
    if isComplex:
        freqNum = 2 * freqNum

    for i in range(catagoryNum-1):
        a, yLength = y[i+1].shape

        newY=np.zeros((a,inputLength))
        yContent = y[i+1]
        for j in range(inputLength):
            newY[:,j] = yContent[:, j % yLength]

        y[i+1]=newY    

    #start to pack

    y_batch = np.zeros((catagoryNum * freqNum, inputLength))
    for i in range(catagoryNum):
        if isComplex:

            y_batch[i * freqNum : (i+1) * freqNum,:] = extract_realAndImg(y[i])
        else: 
            y_batch[i * freqNum : (i+1) * freqNum,:] = y[i]
    

    x_batch = np.zeros(((2 * appendNum + 1) * freqNum , inputLength))
    for col in range(inputLength):
        for row in range(2 * appendNum + 1):
            index = row - appendNum
            appendElement = np.zeros((freqNum,1))
            # append (0,0,...,0) in the beginning and the end

            if ((index + col) < 0) or ((index + col) > (inputLength-1)):
                
                x_batch[row * freqNum : (row + 1) * freqNum, col] = appendElement[:,0]  
            else : 
                
                if isComplex:
                    
                    appendElement = extract_realAndImg(x[:, col + index : col + index +1])
                else:
                   
                    appendElement[:,0] = x[:, col + index]
               
                x_batch[row * freqNum : (row + 1) * freqNum, col] = appendElement[:,0]
    
    x_batch = np.transpose(x_batch)
    y_batch = np.transpose(y_batch)
    return x_batch, y_batch, freqNum

#Model
x,y, freqBand = dataToBatch(wav_lists[0],y_lists[0],batchAppendNum, True)

A, x_freqBand = x.shape
B, y_freqBand = y.shape



input_audio = Input(shape = (x_freqBand, ))
layer1 = Dense(2*hiddenUnitNum, activation=None, use_bias=True ) (input_audio)
layer2 = Dense(2*hiddenUnitNum, activation=None, use_bias=True )(layer1)
layer3 = Dense(hiddenUnitNum, activation=None, use_bias=True )(layer2)
audio_output = Dense(y_freqBand, activation=None, use_bias=True)(layer3)

separator = Model(input_audio, audio_output)

separator.compile(optimizer='adadelta', loss='mean_squared_error')

#create model directory
directory = models_dir+'/unit:' + str(hiddenUnitNum) + '_'  + 'model_DNN_epoch:' +str(nEpoch)+'_batchSize:'+str(nBatchSize)+'_appendNum:'+str(batchAppendNum)
os.makedirs(directory)

models_dir = directory

model_architecture = separator.to_json()

with open(models_dir+'/model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)


#training
for trainingEpoch in range(nEpoch):
    
    print("start epoch :", trainingEpoch+1)   
    
    for dataOrder in range(40):
        print("data : ",dataOrder)
        x_batch, y_batch, freqNum = dataToBatch(x_train[dataOrder], y_train[dataOrder], batchAppendNum, True)
        separator.fit(x_batch, y_batch,
                epochs=1,
                batch_size = nBatchSize,
                shuffle = False)
        
    x_test_batch, y_test_batch, freqNum = dataToBatch(x_test[0], y_test[0], batchAppendNum, True)
    y_predict = separator.predict(x_test_batch)
    MSE = ((y_predict - y_test_batch)**2).mean(axis=None)
    print("MSE:",MSE)
    print("A:", y_predict[700,100:110])
    print("B:", y_test_batch[700,100:110])
    if (trainingEpoch + 1) % 5 == 0:
        separator.save_weights(models_dir + '/DNN_epoch:' + str(trainingEpoch + 1) + '_MSE_'+str(MSE))


#predict

separator.save_weights(models_dir + '/' + str(nBatchSize) + '_' + str(batchAppendNum) + '_' + str(nEpoch) + '_'+str(MSE))


x_test_batch, y_test_batch, freqNum = dataToBatch(x_test[0], y_test[0], batchAppendNum, True)
y_predict = separator.predict(x_test_batch)

def merge_realAndImg(ffSep):
    freqBand, length = ffSep.shape
    origin_freq = int(freqBand/2)
    print(origin_freq)
    # ff_out = np.zeros((origin_freq, length))
    ff_out = ffSep[0 : origin_freq, :] + 1j * ffSep[origin_freq : freqBand, :]
    return ff_out

print("shape:",y_predict.shape)

y_predict = merge_realAndImg(np.transpose(y_predict[:, 0 : freqNum]))

print(y_predict.shape)

z1 = librosa.core.istft(y_predict)

y, sr = librosa.load('../corpus/cat1/1_36.wav')


print(y.shape)
librosa.output.write_wav('test.wav', z1, sr)








    
