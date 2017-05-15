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

from util import dataToRNNBatch, merge_realAndImg, dataToBatch, draw_spectrogram


#keras import
import keras

from keras.models import model_from_json
from keras.layers.wrappers import TimeDistributed
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
from time import ctime



project_dir = "/home/mtl/Documents/audio_separation"




feats_dir = join(project_dir, "feats")
result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models")
file_data_list = join(corpus_dir, "input/data_list_test")

#change variable

models_dir = join(models_dir, 'model_RNN_epoch:50_batchSize:4096_appendNum:1' )

model_structure = join(models_dir, 'model_structure') 

model_weight_path = join(models_dir, 'RNN_epoch:39_MSE_2.21514952814')

batchAppendNum = 1

catagoryNum = 2

nBatchSize = 4096


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


predict_order = 5
predict_part = 0

with open(model_structure) as json_file:  
    model_architecture = json.load(json_file)

separator = model_from_json(model_architecture)

separator.load_weights(model_weight_path, by_name=False)

x_test_batch, y_test_batch, freqNum = dataToRNNBatch(feats_dir ,wav_lists[predict_order], y_lists[predict_order],catagoryNum ,batchAppendNum, True)
y_predict = separator.predict(x_test_batch, batch_size = nBatchSize)
print(y_predict.shape)
for predict_part in range(catagoryNum):
    yy_predict = merge_realAndImg(np.transpose(y_predict[:, predict_part * freqNum : (predict_part + 1) * freqNum]))

    z1 = librosa.core.istft(yy_predict)

    y, sr = librosa.load('../corpus/cat1/1_36.wav')


    librosa.output.write_wav(models_dir+'/'+ str(predict_part + 1) + '_'  + wav_lists[predict_order], z1, sr)

    #draw_spectrogram(z1)

    #original signal record
    yy_predict = merge_realAndImg(np.transpose(y_test_batch[:, predict_part * freqNum : (predict_part + 1) * freqNum]))

    z2 = librosa.core.istft(yy_predict)
   
    librosa.output.write_wav(models_dir+'/origin_'+ str(predict_part + 1) + '_'  + wav_lists[predict_order], z2, sr)

    #draw_spectrogram(z2)

    z3 = z2 - z1

    librosa.output.write_wav(models_dir+'/minus_'+ str(predict_part + 1) + '_'  + wav_lists[predict_order], z3, sr)
    
    #draw_spectrogram(z3)
    
# for predict - input
'''
yy_predict = merge_realAndImg(np.transpose(y_predict[:, predict_part * freqNum : (predict_part + 1) * freqNum]))

z1 = librosa.core.istft(yy_predict)

y, sr = librosa.load('../corpus/cat1/1_36.wav')

librosa.output.write_wav(models_dir+'/'+ str(predict_part + 1) + '_'  + wav_lists[predict_order], z1, sr)

songName = 'Twisted_Logic#talking3.wav'

song, freq = load(join(models_dir, songName))

song = np.array(song)

song = librosa.core.stft(song)
song = librosa.core.istft(song)

z2 = song - z1

librosa.output.write_wav(models_dir+'/origin-predict_'+ str(predict_part + 1) + '_'  + wav_lists[predict_order], z2, sr)

draw_spectrogram(z2)
'''
