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

from util import dataToRNNBatch, merge_realAndImg, dataToBatch


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

models_dir = join(models_dir, 'model_RNN_epoch:50_batchSize:256_appendNum:1' )

model_structure = join(models_dir, 'model_structure') 

model_weight_path = join(models_dir, 'RNN_epoch:30_4096_0_100_4.80055571663')

batchAppendNum = 0

catagoryNum = 2

nBatchSize = 256


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



with open(model_structure) as json_file:  
    model_architecture = json.load(json_file)

separator = model_from_json(model_architecture)

separator.load_weights(model_weight_path, by_name=False)

orderTest = 5

#0 for catagory 1, 1 for catagory 2 ...
predict_part = 0

x_test_batch, y_test_batch, freqNum = dataToBatch(feats_dir ,wav_lists[orderTest], y_lists[orderTest],catagoryNum ,batchAppendNum, True)
y_predict = separator.predict(x_test_batch, batch_size = nBatchSize)

y_predict = merge_realAndImg(np.transpose(y_test_batch[:, predict_part * freqNum : (predict_part + 1) * freqNum]))

print(y_predict.shape)

z1 = librosa.core.istft(y_predict)

y, sr = librosa.load('../corpus/cat1/1_36.wav')


librosa.output.write_wav(model_dirs + str(predict_part) + '_' + wav_lists[5], z1, sr)









