import os
import sys
import time
import numpy as np
from os.path import basename, splitext, join, dirname


import librosa
from librosa.feature import melspectrogram
from librosa import load
from multiprocessing import Pool
from functools import partial

np.random.seed(8763)

opt_fea_X = 1
opt_fea_y = 1

opt_nn = 1
nClass=5
n_cores=8
catagoryNum=2;


project_dir = "/home/mtl/Documents/audio_separation"




feats_dir = join(project_dir, "feats")
result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
file_data_list = join(corpus_dir, "input/data_list")


# read wav list from data_list.txt
fid = open(file_data_list, "r")
wav_lists = []

#initiate y with number of catagory
#y[1]= catagory 1...
y_lists=[]

for i in range(catagoryNum):
    y_lists.append([])

dataNum=0
while True:
    dataline = fid.readline().strip('\n')
    if not dataline:
        break

    component = dataline.split(' ')
    dataline = component[0];
    print("file name: {}".format(dataline))
    wav_lists.append(dataline)

    for i in range(catagoryNum):
        y_lists[i].append(component[i+1])

    dataNum+=1

print(len(y_lists))

if opt_fea_X:
    for i in range(len(wav_lists)):
        X_dir = join(corpus_dir, "input")
        wav_file = join(X_dir, wav_lists[i])
        print("file name: {}".format(wav_file))
        y, sr = load(wav_file)
        #melspec = melspectrogram(y=y, sr=sr)
        
        melspec = librosa.core.stft(y)
        
        #remove the .wav
        sp=wav_lists[i].split('.')
        wavName=sp[0]
        

        X_feats_dir = join(feats_dir, "input")
        file_melspec = join(X_feats_dir, wavName + ".npy")
        print("mel spec:{} is done".format(file_melspec))
        np.save(file_melspec, melspec)


if opt_fea_y:
    # i for file length , j for catagory number
    for i in range(len(wav_lists)):
        for j in range(len(y_lists)):
            y_dir=join(corpus_dir, "cat"+str(j+1));
            wav_file = join(y_dir, y_lists[j][i] )
            print("file name: {}".format(wav_file))
            y, sr = load(wav_file)  
            # melspec = melspectrogram(y=y, sr=sr)
            melspec = librosa.core.stft(y)
            #remove the .wav
            sp=y_lists[j][i].split('.')
            yName=sp[0]

            y_feats_dir = join(feats_dir, "cat"+str(j+1))
            file_melspec = join(y_feats_dir, yName + ".npy")
            print("mel spec:{} is done".format(file_melspec))
            np.save(file_melspec, melspec)

