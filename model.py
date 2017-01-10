# manually transcribed and modified from 
# https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.u8zq6ghon

# according to docs, nvidia used elu

import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten
from keras.layers import Dense, Lambda, SeparableConvolution2D, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.activations import relu, elu

def RELU():
    return Activation(elu)

def nvidia_model(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 -1., input_shape = input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Dense(1164, init="he_normal"))
    model.add(RELU())
    model.add(Dense(100, init="he_normal"))
    model.add(RELU())
    model.add(Dense(50, init="he_normal"))
    model.add(RELU())
    model.add(Dense(10, init="he_normal"))
    model.add(RELU())
    model.add(Dense(1, init="he_normal"))
    return model


def model_a(input_shape):
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(Convolution2D(16, 3,3,  border_mode='valid', subsample=(2,2),activation='relu',name='conv1', dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, border_mode='valid',  subsample=(2,2), activation='relu',name='conv2', dim_ordering='tf'))
    model.add(Convolution2D(8, 5, 5, border_mode='valid', activation='relu',name='conv3', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(4, 4), name ='maxpool2'))
    model.add(Flatten())
    model.add(Dense(20,activation='sigmoid',name='dense20'))
    model.add(Dropout(0.5, name ='dropout'))
    model.add(Dense(20,activation='sigmoid', name = 'dense10'))
    model.add(Dense(1,activation='linear', name = 'final'))
    return model

def model_b(input_shape):
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(SeparableConvolution2D(16, 3,3,  border_mode='valid', subsample=(2,2),activation='relu',name='conv1', dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, border_mode='valid',  subsample=(2,2), activation='relu',name='conv2', dim_ordering='tf'))
    model.add(Convolution2D(64, 5, 5, border_mode='valid', activation='relu',name='conv3', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(4, 4), name ='maxpool2'))
    model.add(Flatten())
    model.add(Dense(100,activation='sigmoid',name='dense100'))
    model.add(Dense(20,activation='sigmoid',name='dense20'))
    model.add(Dropout(0.5, name ='dropout'))
    model.add(Dense(10,activation='sigmoid', name = 'dense10'))
    model.add(Dense(1,activation='linear', name = 'final'))
    return model