import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten
from keras.layers import Dense, Lambda, SeparableConvolution2D, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.activations import relu, elu
from keras import initializations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam

import cv2

def preprocess_image(im):
    #return im
    new_size=(im.shape[1]//2,im.shape[0]//2)
    im=cv2.resize(im,(new_size),interpolation=cv2.INTER_AREA)
    im=im[30:,:,:]
    im=cv2.cvtColor(im,cv2.COLOR_RGB2YUV)
    return  ((im.astype(np.float16) - 128.)/255.)

def split(*lists,n=None,ratio=None):
    '''
    returns lists split either at element or ratio, 
    two lists are returned for every list passed in
    '''
    if ratio is not None:
        n=int(len(lists[0])*ratio)
    if n is None or n > len(lists[0]):
        n = len(lists[0])
    rv = []
    for l in lists:
        rv.append(l[:n])
        rv.append(l[n:])
    return rv

def elements_at_indexes(*lists, indexes):
    '''
    returns elements from lists at given indexes
    '''
    if len(lists)==1:
        l=lists[0]
        return [l[i] for i in indexes]
    else:
        return [elements_at_indexes(l,indexes=indexes) for l in lists]


def random_sample(*lists, n=None, ratio = None, return_remainder = False, random_order=True):
    """
    Returns parallel lists with n random samples from lists without replacement
    
    parameters
    ---
    n                   - number of samples to return in primary set
    ratio               - ratio of samples to return in primary set
                          if and and ratio ar both None, all samples are returned
    return_remainder    - returns two sets of lits, one with the n or 
                          ratio samples, the other with the remaining samples
    random_order        - if True, samples are returned in random order, otherwise
                          they are returned in the same order as in the original lists
    
    """
    list_length = len(lists[0])
        
    all_indexes = list(range(list_length))
    np.random.shuffle(all_indexes)
    i1,i2 = split(all_indexes, n=n, ratio=ratio)
    
    if random_order == False:
        i1.sort()
        i2.sort()
    rv  = []
    for l in lists:
        rv.append(elements_at_indexes(l,indexes=i1))
        if return_remainder:
            rv.append(elements_at_indexes(l,indexes=i2))
    if len(rv) == 1:
        rv = rv[0]
    return rv

def read_driving_log(folder):
    '''
    returns a pandas dataframe of the driving log in folder
    '''
    csv_path = folder+'/driving_log.csv'
    csv_column_names = [
        'center',
        'left',
        'right',
        'steering',
        'throttle',
        'brake',
        'speed']
    
    return pd.read_csv(csv_path,names=csv_column_names,skiprows=1)

def model_a(input_shape):
    '''
    defines the model for this project
    
    This was loosely based on the NVIDIA and LeNet models, but I modified it myself
    
    Three convolutional layers are used to allow medium complexity feature detection
    
    Relu activations are used in several places to introduce non-linearities and 
    add an opportunity for the model to create sparsity
    
    A dropout is used to keep the model from over-fitting
    Two dense layers are used with tanh activations.  Tanh was selected because it 
    is symmetric around zero and in general, steering angles ar centered on zero.
    
    Finally, a linear activation was used in the output layer to ensure outputs weren't
    unecessarily compressed / limited as would be done by tanh, or relu.
    
    Note: The data is normalized by the image preprocessing so no additional normalization 
    is done by the model.
    '''
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(Convolution2D(16, 3,3,  border_mode='valid', subsample=(2,2),activation='relu',name='conv1', dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, border_mode='valid',  subsample=(2,2), activation='relu',name='conv2', dim_ordering='tf'))
    model.add(Convolution2D(8, 5, 5, border_mode='valid', activation='relu',name='conv3', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(4, 4), name ='maxpool2'))
    model.add(Flatten())
    model.add(Dense(20,activation='tanh',name='dense20'))
    model.add(Dropout(0.5, name ='dropout'))
    model.add(Dense(20,activation='tanh', name = 'dense10'))
    model.add(Dense(1,activation='linear', name = 'final'))
    return model


# demo data and simulator data use different format,
# this handles both
def actual_file_path_for_image(data_folder, filename):
    '''
    returns path for filename in the log file
    '''
    rv = "/".join((data_folder,"IMG",filename.split("\\")[-1]))
    rv = rv.replace("/IMG/IMG","/IMG")
    return rv


def read_logs(data_folder, include_side_images = True, side_image_correction=0.12, verbose=True):
    '''
    returns parallel numpy arrays of filenames and steering angles for all sub-folders of data_folder
    '''
    image_files = []
    steering_angles = []
    for folder in os.listdir(data_folder):
        log_path = os.path.join(data_folder,folder)
        log = read_driving_log(log_path)
        
        center_filenames = [actual_file_path_for_image(log_path,f) for f in log.center.values]
        image_files.extend(center_filenames)
        #steering_angles.extend(smooth(log.steering.values))
        steering_angles.extend(log.steering.values)
        
        if include_side_images:
            right_filenames = [actual_file_path_for_image(log_path,f) for f in log.right.values]
            image_files.extend(right_filenames)
            steering_angles.extend(log.steering.values - side_image_correction)
            
            left_filenames = [actual_file_path_for_image(log_path,f) for f in log.left.values]
            image_files.extend(left_filenames)
            steering_angles.extend(log.steering.values + side_image_correction)
            
        if verbose:
            print('folder: {} training_samples: {}'.format(folder,len(steering_angles)))
    return np.array(image_files),np.array(steering_angles)

def load_images_for_files(file_paths):
    '''
    returns a numpy array of preprocessed images for files in file_paths
    '''
    images = None
    for i,f in enumerate(file_paths):
        im=plt.imread(f)
        im=preprocess_image(im)
        if images is None: # create array when we get shape from first image
            images = np.zeros((len(file_paths),*im.shape),dtype=np.float16)
        images[i,:,:,:]=im
    return images

class BehaviorCloner:
    def __init__(self, data_folder = "../SimulatorData"):
        self.data_folder = data_folder
        self.images = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_ratio = 0.95
        self.checkpoint_filename = 'best_model.h5'
        pass
    
    def _compile(self):
        '''
        compiles model with mean square error loss optimization and a slow Adam optimizer
        
        The Adam optimizer was chosen because it has a lower depence on hyperparameters
        and tends to do well at converging
        '''
        self.model.compile(loss='mse', optimizer=Adam(lr=0.0001)) # 0.01 sucked and would stall at 0.0262
    
    def create_model(self):
        '''
        creates an unitialized model, ready to train
        '''
        self.model = model_a(self.X_val[0].shape)
        self._compile()
        
    def load_data(self, sample_limit = None):
        '''
        loads all images from data folder, samples up to sample limit if not None
        '''
        X_path,y=read_logs(self.data_folder)
        if sample_limit is not None:
            X_path,y = random_sample(X_path,y,n=sample_limit, random_order=False)
        X_train_path, X_val_path, y_train, y_val = \
          random_sample(X_path,y,random_order=False,return_remainder=True,ratio=self.train_ratio)
            
        self.y_val = np.array(y_val)
        self.y_train = np.array(y_train)
        self.X_train = load_images_for_files(X_train_path)
        self.X_val = load_images_for_files(X_val_path)
        
    def load_model(self, path = None):
        '''
        loads model with weights from path
        '''
        if path is None:
            path = self.checkpoint_filename
        self.model = keras.models.load_model(path)
        self._compile()

    
    def get_model(self):
        '''
        returns model.  Creates if necessary
        '''
        if self.model is None:
            self.create_model()
        return self.model
        
    def train(self, nb_epoch=200, batch_size=100, patience=5, keep_best = True):
        '''
        trains model
        
        params
        ------
        nb_epoch - maximum number of epochs to train
        batrch_size - batch size to use while training
        patience - number of epochs to wait for improvement in loss before quitting
        keep_best - if true, keeps the model from the epoch with the lowest loss
        '''
        model = self.get_model()
        history = model.fit(self.X_train, self.y_train, shuffle=True,
            nb_epoch=nb_epoch, batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=[ModelCheckpoint(self.checkpoint_filename), EarlyStopping(patience=patience)])
        if keep_best:
            self.load_model(self.checkpoint_filename)
            
    def save_json_and_weights(self, json_filename='model.json', weights_filename='model.h5'):
        '''
        saves model in a format required for project submission.
        '''
        model = self.get_model()
        json = model.to_json()
        with open(json_filename, 'w') as f:
            f.write(json)
        model.save_weights(weights_filename)
        
    def plot_predictions(self):
        '''
        creates plots of predicted vs training values
        '''
        model = self.get_model()
        plt.figure(figsize=(10,10))
        
        plt.subplot(2,2,1)
        y_train_predict = model.predict(self.X_train)
        plt.plot(self.y_train,color='b')
        plt.plot(y_train_predict,color='g')
        plt.title("steering angles - training set\nActual(blue) vs. Predicted(green) ")
        
        plt.subplot(2,2,2)
        y_val_predict = model.predict(self.X_val)
        plt.plot(self.y_val,color='b')
        plt.plot(y_val_predict,color='g')
        plt.title("steering angles - validation set\nActual(blue) vs. Predicted(green)")
        
        plt.subplot(2,2,3)
        r=[min(self.y_train),max(self.y_train)]
        plt.plot(r,r,"k--",linewidth=2,alpha=0.5)
        plt.scatter(self.y_train, y_train_predict,alpha=0.1)
        plt.ylabel("predicted")
        plt.xlabel("actual")
        plt.title("Prediction correlation on training set")

        plt.subplot(2,2,4)
        plt.plot(r,r,"k--",linewidth=2,alpha=0.5)
        plt.scatter(self.y_val, y_val_predict,alpha=0.1)
        plt.ylabel("predicted")
        plt.xlabel("actual")
        plt.title("Prediction correlation on validation set")


