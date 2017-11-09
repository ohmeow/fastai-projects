####################################### 
# imports
#######################################

# core imports
import os, sys, math, json, bcolz, random as py_random
import shutil, zipfile
from glob import glob

# statistical/analytics imports
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import PIL

# ml imports
from sklearn.preprocessing import OneHotEncoder

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.regularizers import *
from keras.utils import to_categorical

####################################### 
# tensorflow specific utility methods
#######################################

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

####################################### 
# batch and data utility methods
#######################################

def get_batches(dirname, gen=image.ImageDataGenerator(), target_size=(224,224), class_mode='categorical', batch_size=4, shuffle=True):
    return gen.flow_from_directory(dirname, target_size, class_mode=class_mode, batch_size=batch_size, shuffle=shuffle)

def get_data(dirname, target_size=(224,224)):
    batches = get_batches(dirname, target_size=target_size, class_mode=None, batch_size=1, shuffle=False)
    return np.concatenate([ batches.next() for i in range(batches.n) ])

def get_batch_info(dirname):
    batches = get_batches(dirname, batch_size=1, shuffle=False)
    return (batches.classes, onehot(batches.classes), batches.filenames)

# reshapes an 1D array of classes to a one-hot encoded array
def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

####################################### 
# finetuning
#######################################

def finetune(model, n_outputs, n_layers_to_remove=1):
    '''
    Removes n_layers from top of the model's layer stack and appends a new
    Dense layer with n_outputs
    '''
    # remove n_layers and fix remaining layer weights
    del model.layers[-n_layers_to_remove:]
    for layer in model.layers: layer.trainable = False
        
    # update the model outputs = the output from the last layer
    model.outputs = [model.layers[-1].output]

    # recover the output from the last layer in the model and use as input to new Dense layer
    last = model.layers[-1].output
    x = Dense(n_outputs, activation="softmax")(last)
    ft_model = Model(model.input, x)
    
    compile(ft_model)
    return ft_model

def compile(model, lr=0.001):
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

####################################### 
# serialization and directory creation helpers
#######################################

# methods for persisting/serializaing numpy arrays
def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    
def load_array(fname):
    return bcolz.open(fname)[:]

# will force create directory
def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)

####################################### 
# plotting utility methods
#######################################

to_bw = np.array([0.299, 0.587, 0.114])

def gray(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).dot(to_bw)
    else:
        return np.rollaxis(img, 0, 3).dot(to_bw)

def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')