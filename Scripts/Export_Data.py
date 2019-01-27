##Library préparation des données
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
from os import listdir
from os.path import isfile, join
import sys
import time

from six.moves import urllib
from six.moves import xrange  
from scipy import miscimageio
import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


##Library d'apprentissage
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, AveragePooling2D, Conv2D
from keras.utils import np_utils
import tensorflow as tf
import math
from math import sqrt


import pylab
from sklearn.model_selection import train_test_split
 
np.random.seed(123)

################################################################
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = r'C:\Users\Soubeiga Armel\Desktop\Mes COURS\SSD_UGA\M2\Python\Projet\Data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

def maybe_download(filename):
  """function de téléchargement des données"""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """function d'extraction des données en format zip.
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
  """Function d'extraction des labelles."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


#exécution function download
train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz') 
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')


# Extract it into np arrays.
train_data = extract_data(train_data_filename,10) # 60000
train_labels = extract_labels(train_labels_filename, 10)
test_data = extract_data(test_data_filename, 5) #
test_labels = extract_labels(test_labels_filename, 5)


#Creation de directory
os.chdir(WORK_DIRECTORY)
if not os.path.isdir("mnist/train-images"):
   os.makedirs("mnist/train-images")


if not os.path.isdir("mnist/test-images"):
   os.makedirs("mnist/test-images")

# process train data
with open("mnist/train-labels.csv", 'w') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(train_data)):
    plt.imsave("mnist/train-images/" + str(i) + ".jpg", train_data[i][:,:,0])
    writer.writerow([str(i) + ".jpg", train_labels[i]])



# repeat for test data
with open("mnist/test-labels.csv", 'w') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(test_data)):
    plt.imsave("mnist/test-images/" + str(i) + ".jpg", test_data[i][:,:,0])
    writer.writerow([str(i) + ".jpg", test_labels[i]])


#function conver imgtopx and labellise

def get_digits_images():
    path_train=WORK_DIRECTORY+"/mnist/train-images/"
    path_test=WORK_DIRECTORY+"/mnist/test-images/"

    train= pd.read_csv(WORK_DIRECTORY+"/mnist/train-labels.csv",sep=",")
    test= pd.read_csv(WORK_DIRECTORY+"/mnist/test-labels.csv",sep=",")
    X_train=[]
    X_test=[]
    
    for row in train.itertuples():
        im=misc.imread(path_train+row[1],flatten =True) # niveaux de gris
        X_train.append(im)
        
    for row in test.itertuples():
        im=misc.imread(path_test+row[1],flatten =True)# niveaux de gris
        X_test.append(im)
        
    return np.asarray(X_train), train.iloc[:,1:2],np.asarray(X_test) #renvoie xtrain,ytrain, xtest


    
