import numpy as np
import sys,os
import gzip


def load_images(filename):
    """
        Load the Images
    """
    with gzip.open(filename,'rb') as f :
        data = np.frombuffer(f.read(),np.uint8,offset = 16)
    data = data.reshape(-1,28,28)

    return data/np.float32(256)

def load_labels(filename):
    """
        Load the labels
    """
    with gzip.open(filename,'rb') as f:
        labels = np.frombuffer(f.read(),np.uint8,offset = 8)

    return labels

def load_mnist(directory_name):
    """
        Load the dataset
    """
    X_train = load_images(directory_name + "train-images-idx3-ubyte.gz")
    y_train = load_labels(directory_name + "train-labels-idx1-ubyte.gz")
    X_test = load_images(directory_name + "t10k-images-idx3-ubyte.gz")
    y_test = load_labels(directory_name + "t10k-labels-idx1-ubyte.gz")

    # One hot encoding
    temp = np.zeros((y_train.shape[0],10))
    temp[np.arange(y_train.shape[0]),y_train] = 1
    y_train = temp
    temp = np.zeros((y_test.shape[0],10))
    temp[np.arange(y_test.shape[0]),y_test] = 1
    y_test = temp

    # reshape
    X_train = X_train.reshape(X_train.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)

    return (X_train,y_train,X_test,y_test)
