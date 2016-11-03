import numpy as np
import matplotlib.pyplot as plt

import gzip

def load_images(filename):
    """
        Load the Images
    """
    with gzip.open(filename,'rb') as f :
        data = np.frombuffer(f.read(),np.uint8,offset = 16)
    # image shape - 28x28
    data = data.reshape(-1,28,28)
    # normalize
    return data/np.float32(256)

def load_labels(filename):
    """
        Load the labels
    """
    with gzip.open(filename,'rb') as f:
        labels = np.frombuffer(f.read(),np.uint8,offset = 8)

    return labels

def load_mnist(directory_name = ""):
    """
        Load the dataset
    """
    X_train = load_images(directory_name + "train-images-idx3-ubyte.gz")
    y_train = load_labels(directory_name + "train-labels-idx1-ubyte.gz")
    X_test = load_images(directory_name + "t10k-images-idx3-ubyte.gz")
    y_test = load_labels(directory_name + "t10k-labels-idx1-ubyte.gz")

    return (X_train,y_train,X_test,y_test)


def get_data():
    """
        Get training and testing data
    """
    X_train,y_train,X_test,y_test = load_mnist("Dataset/")
    
    return (X_train,y_train,X_test,y_test)


def activation(x):
    """
        Activation function
        1 if x>=0
        0 otherwise
    """
    return np.piecewise(x,[x<0,x>=0],[0,1])


def error_plot(error_history):
    """
        Plot Epoch vs Error
    """
    plt.plot(xrange(len(error_history)),error_history)
    plt.xlabel('Epoch')
    plt.ylabel('Misclassification Rate')
    plt.show()

class MCP:
    """
        Multicategory Perceptron
    """
    def __init__(self,X_train,y_train,X_test,y_test,n):
        """
            Initialize weights and inputs
        """
        self.X_train = X_train[:n,:] # X_train shape NxD
        self.y_train = y_train[:n].squeeze() # y_train shape Nx1
        self.b = np.ones(n).reshape(n,1)
        self.X_test = X_test
        self.y_test = y_test.squeeze()

        # One hot encoding of labels
        self.y_one_hot = np.zeros((n,10)) # y_one_hot shape Nx10
        self.y_one_hot[np.arange(n),self.y_train] = 1

        self.W = np.random.rand(28*28,10) # W shape Dx10
     
    def _miss_calculator(self,cond):
        """
            Misclassification Calculator
        """
        if cond == "train":
            data = self.X_train
            label = self.y_train
        else:
            data = self.X_test
            label = self.y_test

        out = np.dot(data,self.W)
        # Get position of max value
        max_pos = np.argmax(out,axis = 1)
        miss = np.sum(max_pos != label)
        return miss

    def train(self,eta = 1,epsilon = 1,max_epoch = 150):
        """
            Training Algorithm
        """
        n = self.X_train.shape[0]
        error_history = []
        error = 100
        epoch = 0

        while (epsilon < error) and (epoch < max_epoch):
            error = self._miss_calculator("train") * 100/n
            error_history.append(error)
            epoch += 1
            print error

            out = activation(np.dot(self.X_train,self.W) + self.b) # out shape Nx10

            for i in xrange(n):
                diff = self.y_one_hot[i,:] - out[i,:] 
                diff = diff.reshape(1,10) # diff shape 1x10
                # Update Weights
                self.W  = self.W + eta * np.dot(self.X_train[i,:].reshape(1,784).T,diff) 

        return error_history

    def predict(self):
        """
            Predict test
        """
        n = self.X_test.shape[0]
        miss = self._miss_calculator("test") * 100/n
        return miss


if __name__ == "__main__":
    X_train,y_train,X_test,y_test = get_data()

    n = 60000
    classifier = MCP(X_train.reshape(X_train.shape[0],-1),y_train,
            X_test.reshape(X_test.shape[0],-1),y_test,n)
    # Train
    error_history = classifier.train(eta = 1,epsilon = 20)
    # Plot
    error_plot(error_history)
    # Predict
    error_perc = classifier.predict()
    print error_perc
