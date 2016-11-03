import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import *
import time

class NeuralNet:
    """
    Neural Network class
    """
    def __init__(self,X_train,y_train,X_test,y_test,hid_list,hyperparams):
        """
        Initialize
        Inputs - 
            data : Input shape NXD
            label : Desired output NXC
            X_train : Training data
            y_train : Training output
            X_test : Testing data
            y_test : Testing output
            hyperparams : Dictionary of hyper parameters 
                reg : regularization 
                learning_rate : learning rate 
                learning_rate_decay : learning rate decay
                epoch : number of epochs
                batch_size : minibatch size
        """
        # # Development
        # X_train = X_train[:2200]
        # y_train = y_train[:2200]
        # X_test = X_test[:500]
        # y_test = y_test[:500]

        # store data; training 80%, validation 20%
        loc = int(0.8 * X_train.shape[0])
        self.X_train = X_train[:loc]
        self.y_train = y_train[:loc]
        self.X_val = X_train[loc:]
        self.y_val = y_train[loc:]
        self.X_test = X_test
        self.y_test = y_test

        # get dimensions
        self.n,self.D = self.X_train.shape
        _,self.C = self.y_train.shape 
        self.hid_list = hid_list

        # add number of output
        hid_list.append(self.C)
        self.num_layers = len(hid_list)

        # Dictionary of parameters(network weights, biases, adam)
        self.params = {}
        self.grads = {}
        self.adam_config = {}
        # Network weight and bias initialization
        for i in xrange(self.num_layers):
            if i+1 == 1:
                self.params["W" + str(i + 1)] = np.random.normal(size=(self.D * hid_list[i]),scale = 1e-3).reshape(self.D,-1)
            else:
                self.params["W" + str(i + 1)] = np.random.normal(size=(hid_list[i - 1] * hid_list[i]),scale = 1e-3).reshape(hid_list[i-1],-1)
            # adam config for weights
            self.adam_config["W" + str(i+1)] = {}
            wshape = self.params["W" + str(i+1)].shape
            self.adam_config["W" + str(i+1)]["momentum"] = np.zeros(wshape)
            self.adam_config["W" + str(i+1)]["velocity"] = np.zeros(wshape)
            self.adam_config["W" + str(i+1)]["t"] =  0

        for i in xrange(self.num_layers):
            self.params["b" + str(i + 1)] = np.random.normal(size = (self.hid_list[i]),scale = 1e-3).reshape(1,-1)
            self.adam_config["b" + str(i + 1)] = {}
            bshape = self.params["b" + str(i+1)].shape
            self.adam_config["b" + str(i+1)]["momentum"] = np.zeros(bshape)
            self.adam_config["b" + str(i+1)]["velocity"] = np.zeros(bshape)
            self.adam_config["b" + str(i+1)]["t"] = 0 

        # Dropout config initialization
        self.dropout_config = {}
        self.dropout_config["p"] = 0.5
        self.dropout_config["mode"] = "train"

        # Hyperparameters
        self.epoch = hyperparams["epoch"]
        self.batch_size = hyperparams["batch_size"]
        self.reg = hyperparams["reg"]
        self.lr = hyperparams["learning_rate"]
        self.lr_decay = hyperparams["learning_rate_decay"]

        # Adam optimization config initialization
        self.adam_config["beta_1"] = 0.9
        self.adam_config["beta_2"] = 0.999
        self.adam_config["epsilon"] = 1e-8

        # others
        self.verbose = True
        self.print_atevery = 20
        self.num_batch = self.n/self.batch_size
        self.mode = "train"
   

    # Forward prop
    def _forward_step(self,x,w,b):
        """
        One step of forward propagation
        Inputs - 
            x : Forward flow input shape NXD
            w : weights shape DXH
            b : bias shape 1XH
        """
        out = np.dot(x,w) + b
        # Relu
        reluout = np.maximum(0,out)

        # dropout
        p = self.dropout_config["p"]
        mode = self.dropout_config["mode"]
        mask = (np.random.rand(*reluout.shape) < p)/p
        if mode == "train":
            reluout *= mask
        cache = x,w,out,mask,mode

        return reluout,cache

    # Final layer
    def _final_forward(self,x,w,b):
        """
        Final layer no transformation
        Inputs - 
            x : forward flow input shape NXD
            w : weights shape DXH
            b : bias shape 1XC
        """
        out = np.dot(x,w) + b
        cache = x,w

        return out,cache

    def forward_prop(self,x):
        """
        Forward propagation
        Input - 
            x : input
        """
        hid_cache = []
        hid_output = []
        for i in xrange(self.num_layers):
            if i+1 == self.num_layers:
                x,cache = self._final_forward(x,self.params["W" + str(i+1)],self.params["b" + str(i+1)])
            else:
                x,cache = self._forward_step(x,self.params["W" + str(i+1)],self.params["b" + str(i+1)])

            hid_cache.append(cache)

        return x,hid_cache
    #######################

    # Backward prop
    def _final_backward(self,delta,cache):
        """
        Final layer backprop
        Inputs - 
            delta backward flow input shape NXH
        """
        x,w = cache

        db = np.sum(delta,axis = 0) 
        dw = np.dot(x.T,delta)
        dx = np.dot(delta,w.T)

        return dx,dw,db

    def _backward_step(self,delta,cache):
        """
        One step of backward propagation
        Inputs - 
            delta : Backward flow input
        """
        x,w,rel,mask,mode = cache 

        # dropout
        if mode == "train":
            delta *= mask

        delta[rel<0] = 0
        db = np.sum(delta,axis = 0) 
        dw = np.dot(x.T,delta)
        dx = np.dot(delta,w.T)

        return dx,dw,db

    def backward_prop(self,delta,hid_cache):
        """
        Backward propagation
        Inputs - 
            delta : Derivative of scores
        """
        for i in reversed(xrange(self.num_layers)):
            if i+1 == self.num_layers:
                delta,dw,db = self._final_backward(delta,hid_cache[i])
            else:
                delta,dw,db = self._backward_step(delta,hid_cache[i])
            self.grads["W" + str(i+1)],self.grads["b" + str(i+1)] = dw,db
            self.grads["W" + str(i+1)] += self.reg * self.params["W" + str(i+1)]
    ##################################
    
    # Loss calculator
    def softmax_loss(self,out,y = None):
        """
        Calculates softmax loss
        Inputs - 
            x : Network output
            y : Desired output
        """
        # numerical stability
        out = np.exp(out - np.max(out,axis = 1,keepdims = True))
        out /= np.sum(out,axis = 1,keepdims = True)
        # for test data
        if y is None:
            return out
        # get desired labels
        z = np.argmax(y,axis = 1)
        # loss
        loss = -np.sum(np.log(out[np.arange(y.shape[0]),z]))/out.shape[0]
        delta = out.copy()
        # derivative changes only for correct label
        delta[np.arange(out.shape[0]),z] -= 1
        delta /= out.shape[0]

        return loss,delta,out

    def loss_grad(self,x,y = None):
        """
        Calculates loss and gradients
        Inputs - 
            x : Network output
            y : Desired output
        """
        # cross entropy loss (softmax)
        data_loss,delta,out = self.softmax_loss(x,y)
        # L2 regularization loss
        reg_loss = 0
        for i in xrange(self.num_layers):
            reg_loss += np.sum(self.params["W" + str(i+1)] * self.params["W" + str(i+1)])
        reg_loss *= 0.5 * self.reg 
        # total loss
        loss = data_loss + reg_loss

        return loss,delta,out
    ###########################

    # Update Parameters
    def adam_update(self,k,parameter,grad_parameter):
        """
        Update using Adam method
        Inputs - 
            parameters - Dictionary of parameters with keys - 
                W : Network weights
                b : Network biases
            grad_parameters - Dictionary of gradient of parameters with the same keys 
            returns updated parameters
        """
        # retreive values
        t = self.adam_config[k]["t"] + 1
        learning_rate = self.lr
        beta_1 = self.adam_config["beta_1"]
        beta_2 = self.adam_config["beta_2"]
        momentum = self.adam_config[k]["momentum"]
        velocity = self.adam_config[k]["velocity"]
        eps = self.adam_config["epsilon"]

        momentum = beta_1 * momentum + (1 - beta_1) * grad_parameter
        velocity = beta_2 * velocity + (1 - beta_2) * (grad_parameter**2)
        mhat = momentum/(1 - (beta_1 ** t))
        vhat = velocity/(1 - (beta_2 ** t))
        parameter -= learning_rate * mhat/(np.sqrt(vhat) + eps)
        self.adam_config[k]["t"] = t
        self.adam_config[k]["momentum"] = momentum
        self.adam_config[k]["velocity"] = velocity

        return parameter

    def iter_update(self):
        """
        Iterate over all parameters and update the values
        """
        for k in self.params.keys():
            self.params[k] = self.adam_update(k,self.params[k],self.grads[k])
    #######################

    def missclassification_rate(self,x,y):
        """
        Check missclassification rate
        """
        return np.mean(np.sum(x!=y))

    def model_process(self,x,y,mode = "training"):
        """
        Starts the process of the model
        Steps - 
            1) Forward propagate the network
            2) Calculate loss and gradient at output
            3) Backpropagate the network
            4) Update the parameters using gradient descent
        """
        if mode == "training":
            self.dropout_config["mode"] = "train"
        else:
            self.dropout_config["mode"] = "test"
        # forward prop
        out,hid_cache = self.forward_prop(x)
        # calculate loss 
        loss,delta,out = self.loss_grad(out,y)

        # validation dont update wts and return loss
        if mode != "training":
            return loss,out

        # backprop
        self.backward_prop(delta,hid_cache)
        # update weights
        self.iter_update()

        return loss,out

    def run(self):
        """
        Run the model
        """
        # initialize
        num_iter = self.epoch * self.num_batch
        training_loss_history = []
        training_miss_history = []
        validation_loss_history = []
        validation_miss_history = []
        epoch_num = 0
        batch_id = 0

        # for book keeping the best result
        best_params = {}

        # patience count for early stopping
        patience = 10

        for iteration in range(num_iter):
            # validate at the end of every epoch
            if iteration%self.num_batch == 0:
                epoch_num += 1
                # calculate loss of validation
                val_loss,out = self.model_process(self.X_val,self.y_val,mode = "validation")
                validation_loss_history.append(val_loss)
                val_miss = self.missclassification_rate(np.argmax(out,axis = 1),np.argmax(self.y_val,axis = 1))*100/out.shape[0]
                validation_miss_history.append(val_miss)

                # decrease patience if there is no sign of decrease in miss rate
                if len(validation_miss_history) > 2 and validation_miss_history[-2] <= val_miss:
                    patience -= 1
                    if patience == 0:
                        break
                # book keep best parameters and restore patience
                else:
                    best_params = self.params.copy()
                    patience = 10

                # calculate loss of training
                train_loss,out = self.model_process(self.X_train,self.y_train,mode = "validation")
                training_loss_history.append(train_loss)
                train_miss = self.missclassification_rate(np.argmax(out,axis = 1),np.argmax(self.y_train,axis = 1))*100/self.n
                training_miss_history.append(train_miss)
                # decay learning rate after certain number of epochs have completed
                # if epoch_num == 50:
                #     self.lr *= self.lr_decay
                if self.verbose and epoch_num%self.print_atevery == 0 or epoch_num == 1:
                    print " Epoch no : {},  Training loss is {} and validation loss is {}".format(epoch_num,train_loss,val_loss)

                batch_id = 0

            # continue normal training
            else:
                # Get minibatches and start the process
                x = self.X_train[batch_id * self.batch_size : (batch_id+1) * self.batch_size,:]
                y = self.y_train[batch_id * self.batch_size : (batch_id+1) * self.batch_size,:]
                batch_id = (batch_id + 1)%self.num_batch
                self.model_process(x,y)

        # plot
        _,axes = plt.subplots(1,2,figsize = (30,10))
        axes[0].plot(xrange(len(training_loss_history)),training_loss_history,label = "training loss")
        axes[0].plot(xrange(len(validation_loss_history)),validation_loss_history,label = "validation loss")
        axes[0].legend()
        axes[1].plot(xrange(len(training_miss_history)),training_miss_history,label = "training miss")
        axes[1].plot(xrange(len(validation_miss_history)),validation_miss_history,label = "validation miss")
        axes[1].legend()
        plt.show()
        print training_miss_history[-1]
        print validation_miss_history[-1]

        # get the best params
        self.params = best_params.copy()
        test_loss,out = self.model_process(self.X_test,self.y_test,mode = "testing")
        test_miss = self.missclassification_rate(np.argmax(out,axis = 1),np.argmax(self.y_test,axis = 1))*100/self.y_test.shape[0]

        print test_miss


if __name__ == "__main__":
    # Gather data
    X_train,y_train,X_test,y_test = load_mnist("Dataset/")
    # list of hidden neurons in each layer
    # hid_list = [1000,500,100]
    hid_list = [1000,250]

    # hyperparameters
    hyperparams = {}
    hyperparams["learning_rate"] = 1e-3
    hyperparams["learning_rate_decay"] = 0.9
    hyperparams["batch_size"] = 200
    hyperparams["epoch"] = 100
    hyperparams["reg"] = 1e-3

    # Initialize network
    net = NeuralNet(X_train,y_train,X_test,y_test,hid_list,hyperparams)
    # Run the network
    start_time = time.time()
    net.run()
    end_time = time.time()
    print "Total time to run is {}".format(end_time - start_time)

