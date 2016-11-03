import numpy as np
import matplotlib.pyplot as plt

class Neuralnet:
    """
    Neural network instance
    """
    def __init__(self,n,h):
        """
        Instance initialization
        """
        # Input
        self.X = np.random.uniform(0,1,size = n).reshape(n,1)
        V = np.random.uniform(-0.1,0.1,size = n).reshape(n,1)

        # Desired Output
        self.D = np.sin(20 * self.X) + 3 * self.X + V

        # initalize weights
        self.W = np.random.rand(1,3*h + 1)

        # Number of inputs, outputs
        self.n = n
        self.h = h

        # learning rate
        self.lr = 1e-2

    def forward_prop(self,i):
        """
        Forward propagation
        """
        n,h = self.n,self.h

        # get weights and bias
        W1 = self.W[0,:h].reshape(1,h)
        b1 = self.W[0,h:2*h].reshape(1,h)
        W2 = self.W[0,2*h: 3*h].reshape(1,h)
        b2 = self.W[0,3*h:].reshape(1,1)

        # calculate output
        hiddenout = np.zeros((1,h))
        out = np.zeros((1,1))
        hiddenout = np.tanh(self.X[i] * W1 + b1)
        out = np.sum(hiddenout * W2 + b2)
        
        # store for backprop
        cache = W2

        return out,hiddenout,cache


    def mse_loss(self,out,y):
        """
        Calculate MSE loss
        """
        return (y - out)**2

    def tanh_derivative(self,x):
        """
        Derivative of tanh
        """
        return 1 - x*x

    def backward_prop(self,i,out,hiddenout,cache):
        """
        Backward propagation
        """

        W2 = cache
        h = self.h

        # Cost derivative
        cost_deriv = 2 * (out - self.D[i])
        
        # initailize
        gradW1 = np.zeros((1,h))
        gradb1 = np.zeros((1,h))
        gradW2 = np.zeros((1,h))
        gradb2 = np.zeros((1,1))

        # calculate gradients
        delta2 = cost_deriv * 1 
        gradW2 = hiddenout * delta2 
        gradb2 = delta2
        delta1 = delta2 * W2 * self.tanh_derivative(hiddenout)
        gradW1 = self.X[i] * delta1
        gradb1 = delta1

        # reshape
        gradW1 = gradW1.reshape(1,h)
        gradb1 = gradb1.reshape(1,h)
        gradW2 = gradW2.reshape(1,h)
        gradb2 = gradb2.reshape(1,1)

        grads = np.concatenate((gradW1,gradb1,gradW2,gradb2),axis = 1)

        return grads

    def update(self,grads):
        """
        Update the parameters
        """
        self.W -= self.lr * grads

    def run(self):
        """
        Run the network
        """
        # Online training of the network
        for epoch in range(100000):
            for i in range(self.n):
                out,hiddenout,cache = self.forward_prop(i)
                loss = self.mse_loss(out,self.D[i])
                grads = self.backward_prop(i,out,hiddenout,cache)
                self.update(grads)

        # Final output with trained weights
        net_out = np.zeros((1,self.n))
        for i in range(self.n):
            out,_,_ = self.forward_prop(i)
            net_out[0,i] = out

        # plot
        plt.scatter(self.X,self.D,c = "b")
        plt.scatter(self.X,net_out,c = "r")
        plt.show()

if __name__ == "__main__":
    net = Neuralnet(30,24)
    net.run()
