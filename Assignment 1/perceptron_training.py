import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    """
        Step function returns
        1 if x >= 0
        0 otherwise
    """
    return np.piecewise(x,[x<0,x>=0],[0,1])


class Perceptron:
    """
        Perceptron Class
    """
    
    def _inp(self,n):
        """
            Randomly choose the inputs
        """
        x = np.random.uniform(-1,1,size = n * 2).reshape(n,2)
        return x

    def _vis(self,S0,S1,W):
        """
            Visualize scatter plot
        """
        w0,w1,w2 = W[0],W[1],W[2]
        x = S0[:,0]
        y = S0[:,1]
        s0plot = plt.scatter(x,y,c = 'r',label = 's0')
        x = S1[:,0]
        y = S1[:,1]
        s1plot = plt.scatter(x,y,c= 'b',label = 's1')
        x = np.array(range(-1,2))
        y = (w0 + w1 * x)/(-w2)
        boundary = plt.plot(x,y,c = 'k',label = 'boundary')
        plt.legend()
        plt.show()

    def _algo(self,neta,S_mod,W_dash,truth):
        """
            Perceptron Training Algorithm
        """

        # Training Algorithm
        epoch = 0
        num_miss = 100
        training_parameter = neta
        miss_history = []
        n = S_mod.shape[0]
        while num_miss != 0:
            epoch += 1
            for i in xrange(n):
                X = (S_mod[i,:]).reshape(1,3) # X shape 1X3
                out = step_function(np.dot(X,W_dash))
                diff = truth[i] - out
                W_dash = W_dash + training_parameter * np.dot(X.T,diff).reshape(3,1)
            out = step_function(np.dot(S_mod,W_dash))
            num_miss = np.sum(out != truth)
            miss_history.append(num_miss)

        # For comparing W and W_final
        if training_parameter == 1:
            self.W_final = W_dash
        
        return (epoch,miss_history)


    def _missvis(self,training_parameter,total_epoch,miss_history):
        """
            Visualize misclassification
        """
        plt.plot(xrange(total_epoch),miss_history,label = training_parameter)
        plt.xlabel('Epoch')
        plt.ylabel('Misclassifications')
    
    def train(self,n,parameters):
        """
            Training perceptron
        """

        # Initialize weights
        w0 = np.random.uniform(-0.25,0.25)
        w1 = np.random.uniform(-1,1)
        w2 = np.random.uniform(-1,1)
        W = np.vstack((w0,w1,w2)) # W shape 3x1

        print W

        # Input
        S = self._inp(n)  # S shape - Nx2
        temp = np.ones(n)
        S_mod = np.column_stack((temp,S)) # S_mod shape Nx3

        # Desired Output
        truth = step_function(np.dot(S_mod,W)) # truth shape Nx1

        # S0 and S1
        mask = np.zeros_like(S_mod)
        mask = truth == 0
        mask = mask.squeeze()
        S0 = S[mask]
        S1 = S[np.logical_not(mask)]

        # Visualize
        self._vis(S0,S1,W)

        # Random weights
        w00 = np.random.uniform(-1,1)
        w01 = np.random.uniform(-1,1)
        w02 = np.random.uniform(-1,1)
        W_dash = np.vstack((w00,w01,w02)) # W shape 3x1 

        print W_dash 

        for training_parameter in parameters:
            total_epoch,miss_history = self._algo(training_parameter,S_mod,W_dash,truth)
            self._missvis(training_parameter,total_epoch,miss_history)

        # Show all missvis in a single graph
        plt.legend()
        plt.show()

        print self.W_final

if __name__  == "__main__":
    classifier = Perceptron()
    classifier.train(100,[1,10,0.1])
    classifier.train(1000,[1,10,0.1])
