import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.arange(1,51).reshape(50,1).astype('float64')
    x = np.concatenate((np.ones(50).reshape(50,1),x),axis = 1)
    u = np.random.uniform(-1,1,size = 50).reshape(50,1)
    y = x[:,1].reshape(50,1) + u 
    inv_mat = np.linalg.inv(x.T.dot(x))
    w = inv_mat.dot(x.T).dot(y)
    print(w)
    xx = np.linspace(0,51,10)
    yy = np.array(w[0]+ w[1] * xx)
    plt.plot(xx,yy.T,'r')
    plt.scatter(x[:,1],y)
    plt.show()

	
	
	
