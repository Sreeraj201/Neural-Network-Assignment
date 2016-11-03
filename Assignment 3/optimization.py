import time
import numpy as np
import matplotlib.pyplot as plt

def rand_generator():
    """
    Random x and y coordinates generator
    Satisifies the following equations
        - x > 0 
        - y > 0
        - x + y < 1
    """
    while True:
        w = np.random.randn(1,2)
        x,y = w[0,0],w[0,1]
        if ((x+y) < 1) and (x > 0) and (y > 0): break
    return w

def f(x,y):
    """
    Function: -(log(1-x-y))-logx-logy
    """
    if ((x+y) < 1) and (x > 0) and (y > 0):
        return -np.log(1-x-y)-np.log(x)-np.log(y)
    else:
        return None

def grad_f(x,y):
    """
    Gradient of f wrt x,y
    """
    grad_x = (1/(1-x-y)) - (1/x)
    grad_y = (1/(1-x-y)) - (1/y)
    return np.asarray([grad_x,grad_y])

def hess_f(x,y):
    """
    Hessian matrix of f wrt x,y
    """
    gradx_x = (1/((1-x-y)* (1-x-y))) + (1/(x*x))
    gradx_y = 1/((1-x-y) * (1-x-y))
    grady_x = gradx_y
    grady_y = (1/((1-x-y)* (1-x-y))) + (1/(y*y))
    return np.asarray([gradx_x,gradx_y,grady_x,grady_y]).reshape(2,2)

def plotter(num,energy_history):
    """
    plot of the energy function
    """
    plt.plot(range(num),energy_history)
    plt.show()

def change(x,y):
    """
    Return relative difference
    """
    return np.max(np.abs(x - y)/(np.maximum(1e-10,np.abs(x) + np.abs(y))))

def optimize(eta = 1.0,hess = False,verbose = False):
    """
    Optimization : Gradient Descent or Newton
    Inputs: 
        eta : learning rate; default is 1.0
        hess : hessian indicator; if False then optimization is GD,otherwise Newton
        verbose : Print values at each iteration
    """
    # Time
    start_time = time.time()

    # initialize
    w = rand_generator()
    print(w)
    energy_history = []
    num_iter = 0 

    x,y = w[0,0],w[0,1] 
    prev_ener = f(x,y)
    w_prev = w + 1

    energy_history.append(prev_ener)

    # till relative change is very less (for faster approx. convergence)
    while change(w_prev,w)> 1e-10:

        # increase iterations
        num_iter += 1
        # copy values
        if num_iter > 1:prev_ener = next_ener
        w_prev = w.copy()

        # Gradient Descent
        if not hess: w = w - eta * grad_f(x,y) 
        # Newton Method
        else: w = w - eta * np.dot(grad_f(x,y),np.linalg.inv(hess_f(x,y)))  

        x,y = w[0,0],w[0,1]
        next_ener = f(x,y) 

        # If undefined, restart at new w with learning rate decreased by half
        if next_ener is None:return (True,eta/2)

        # print every iteration if verbose
        if verbose: print("Energy at iteration {} is {}".format(num_iter,next_ener))

        if (prev_ener >= next_ener):
            # store values
            energy_history.append(next_ener)
        else:
            # Revert to previous weights,energies and decrease learning rate by order of 10
            next_ener = prev_ener
            num_iter -= 1
            w = w_prev.copy()
            w_prev += 1
            eta = eta/10

    # Remove last repeated value
    energy_history.pop()
    # Time
    end_time = time.time() - start_time
    print("total time is {}".format(end_time))

    plotter(num_iter,energy_history)
    return (False,eta)

if __name__ == "__main__":
    loop = True
    eta = 1.0
    while loop:
        loop,eta = optimize(eta = eta,hess = False,verbose = True)
