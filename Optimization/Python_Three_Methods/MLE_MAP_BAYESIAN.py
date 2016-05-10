'''
                                                                                                        Hyungwon Yang
                                                                                                           2016.04.10
                                                                                                            EMCS labs

This script introduces three methods: MLE, MAP, and fully Bayesian approch.
Please set the sample variable ranged between -1 to 1 in the box below.
This script is runnable through terminal by typing: python MLE_MAP_BAYESIAN.py
If you want to run this script in the python IDE then use result_plot function.


'''

###### PLEASE TYPE YOUR TEST SAMPLE X HERE ######
                                              ###
sample_X = 0.3                                ###
                                              ###
#################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

mu = 0
alpha = 5 * 10**-3
beta = 11.1
noise = np.random.normal(mu,1/beta,[100])
train_X = np.random.rand(100) * 2 - 1
target_Y = np.sin(2*np.pi*train_X) + noise

# True sine wave without noise
true_X = np.linspace(-1,1,100)
true_Y = np.sin(2*np.pi*true_X)

def MLE_prediction(sample_X,train_X,target_Y):

    '''
    Normal Equation: find optimal x values.
    >> Ax = b

    >> x = inv(A'*A)*A'*B

    My equation to solve
    >> w0x + w1x + w2x^2 + w3x^3 + w4x^4 + w5x^5 + w6x^6 + w7x^7 + w8x^8 + w9x^9 = target_Y

    What is the optimal values of W? Finding W that explains target_Y nicely is the task.
    '''

    # Generate initial input_X by polynomial equation.
    equation_X = lambda X:np.array([X**0, X**1, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9])
    input_X = equation_X(train_X).transpose()

    # Training the data by normal equation in oder to obtain optimal weight values.
    normal_equation = lambda train_X,target_Y: np.dot(np.dot(np.linalg.inv(np.dot(train_X.transpose(),train_X)),train_X.transpose()), target_Y)
    optimal_W = normal_equation(input_X,target_Y)

    # Predict the sample_X
    prediction_X = lambda sample_X,W : np.dot(W.transpose(),equation_X(sample_X))
    sample_Y = prediction_X(sample_X,optimal_W)
    print 'MLE predction result: predicted value is %7.5f given X value: %3.1f' % (sample_Y.squeeze(),sample_X)

    return np.array([sample_Y])

def MAP_prediction(sample_X,train_X,target_Y):

    '''
    MAP adds prior to the equation. However it is still looking for optimal W.
    Optimization: Powell method

    '''

    # Set initial W values and equations
    W = np.random.rand(10)
    equation_XW = lambda X,W: np.array(W[0]*X ** 0 + W[1]*X ** 1 + W[2]*X ** 2 + W[3]*X ** 3 + W[4]*X ** 4 +
                                       W[5]*X ** 5 + W[6]*X ** 6 + W[7]*X ** 7 + W[8]*X ** 8 + W[9]*X ** 9)
    equation_X = lambda X: np.array([X ** 0, X ** 1, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9])


    # Training the data by optimizatin toolbox.
    # Scipy.optimize.minimize > conjugate gradient method.
    loss_function = lambda W: beta/2.0*sum((np.dot(equation_X(train_X).transpose(),W) - target_Y)**2) + alpha/2.0*np.dot(W.transpose(),W)
    optimized = minimize(loss_function,W,jac=None,method='Powell',options={'disp':True})
    optimal_W = optimized.x

    # Predict the sample_X
    sample_Y = equation_XW(sample_X,optimal_W)
    print 'MAP predction result: predicted value is %7.5f given X value: %3.1f' % (sample_Y.squeeze(), sample_X)

    return np.array([sample_Y])

def Full_Bayesian(sample_X,train_X,target_Y):
    '''
    Full Bayesian does not use W values anymore. Y value is predicted only based on the probability.

    '''

    # phi_sample_X: for sample data
    phi_sample_X = lambda s_X: np.array([s_X**0, s_X**1, s_X**2, s_X**3, s_X**4, s_X**5, s_X**6, s_X**7, s_X**8, s_X**9]).squeeze()
    # phi_X: for training data
    phi_X = lambda X: np.array([X**0, X**1, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9]).squeeze()
    # S
    S = lambda X: np.linalg.inv(alpha*np.eye(10) + beta*np.dot(phi_X(X),phi_X(X).transpose()))
    # s**2
    S_square = lambda s_X,X,Y: 1./beta + np.dot(phi_sample_X(s_X).transpose(),np.dot(S(X),phi_sample_X(s_X)))
    # M
    M = lambda s_X,X,Y: beta * np.dot(phi_sample_X(s_X).transpose(),np.dot(S(X),np.dot(phi_X(X),Y)))

    # Predict the sample_X
    sample_mean_Y = M(sample_X,train_X,target_Y)
    sample_variance_Y = S_square(sample_X,train_X,target_Y)
    print 'Fuly_Bayesian predction result: predicted value is %7.5f given X value: %3.1f' % (sample_mean_Y.squeeze(), sample_X)

    return np.array([sample_mean_Y]), np.array([sample_variance_Y])


# Plotting predicted values, reference value(sine wave without noise), and target values (sine wave with noise)
def result_plot():

    # Derive all the predicted values from MLE, MAP, Fully Bayesian functions.
    MLE_Y = []
    MAP_Y = []
    FB_mean = []
    FB_variance = []

    # Prediction for true_X values.
    for iter in range(100):
        MLE_Y = np.concatenate((MLE_Y, MLE_prediction(true_X[iter], train_X, target_Y)))
        MAP_Y = np.concatenate((MAP_Y, MAP_prediction(true_X[iter], train_X, target_Y)))
        fb_mean, fb_variance = Full_Bayesian(true_X[iter], train_X, target_Y)
        FB_mean = np.concatenate((FB_mean, fb_mean))
        FB_variance = np.concatenate((FB_variance, fb_variance))

    # Prediction for one sample_X value.
    MLE_sample_Y = MLE_prediction(sample_X, train_X, target_Y)
    MAP_sample_Y = MAP_prediction(sample_X, train_X, target_Y)
    FB_mean_sample_Y, FB_variance_sample_Y = Full_Bayesian(sample_X, train_X, target_Y)

    plt.figure(1)

    # MLE
    plt.subplot(131)
    plt.plot(true_X, true_Y, 'r',label='Sine wave') # True sine wave without noise.
    plt.plot(train_X, target_Y, 'yo',label='Target points') # target sine wave. (added noise to true sine wave)
    plt.plot(true_X,MLE_Y,'b',label='MLE predicted points') # MLE Predicted wave.
    plt.plot(sample_X, MLE_sample_Y, 'ko', markerfacecolor='k', markersize=10,label='Predicted sample') # MLE predicted sample data.
    plt.title('MLE',fontsize=15)
    plt.xlabel('X range',fontsize=13)
    plt.ylabel('Y range',fontsize=13)
    plt.legend(loc='upper center',fontsize=8)
    plt.axis([-1, 1, -2, 2])

    # MAP
    plt.subplot(132)
    plt.plot(true_X, true_Y, 'r',label='Sine wave') # True sine wave without noise.
    plt.plot(train_X, target_Y, 'yo',label='Target points') # target sine wave. (added noise to true sine wave)
    plt.plot(true_X,MAP_Y,'b',label='MAP predicted points') # MAP Predicted wave.
    plt.plot(sample_X, MAP_sample_Y, 'ko', markerfacecolor='k', markersize=10,label='Predicted sample') # MAP predicted sample data.
    plt.title('MAP',fontsize=15)
    plt.xlabel('X range',fontsize=13)
    plt.ylabel('Y range',fontsize=13)
    plt.legend(loc='upper center',fontsize=8)
    plt.axis([-1, 1, -2, 2])

    # Fully Bayesian.
    plt.subplot(133)
    plt.plot(true_X, true_Y, 'r',label='Sine wave') # True sine wave without noise.
    plt.plot(train_X, target_Y, 'yo',label='Target points') # target sine wave. (added noise to true sine wave)
    plt.plot(true_X,FB_mean,'b',label='FB predicted points') # FB Predicted wave.
    max_range = FB_mean + FB_variance
    min_range = FB_mean - FB_variance
    plt.plot(true_X,max_range,'--g',label='+-1 std range')
    plt.plot(true_X,min_range,'--g')
    plt.plot(sample_X, FB_mean_sample_Y, 'ko', markerfacecolor='k', markersize=10,label='Predicted sample') # FB predicted sample data.
    plt.title('Fully Bayesian',fontsize=15)
    plt.xlabel('X range',fontsize=13)
    plt.ylabel('Y range',fontsize=13)
    plt.legend(loc='upper center',fontsize=8)
    plt.axis([-1, 1, -2, 2])

    plt.show()

if __name__ == '__main__':
    result_plot()