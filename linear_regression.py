"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv


###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    k=np.square(np.subtract(np.matmul(X,w),y))
    k=np.mean(k)
    return k
    #####################################################
    #err = None
    #return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """

    w=np.matmul(np.transpose(X),X)
    w=inv(w)
    w = np.matmul(w, np.transpose(X))
    w = np.matmul(w, y)
    return w

  #####################################################


  #####################################################
    #w = None
    #return w

###### Q1.3 ######
def linear_regression_invertible(X, y):

    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    Xnew = np.matmul(np.transpose(X), X)
    ev = np.linalg.eigvals(Xnew)
    min = np.min(ev)
    k = 10 ** (-5)
    lam = 10 ** (-1)
    while(min<k):
            Xnew = Xnew+ lam*np.identity(np.size(X,1))#np.size(X,0) gives no of rows in X; (X,1): No of cols in X
            ev = np.linalg.eigvals(Xnew)
            min = np.min(ev)
    w=inv(Xnew)
    w=np.matmul(w,np.transpose(X))
    w=np.matmul(w,y)
    return w
    #####################################################
    #w = None
    #return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    w=np.matmul(np.transpose(X),X)
    t=lambd*np.identity(np.size(X,1))
    w=w+t
    w=inv(w)
    w=np.matmul(w,np.transpose(X))
    w=np.matmul(w,y)
    return w
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    #w = None
    #return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    import math
    mse=1.0e10
    bestlam=10**-19
    lam=10**-19
    ini=-19
    while(ini<20):
        while (lam<=10**19):
            w=regularized_linear_regression(Xtrain,ytrain,lam)
            mse1=mean_square_error(w,Xval,yval)
            if(mse1<mse):
                mse=mse1
                bestlam=lam
            ini=ini+1
            lam=math.pow(10,ini)
    return bestlam
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################

    

###### Q1.6 ######
def mapping_data(X, power):
    k=X
    i=2
    for i in range(2,power+1):
        X_new=X**i
        k=np.append(k,X_new[:,1:],axis=1)
    return k
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    
    return X


