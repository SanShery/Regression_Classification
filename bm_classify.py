import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0
    y[y == 0] = -1
    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #

        matrix_sign = np.array(N * [0])
        X = np.insert(X, 0, [1], axis=1)
        i = 0
        while (i < max_iterations):
            w = np.insert(w, 0, [b])
            a1 = np.multiply(y, np.matmul(X, w))

            k = a1 <= 0
            matrix_sign = k.astype(int)

            Modif = np.matmul(np.transpose(np.multiply(matrix_sign * step_size, y)), X)
            Modif = Modif / N
            w = w + Modif
            b = w[0]
            w = w[1:]
            i = i + 1
            ############################################

    elif loss == "logistic":

        X = np.insert(X, 0, [1], axis=1)
        i = 0

        while i < max_iterations:
            w = np.insert(w, 0, [b])
            a1 = np.multiply(y, np.matmul(X, w))
            Modif = np.matmul(np.transpose(np.multiply(sigmoid(-a1) * step_size, y)), X)
            w = w + Modif / N
            b = w[0]
            w = w[1:]
            i = i + 1


            ############################################
            # TODO 2 : Edit this if part               #
            #          Compute w and b here            #
            # w = np.zeros(D)
            # b = 0
            ############################################


    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    d = np.exp(-z)
    d_new = 1 + d
    value = 1 / d_new

    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        preds = np.zeros(N)
        o = np.matmul(X, w)
        o = o + b
        o[o <= 0] = 0
        o[o > 0] = 1
        # np.place(o, o > 0, [1])
        # np.place(o, o <= 0, [0])

        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #


        ############################################
        preds = o

    elif loss == "logistic":
        preds = np.zeros(N)
        o = np.matmul(X, w)
        o1 = o + b
        o2 = sigmoid(o1)
        o2[o2 <= 0.5] = 0
        o2[o2 > 0.5] = 1
        # np.place(o2, o2 > 0.5, [1])
        # np.place(o2, o2 <= 0.5, [0])
        preds = o2


        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #

        ############################################


    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    N, D = X.shape
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    b = np.zeros(C)
    if b0 is not None:
        b = b0
    np.random.seed(42)
    if gd_type == "sgd":
        X = np.insert(X, D, [1], axis=1)

        for i in range(max_iterations):
            n = np.random.choice(range(N))
            w = np.insert(w, D, b, axis=1)
            pro = np.matmul(w, np.transpose(X[n]))
            pro_1 = softMax(pro)
            pro_1 = pro_1[:, np.newaxis]
            XX = X[n]
            XX = XX[:, np.newaxis]
            pro_1[y[n]] = pro_1[y[n]] - 1
            pro_2 = np.matmul(pro_1, np.transpose(XX))
            w = w - step_size * pro_2
            b = w[:, D]
            w = w[:, :-1]
    elif gd_type == "gd":
        X = np.insert(X, D, [1], axis=1)
        targets = y.reshape(-1)
        y_new = np.eye(C)[targets]
        for i in range(max_iterations):
            w = np.insert(w, D, b, axis=1)
            pro = np.matmul(X, np.transpose(w))
            pro = pro - pro.mean(axis=1, keepdims=True)
            pro_1 = softMax_1(pro, N)
            # pro_1[z][np.where(y_new[z] == 1)] = pro_1[z][np.where(y_new[z] == 1)] - 1
            # pro_1[:, np.where(y_new[:, ] == 1)]
            pro_1[np.where(y_new[:, ] == 1)] = pro_1[np.where(y_new[:, ] == 1)] - 1
            pro_2 = np.matmul(np.transpose(pro_1), X)
            w = w - step_size * pro_2 / N
            b = w[:, D]
            w = w[:, :-1]
    else:
        raise "Type of Gradient Descent is undefined."
    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def softMax(X):
    e_x = np.exp(X - np.max(X))
    return e_x / np.sum(e_x, axis=0)


def softMax_1(X, N):
    X = np.exp(X)
    X = X / X.sum(axis=1, keepdims=True)
    return X

    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #

    preds = np.zeros(N)
    Pre = (np.matmul(X, np.transpose(w)) + b)
    preds = np.argmax(Pre, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds

