import numpy as np
from matplotlib import pyplot
import pickle


class NeuralNet:

    def __init__(self, X, y, **kwargs):
        self.k = np.max(y) + 1
        self.d = np.shape(X)[0] #dimension
        self.N = np.shape(X)[1] #num_samples
        self.w = self.w_initializer()
        self.b = self.b_initializer()

        defaults = {
            "lr": 0.01,  # learning rate
            "m_weights": 0,  # mean of the weights
            "sigma_weights": 0.01,  # variance of the weights
            "labda": 0.1,  # regularization parameter
            "batch_size": 100,  # #examples per minibatch
            "epochs": 40,  # number of epochs
            "h_param": 1e-6  # parameter h for numerical grad check
        }

        for key, def_value in defaults.items():
            setattr(self, key, kwargs.get(key, def_value))

    def w_initializer(self):
        """
        A function to initialize the weights
        :returns
        W: matrix of size K x d
        s  = W * x + b = [K x d] * [d x N] + [K * N]
        """
        return 0.01 * np.random.randn(self.k, self.d)

    def b_initializer(self):
        """
        A function to initialize bias
        :returns
        b : matrix of size K x N
        s  = W * x + b = [K x d] * [d x N] + [K * N]
        """
        return 0.01 * np.random.randn(self.k, 1)




'''
def Softmax(s):
    """
    Softmax function than receives scores and translates them to probabilities (k x N)
    """
    max = np.max(s)
    z = np.exp(s - max) # for numerical stability purposes
    p = z / np.sum(z)
    return p

def EvaluateClassifier(X, W, b):
    """
    :param X: the image pixel data of size d x N = 3072 x 10000
    :param W: weight matrix of size K x d
    :param b: bias matrix of size K x N
    :return:
        the class probabilities for every sample. that is a matrix of size K x N
    """
    s = np.dot(W, X) + b
    p = Softmax(s)

    return p


def CrossEntropyLoss(p,Y):
    l_cross = np.sum(-np.log(np.sum(Y * p, axis=0)), axis=0)
    return l_cross



def ComputeCost(p, Y_train, W, lamda, N):
    """
    Arguments:
        p:
        Y_train:
        W:
        lamda:
        N:
    Returns:
        J:
    """

    l_cross = CrossEntropyLoss(p, Y_train)
    reg = lamda * np.sum(np.square(W))
    J = (l_cross / N) + reg
    return J

def predict(p):
    """
    :param p: a matrix of probabilities of size k x N
    :return:
        a vector of predictions, of size 1 x N, that contains the corresponding label (int between 0 and 9)
        of the class of maximum probability
    """
    maximum_class = np.argmax(p, axis=0)
    return maximum_class

def accuracy(predicted, actual):
    """
    :param predicted: the output of predict function
    :param actual: the y_label (not the one hot encoded ones)
    :return:
        the accuracy
    """
    N = np.shape(predicted)[0] # number of samples
    i=0
    sum = 0
    for i in range(N):
        if predicted[i] == actual[i]:
            sum += 1
        i += 1
    accuracy = sum / N
    return accuracy


def ComputeGradients(x_batch, y_batch, predicted_batch, W):
    """
    Recieves mini-batch of dataset, and corresponding p_batch
    and yields the gradients

    Arguments:
        x_batch: image pixel data, size d x n_b
        y_batch:  one hot representation of labels, size K x n_b
        predicted_batch: probabilities of predicted labels, size K x n_b
    Returns:
        the gradient of W, of size K x d
        the gradient of b, of size K x 1
    """
    n_b = np.shape(x_batch)[1]  # size of mini-batch
    db = np.zeros((10, 1))

    G_batch = - (y_batch - predicted_batch) # 10 x n_b


    dW = np.dot(G_batch, x_batch.T) / n_b
    dW += 2 * lamda * W
    db = np.dot(G_batch, np.ones(n_b).T).reshape(-1, 1) / n_b

    return dW, db

def MinibatchGD(X_train, Y_train, GDparams, W, b, lamda, N):
    """
    receives whole dataset, divides into batches and performs SGD
    Arguments:

    Returns:
        W: learnt weights
        b: learnt biases

    """
    dW, db = None, None
    n_batch = GDparams['n_batch']
    eta = GDparams['eta']
    n_epochs = GDparams['n_epochs']

    W, b = initializer(10, 3072)

    W_star = np.zeros(np.shape(W))
    b_star = np.zeros(np.shape(b))
    for epoch in range(n_epochs):

        for i in range(int(N / n_batch)):
            # create the mini batch
            i_start = i * n_batch
            i_end = i * n_batch + n_batch
            X_batch = X_train[:, i_start:i_end]
            Y_batch = Y_train[:, i_start:i_end]

            predicted_batch = EvaluateClassifier(X_batch, W_star, b_star)
            dW, db = ComputeGradients(X_batch, Y_batch, predicted_batch, W_star)

            W_star -= eta * dW
            b_star -= eta * db

        p = EvaluateClassifier(X_train, W_star, b_star)
        J = ComputeCost(p, Y_train, W, lamda, 10000)
        predicted = predict(p)
        acc = accuracy(predicted, y_train)
        print(f'for epoch:{epoch} the cost is {J} and the accuracy is {acc}\n')

    return W_star, b_star





'''













