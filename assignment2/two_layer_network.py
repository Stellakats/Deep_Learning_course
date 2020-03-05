import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, X, y, **kwargs):

        self.k = np.max(y) + 1
        self.d = np.shape(X)[0] #dimension
        self.N = np.shape(X)[1] #num_samples
        self.w1, self.w2 = self.w_initializer()
        self.b1, self.b2 = self.b_initializer()

        self.m = self.h_size
        self.ns = 2 * int(self.n / self.batch_size)
        print("ns: ", self.ns)

        defaults = {
            "lr": 0.01,  # learning rate
            "lamda": 1,  # regularization parameter
            "batch_size": 100,
            "epochs": 40,
            "h_param": 1e-6,  # limit h for the numerical gradient check
            "h_size": 50,  # number of nodes in the hidden layer
            "lr_max": 1e-1,  # maximum for cyclical learning rate
            "lr_min": 1e-5  # minimum for cyclical learning rate
        }

        for key, def_value in defaults.items():
            setattr(self, key, kwargs.get(key, def_value))

    def w_initializer(self):
        """
        A function to initialize the weights with 0 mean and 0.01 variance
        formula used: sigma * np.random.randn(...) + mu
        :returns
        W: matrix of size K x d
        dim-sanity check : s  = W * x + b = [K x d] * [d x N] + [K * N]
        """
        w1 = 1/np.sqrt(self.d) * np.random.randn(self.m, self.d)
        w2 = 1/np.sqrt(self.m) * np.random.randn(self.k, self.m)
        return w1, w2

    def b_initializer(self):
        """
        A function to initialize bias ith 0 mean and 0.01 variance
        :returns
        b : matrix of size K x N
        dim-sanity check : s  = W * x + b = [K x d] * [d x N] + [K * N]
        """
        b1 = 0.01 * np.random.randn(self.m, 1)
        b2 = 0.01 * np.random.randn(self.k, 1)
        return b1, b2

    def softmax(self, s):
        """
        Softmax function than receives scores and translates them to probabilities (k x N)
        (will be used in evaluate_classifier function)
        """

        p = np.exp(s) / np.dot(np.ones(s.shape[0]).T, np.exp(s))
        return p

    def evaluate_classifier(self, X):
        """
        :param X: the image pixel data of size d x N = 3072 x 10000
        :param W: weight matrix of size K x d
        :param b: bias matrix of size K x N
        :return:
            the class probabilities for every sample. that is a matrix of size K x N
        """
        s1 = np.dot(self.w1, X) + self.b1
        h = np.max(0, s1)
        s = np.dot(self.w2, h) + self.b2
        Y_pred = self.softmax(s)
        return Y_pred, h


    def cross_entropy_loss(self, Y_pred , Y_true):
        """
        will be used in the cost function
        """
        l_cross = np.sum(-np.log(np.sum(Y_true * Y_pred, axis=0)), axis=0)
        return l_cross

    def compute_cost(self, X, Y_true):
        """
        computed cost. to this end, it currently estimates Y pred
        :param Y_true:
        """
        Y_pred, h = self.evaluate_classifier(X)
        current_N = X.shape[1]

        l_cross = self.cross_entropy_loss(Y_pred, Y_true)
        reg = self.lamda * ( np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)) )
        j = (l_cross / current_N) + reg
        return j


    def accuracy(self, Y_pred, Y_true):
        """
        :param predicted: the output of predict function
        :param actual: the y_label (not the one hot encoded ones)
        :return:
            the accuracy
        """
        y_pred = np.array(np.argmax(Y_pred, axis=0))
        y_true = np.array(np.argmax(Y_true, axis=0))
        correct = len(np.where(y_true == y_pred)[0])

        return correct/ y_true.shape[0]

