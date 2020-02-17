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
            "lamda": 0.1,  # regularization parameter
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
        return np.random.normal(0, 0.01, (self.k, self.d))

    def b_initializer(self):
        """
        A function to initialize bias
        :returns
        b : matrix of size K x N
        s  = W * x + b = [K x d] * [d x N] + [K * N]
        """
        return np.random.normal(0, 0.01, (self.k, 1))

    def softmax(self, s):
        """
        Softmax function than receives scores and translates them to probabilities (k x N)
        (will be used in evaluate_classifier function)
        """

        p = np.exp(s) / np.dot(np.ones(s.shape[0]).T , np.exp(s))
        return p

    def evaluate_classifier(self, X):
        """
        :param X: the image pixel data of size d x N = 3072 x 10000
        :param W: weight matrix of size K x d
        :param b: bias matrix of size K x N
        :return:
            the class probabilities for every sample. that is a matrix of size K x N
        """
        s = np.dot(self.w, X) + self.b
        Y_pred = self.softmax(s)
        return Y_pred


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
        :return:
        """
        Y_pred = self.evaluate_classifier(X)
        current_N = X.shape[1]
        l_cross = self.cross_entropy_loss(Y_pred, Y_true)
        reg = self.lamda * np.sum(np.square(self.w))
        j = (l_cross / current_N) + reg
        return j

    def predict(self, Y_pred):
        """
        will be used in accuracy method
        :param Y_pred: a matrix of probabilities of size k x N
        :return:
            an array (vector) of predictions, of size 1 x N, that contains the corresponding label (int between 0 and 9)
            of the class of maximum probability
        """
        maximum_class = np.argmax(Y_pred, axis=0)
        return np.array(maximum_class)

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


    def compute_gradients(self, x_batch, y_true_batch, y_predicted_batch):
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
        G_batch = - (y_true_batch - y_predicted_batch) # 10 x batch_size

        dW = 1 / self.batch_size * np.dot(G_batch, x_batch.T)
        dW += 2 * self.lamda * self.w

        db = 1 / self.batch_size * np.sum(G_batch, axis=1).reshape(-1, 1)

        return dW, db


    def mini_batch_GD(self, X_train, Y_train):
        """
        receives whole dataset, divides into batches and performs SGD
        Arguments:

        Returns:
            W: learnt weights
            b: learnt biases

        """
        n_batches = int(self.N / self.batch_size)

        for epoch in range(self.epochs):

            for i in range(n_batches):
                # create the mini batch
                i_start = i * self.batch_size
                i_end = i * self.batch_size + self.batch_size
                X_batch = X_train[:, i_start:i_end]
                Y_batch = Y_train[:, i_start:i_end]

                Y_pred_batch = self.evaluate_classifier(X_batch)
                dW, db = self.compute_gradients(X_batch, Y_batch, Y_pred_batch)

                self.w -= self.lr * dW
                self.b -= self.lr * db


            Y_pred_train = self.evaluate_classifier(X_train)
            #Y_pred_val = self.evaluate_classifier(X_val)
            cost_train = self.compute_cost(X_train, Y_pred_train)
            acc_train = self.accuracy(Y_pred_train, Y_train)
            #cost_val = self.compute_cost(X_val, Y_pred_val)
            #acc_val = self.accuracy(Y_pred_val, Y_val)

            print("Epoch ", epoch, " // Train accuracy: ", acc_train, " // Train cost: ", cost_train)

        #return self.w, self.b



















