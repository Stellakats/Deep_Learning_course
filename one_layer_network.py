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
        return 0.01 * np.random.randn(self.k, self.d)

    def b_initializer(self):
        """
        A function to initialize bias
        :returns
        b : matrix of size K x N
        s  = W * x + b = [K x d] * [d x N] + [K * N]
        """
        return 0.01 * np.random.randn(self.k, 1)

    def softmax(self, s):
        """
        Softmax function than receives scores and translates them to probabilities (k x N)
        (will be used in evaluate_classifier function)
        """
        max = np.max(s)
        z = np.exp(s - max)  # for numerical stability purposes
        p = z / np.sum(z)
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
        correct = len(np.where(Y_true == Y_pred))
        print(Y_pred.shape, Y_true.shape) #[0] ???

        return correct/self.N











'''


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













