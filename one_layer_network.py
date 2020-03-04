import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, X, y, **kwargs):

        self.k = np.max(y) + 1
        self.d = np.shape(X)[0] #dimension
        self.N = np.shape(X)[1] #num_samples
        self.w = self.w_initializer()
        self.b = self.b_initializer()

        defaults = {
            "lr": 0.01,  # learning rate
            "lamda": 1,  # regularization parameter
            "batch_size": 100,
            "epochs": 40,
            "h_param": 1e-6  # limit h for the numerical gradient check
        }

        for key, def_value in defaults.items():
            setattr(self, key, kwargs.get(key, def_value))

    def w_initializer(self):
        """
        A function to initialize the weights with 0 mean and 0.01 variance
        :returns
        W: matrix of size K x d
        dim-sanity check : s  = W * x + b = [K x d] * [d x N] + [K * N]
        """
        return 0.01 * np.random.randn(self.k, self.d)

    def b_initializer(self):
        """
        A function to initialize bias ith 0 mean and 0.01 variance
        :returns
        b : matrix of size K x N
        dim-sanity check : s  = W * x + b = [K x d] * [d x N] + [K * N]
        """
        return 0.01 * np.random.randn(self.k, 1)

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


    def mini_batch_GD(self, X_train, Y_train, X_val, Y_val, al):
        """
        receives whole dataset, divides into batches and performs SGD
        Arguments:

        Returns:
            W: learnt weights
            b: learnt biases

        """
        n_batches = int(self.N / self.batch_size)
        self.training_accuracies = []
        self.validation_accuracies = []
        self.training_costs = []
        self.validation_costs = []

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
            train_cost = self.compute_cost(X_train, Y_pred_train)
            train_accuracy = self.accuracy(Y_pred_train, Y_train)

            Y_pred_val = self.evaluate_classifier(X_val)
            validation_cost = self.compute_cost(X_val, Y_pred_val)
            validation_accuracy = self.accuracy(Y_pred_val, Y_val)

            self.training_accuracies.append(train_accuracy)
            self.validation_accuracies.append(validation_accuracy)
            self.validation_costs.append(validation_cost)
            self.training_costs.append(train_cost)

            print(f"Epoch {epoch}: train accuracy: {train_accuracy}  cost : {train_cost} and validation accuracy: {validation_accuracy} ")

        self.plot_over_epochs()
        self.plot_w(al)

    def plot_over_epochs(self):

        fig = plt.figure(figsize=(10, 7))

        fig.suptitle(f'$\lambda$={self.lamda} , l_r={self.lr}', fontsize=16, y=0.98)

        x = np.arange(1, self.epochs+1)

        ax1 = fig.add_subplot(121)
        plt.plot(x, self.training_accuracies, c='k', label="Training Accuracy")
        plt.plot(x, self.validation_accuracies, c='r', label="Validation Accuracy")
        ax1.set(title=f'Accuracy over Epochs', ylabel='Accuracy', xlabel='epochs')
        ax1.set(xlim=[0, self.epochs])
        ax1.legend(loc='best')

        ax2 = fig.add_subplot(122)
        plt.plot(x, self.training_costs, c='k', label='Training Cost')
        plt.plot(x, self.validation_costs, c='r', label='Validation Cost')
        ax2.set(title='Cost over Epochs', ylabel='Cost', xlabel='epochs')
        ax2.set(xlim=[0, self.epochs+1])
        ax2.legend(loc='best')

        plt.show()
        #plt.savefig('')

    def plot_w(self, al):

        if al:
            for i in range(self.k):
                w_image = self.w[i, :].reshape((32, 32, 3), order='F')
                w_image = ((w_image - w_image.min()) / (w_image.max() - w_image.min()))
                w_image = np.rot90(w_image, 3)
                plt.imshow(w_image)
                plt.xticks([])
                plt.yticks([])
                plt.title("Class " + str(i))

                plt.savefig(f'{i}')

        else:
            w_image = self.w[1, :].reshape((32, 32, 3), order='F')
            w_image = ((w_image - w_image.min()) / (w_image.max() - w_image.min()))
            w_image = np.rot90(w_image, 3)
            plt.imshow(w_image)
            plt.xticks([])
            plt.yticks([])
            plt.title("Class " + str(1))
            plt.show()
            #plt.savefig('')

    ##### Code for Gradient checking #####

    def check_gradients(self, X, Y, method='finite_diff'):
        grad_w_num = np.zeros((self.k, self.d))
        Y_pred = self.evaluate_classifier(X)
        grad_b, grad_w = self.compute_gradients(X, Y, Y_pred)
        if method == 'finite_diff':
            grad_b_num, grad_w_num = self.compute_gradient_num_fast(X, Y)
        elif method == 'centered_diff':
            grad_b_num, grad_w_num = self.compute_gradient_num_slow(X, Y)
        else:
            print(method, "is not a valid method to check gradients")

        grad_w_vec = grad_w.flatten()
        grad_w_num_vec = grad_w_num.flatten()
        x_w = np.arange(1, grad_w_vec.shape[0] + 1)
        plt.bar(x_w, grad_w_vec, 0.25, label='Analytical gradient', color='k')
        plt.bar(x_w + 0.25, grad_w_num_vec, 0.25, label=method, color='red')
        plt.legend()
        plt.title(("w gradient, batch size = " + str(X.shape[1])))
        plt.show()
        rel_error = abs(grad_w_vec / grad_w_num_vec - 1)
        print("method = ", method)
        print("W gradients")
        print("mean relative error: ", np.mean(rel_error))

        grad_b_vec = grad_b.flatten()
        grad_b_num_vec = grad_b_num.flatten()
        x_b = np.arange(1, grad_b.shape[0] + 1)
        plt.bar(x_b, grad_b_vec, 0.25, label='Analytical gradient', color='k')
        plt.bar(x_b + 0.25, grad_b_num_vec, 0.25, label=method, color='red')
        plt.legend()
        plt.title(("b gradient check, batch size = " + str(X.shape[1])))
        plt.show()
        rel_error = abs(grad_b_vec / grad_b_num_vec - 1)
        print("Bias gradients")
        print("mean relative error: ", np.mean(rel_error))


    def compute_gradient_num_fast(self, X, Y_true):
        grad_w = np.zeros((self.k, self.d))
        grad_b = np.zeros((self.k, 1))
        c = self.compute_cost(X, Y_true)
        for i in range(self.b.shape[0]):
            self.b[i] += self.h_param
            c2 = self.compute_cost(X, Y_true)
            grad_b[i] = (c2 - c) / self.h_param
            self.b[i] -= self.h_param
        for i in range(self.w.shape[0]):  # k
            for j in range(self.w.shape[1]):  # d
                self.w[i, j] += self.h_param
                c2 = self.compute_cost(X, Y_true)
                grad_w[i, j] = (c2 - c) / self.h_param
                self.w[i, j] -= self.h_param
        return grad_b, grad_w

    def compute_gradient_num_slow(self, X, Y_true):
        grad_w = np.zeros((self.k, self.d))
        grad_b = np.zeros((self.k, 1))
        for i in range(self.b.shape[0]):
            self.b[i] -= self.h_param
            c1 = self.compute_cost(X, Y_true)
            self.b[i] += 2 * self.h_param
            c2 = self.compute_cost(X, Y_true)
            grad_b[i] = (c2 - c1) / (2 * self.h_param)
            self.b[i] -= self.h_param
        for i in range(self.w.shape[0]):  # k
            for j in range(self.w.shape[1]):  # d
                self.w[i, j] -= self.h_param
                c1 = self.compute_cost(X, Y_true)
                self.w[i, j] += 2 * self.h_param
                c2 = self.compute_cost(X, Y_true)
                grad_w[i, j] = (c2 - c1) / (2 * self.h_param)
                self.w[i, j] -= self.h_param
        return grad_b, grad_w















