import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, X, y, **kwargs):

        defaults = {
            "lr": 0.01,  # learning rate
            "lamda": 1,  # regularization parameter
            "batch_size": 100,
            "epochs": 10,
            "h_param": 1e-6,  # limit h for the numerical gradient check
            "m": 50,  # number of nodes in the hidden layer
            "lr_max": 1e-1,  # maximum for cyclical learning rate
            "lr_min": 1e-5  # minimum for cyclical learning rate
        }

        for key, def_value in defaults.items():
            setattr(self, key, kwargs.get(key, def_value))

        self.k = np.max(y) + 1
        self.d = np.shape(X)[0] #dimension
        self.N = np.shape(X)[1] #num_samples
        self.w1, self.w2 = self.w_initializer()
        self.b1, self.b2 = self.b_initializer()
        self.m = self.m
        self.ns = 2 * int(self.N / self.batch_size)
        print("ns: ", self.ns)



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
        h = np.maximum(0, s1)
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


    def compute_gradients(self, X_batch, y_true_batch, y_predicted_batch, h_batch):
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

        dW2 = 1 / self.batch_size * np.dot(G_batch, h_batch.T)
        db2 = 1 / self.batch_size * np.sum(G_batch, axis=1).reshape(-1, 1)

        G_batch = np.dot(self.w2.T, G_batch)
        ind = np.zeros(h_batch.shape)
        for i in range(h_batch.shape[0]):
            for j in range(h_batch.shape[1]):
                if h_batch[i, j] > 0:
                    ind[i, j] = 1
        G_batch = G_batch * ind

        dW1 = 1 / self.batch_size * np.dot(G_batch, X_batch.T)
        db1 = 1 / self.batch_size * np.sum(G_batch, axis=1).reshape(-1, 1)

        dW1 += 2 * self.lamda * self.w1
        dW2 += 2 * self.lamda * self.w2


        return dW1, db1, dW2, db2


    def mini_batch_GD(self, X_train, Y_train, X_val, Y_val):
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

                Y_pred_batch, h_pred_batch = self.evaluate_classifier(X_batch)
                dW1, db1, dW2, db2 = self.compute_gradients(X_batch, Y_batch, Y_pred_batch, h_pred_batch)

                self.w1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.w2 -= self.lr * dW2
                self.b2 -= self.lr * db2

                l = int((epoch * n_batches + i) / (2 * self.ns))
                if (epoch * n_batches + i) < (2 * l + 1) * self.ns:
                    self.lr = self.lr_min + ((epoch * n_batches + i) - 2 * l * self.ns) / self.ns * (self.lr_max - self.lr_min)
                else:
                    self.lr = self.lr_max - ((epoch * n_batches + i) - (2 * l + 1) * self.ns) / self.ns * (self.lr_max - self.lr_min)


            Y_pred_train, h = self.evaluate_classifier(X_train)
            train_cost = self.compute_cost(X_train, Y_pred_train)
            train_accuracy = self.accuracy(Y_pred_train, Y_train)

            Y_pred_val, h = self.evaluate_classifier(X_val)
            validation_cost = self.compute_cost(X_val, Y_pred_val)
            validation_accuracy = self.accuracy(Y_pred_val, Y_val)

            self.training_accuracies.append(train_accuracy)
            self.validation_accuracies.append(validation_accuracy)
            self.validation_costs.append(validation_cost)
            self.training_costs.append(train_cost)

            print(f"Epoch {epoch}: train accuracy: {train_accuracy}  train cost : {train_cost}  validation accuracy: {validation_accuracy} validation cost {validation_cost}")

        self.plot_over_epochs()


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


    ##### Code for Gradient checking #####

    def check_gradients(self, X, Y, method='finite_diff'):
        dw_num = np.zeros((self.k, self.d))
        Y_pred, h = self.evaluate_classifier(X)
        dW1, db1, dW2, db2 = self.compute_gradients(X, Y, Y_pred)
        if method == 'finite_diff':
            dW1_num, db1_num, dW2_num, db2_num = self.compute_gradient_num_fast(X, Y)
        elif method == 'centered_diff':
            dW1_num, db1_num, dW2_num, db2_num = self.compute_gradient_num_slow(X, Y)
        else:
            print("not valid name of the checking method")

        dw1_vector = dw1.flatten()
        dw1_num_vector = dw1_num.flatten()
        x_w1 = np.arange(1, dw1_vector.shape[0] + 1)
        plt.bar(x_w1, dw1_vector, 0.35, label='Analytical gradient', color='blue')
        plt.bar(x_w1 + 0.35, dw1_num_vector, 0.35, label=method, color='red')
        plt.legend()
        plt.title(("Gradient check of w1, batch size = " + str(X.shape[1])))
        plt.show()

        dw2_vector = dw2.flatten()
        dw2_num_vector = dw2_num.flatten()
        x_w2 = np.arange(1, dw2_vector.shape[0] + 1)
        plt.bar(x_w2, dw2_vector, 0.35, label='Analytical gradient', color='blue')
        plt.bar(x_w2 + 0.35, dw2_num_vector, 0.35, label=method, color='red')
        plt.legend()
        plt.title(("Gradient check of w2, batch size = " + str(X.shape[1])))
        plt.show()

        db1_vector = db1.flatten()
        db1_num_vector = db1_num.flatten()
        x_b1 = np.arange(1, db1.shape[0] + 1)
        plt.bar(x_b1, db1_vector, 0.35, label='Analytical gradient', color='blue')
        plt.bar(x_b1 + 0.35, db1_num_vector, 0.35, label=method, color='red')
        plt.legend()
        plt.title(("Gradient check of b1, batch size = " + str(X.shape[1])))
        plt.show()

        db2_vector = db2.flatten()
        db2_num_vector = db2_num.flatten()
        x_b2 = np.arange(1, db2.shape[0] + 1)
        plt.bar(x_b2, db2_vector, 0.35, label='Analytical gradient', color='blue')
        plt.bar(x_b2 + 0.35, db2_num_vector, 0.35, label=method, color='red')
        plt.legend()
        plt.title(("Gradient check of b2, batch size = " + str(X.shape[1])))
        plt.show()


    def compute_gradient_num_fast(self, X, Y_true):
        dw = np.zeros((self.k, self.d))
        db = np.zeros((self.k, 1))
        c = self.compute_cost(X, Y_true)
        for i in range(self.b.shape[0]):
            self.b[i] += self.h_param
            c2 = self.compute_cost(X, Y_true)
            db[i] = (c2 - c) / self.h_param
            self.b[i] -= self.h_param
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w[i, j] += self.h_param
                c2 = self.compute_cost(X, Y_true)
                dw[i, j] = (c2 - c) / self.h_param
                self.w[i, j] -= self.h_param
        return db, dw

    def compute_gradient_num_slow(self, X, Y_true):
        dw = np.zeros((self.k, self.d))
        db = np.zeros((self.k, 1))
        for i in range(self.b.shape[0]):
            self.b[i] -= self.h_param
            c1 = self.compute_cost(X, Y_true)
            self.b[i] += 2 * self.h_param
            c2 = self.compute_cost(X, Y_true)
            db[i] = (c2 - c1) / (2 * self.h_param)
            self.b[i] -= self.h_param
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w[i, j] -= self.h_param
                c1 = self.compute_cost(X, Y_true)
                self.w[i, j] += 2 * self.h_param
                c2 = self.compute_cost(X, Y_true)
                dw[i, j] = (c2 - c1) / (2 * self.h_param)
                self.w[i, j] -= self.h_param
        return db, dw















