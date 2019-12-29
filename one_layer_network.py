import numpy as np
from matplotlib import pyplot
import pickle


batch_1 = "/Users/stella/Desktop/data_batch_1"
names_in_strings = "/Users/stella/Desktop/batches.meta"


class ViewImage():

    def __init__(self, path, index):
        self.path = path
        self.image = index

    def unpickle(self, datapath):
        import pickle
        with open(datapath, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


    def plot_it(self):
        # unpickle the chosen data batch
        unpickled  = self.unpickle(self.path)
        #gets the numpy array of the image pixel data of all images in the batch
        image_data = unpickled[b'data']
        # gets the labels of all the images in the batch
        labels = unpickled[b'labels']
        # gets the actual names of the labels
        names = self.unpickle("/Users/stella/Desktop/batches.meta")

        #working on one image
        one_image = image_data[self.image, :]
        one_label = labels[self.image]
        one_name = names[b"label_names"][one_label]

        # setting the channels
        red = one_image[0:1024]
        red = np.reshape(red, (32, 32))
        green = one_image[1024:2048]
        green = np.reshape(green, (32, 32))
        blue = one_image[2048:]
        blue = np.reshape(blue, (32, 32))

        # depth wise stacking up to create the RGB matrix
        one_image = np.dstack((red, green, blue))

        # plot the image
        pyplot.imshow(one_image , interpolation='nearest')
        pyplot.title(f'{one_name}')
        pyplot.show()


def LoadBatch(path):
    """
    Arguments:
        path: the path of the dataset
    Returns:
        X: the image pixel data of size d x N = 3072 x 10000
        y: vector of length N that contains the labels of the images as integers between 0-1
        Y: matrix of size K x N that contains one hot representaion of the labels of the images. K=10
    """
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data'] / 255
    y = dict[b'labels']
    K = np.max(y)+1
    Y = np.eye(K)[y]

    return X.T, y, Y.T

def sizes(X,y):
    num_classes = np.max(y) + 1
    dimension = np.shape(X)[0]
    num_samples = np.shape(X)[1]
    return  num_classes, dimension, num_samples


def initializer(K, d, N):
    """
    A function to initialize the weights and bias
    :returns
    W: matrix of size K x d
    b : matrix of size K x N
    s  = W * x + b = [K x d] * [d x N] + [K * N]
    """

    W = 0.01 * np.random.randn(K, d)
    b = 0.01 * np.random.randn(K, N)

    return W, b

def EvaluateClassifier(X, W, b):
    """
    :param X: the image pixel data of size d x N = 3072 x 10000
    :param W: weight matrix of size K x d
    :param b: bias matrix of size K x N
    :return:
        the class probabilities for every sample. that is a matrix of size K x N = 10 x 10000
    """
    s = np.dot(W, X) + b
    return s


def Softmax(s):
    """
    Softmax function than receives scores and translates them to probabilities (k x N)
    """
    max = np.max(s)
    z = np.exp(s - max) # for numerical stability purposes
    p = z / np.sum(z)
    return p

def CrossEntropyLoss(p, Y):
    l_cross = np.log(np.dot(Y.T, p))
    return l_cross

def ComputeCost(l_cross, W, lamda, d):

    sum_1 = np.sum(l_cross)
    sum_2 = lamda * np.sum(np.square(W))
    J = (sum_1 / d) + sum_2
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



if __name__ == "__main__":

    # image = ViewImage(batch_1, 40)
    # image.plot_it()

    X_train, y_train, Y_train = LoadBatch("/Users/stella/Desktop/data_batch_1")
    K, d, N = sizes(X_train, y_train)
    W, b = initializer(K, d, N)
    s = EvaluateClassifier(X_train, W, b)
    p = Softmax(s)
    l_cross = CrossEntropyLoss(p, Y_train)
    J = ComputeCost(l_cross, W, 0.1, d)
    predicted = predict(p)
    accuracy = accuracy(predicted, y_train)


    print(np.shape(X_train), np.shape(y_train), np.shape(Y_train))







