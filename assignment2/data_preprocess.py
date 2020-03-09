import numpy as np
from matplotlib import pyplot
import pickle



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

class LoadData():

    def __init__(self, path):
        self.path = path

    def load_batch(self):
        """
        Arguments:
            path: the path of the dataset
        Returns:
            X: the image pixel data of size d x N = 3072 x 10000
            y: vector of length N that contains the labels of the images as integers between 0-1
            Y: matrix of size K x N that contains one hot representaion of the labels of the images. K=10
        """
        with open(self.path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        X = self.convert(dict[b'data'])
        y = dict[b'labels']
        K = np.max(y)+1
        Y = self.one_hot_encode(K, y)

        return X.T, y, Y.T

    
    def convert(self, array):
        """
        converts pixel values to 0-1 kind of values
        and centers them so that they have zero mean
        """
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        array /= 255.0
        array -= mean
        return array /std

    def one_hot_encode(self, K, y):
        """
        one hot encoding of the labels vector
        K is the number of classes
        """
        return np.eye(K)[y]


