import numpy as np
from matplotlib import image, pyplot


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



if __name__ == "__main__":

    image = ViewImage(batch_1, 30)
    image.plot_it()





