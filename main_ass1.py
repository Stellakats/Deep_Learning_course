from one_layer_network import *
from data_preprocess import *


def main():

    """
    First, training, test and validation sets are loaded.
    Then, various experiments are conducted so as to test the network's performance.
    The flags below enable different experiments:
    exp0: This compares the gradients computed analytically in this assignment, against a numerical gradient
    computation method: centered_diff or  finite_diff (which is faster than the first but less accurate). It then plots
    the analytical and numerical grads for visual inspection purposes.
    exp1: Tests the net's performance for learning rate=0.01 and λ=0 and plots the weight vector of the 1st class.
    exp2: Tests the net's performance for learning rate=0.1 and λ=0 and plots the weight vector of the 1st class.
    exp3: Tests the net's performance for learning rate=0.01 and λ=0.1 and plots the weight vector of the 1st class.
    exp4: Tests the net's performance for learning rate=0.01 and λ=1 and plots the weight vector of the 1st class.
    exp5: Plots the weight vectors of all classes, for the optimum set parameters.
    """

    # Flags to make experiments :
    VIEW_IMAGE = False
    LOAD_TRAINING_DATASET = True
    LOAD_TEST_DATASET = True
    LOAD_VALIDATION_DATASET = True
    EXP0 = True
    EXP1 = False
    EXP2 = False
    EXP3 = False
    EXP4 = False
    EXP5 = False

    if VIEW_IMAGE:
        path = ""
        index = None
        image_viewer = ViewImage(path, index)
        image_viewer.plot_it()

    if LOAD_TRAINING_DATASET:
        train_data = "/Users/stella/Desktop/data_batch_1"
        load_train_data = LoadData(train_data)
        X_train, y_train, Y_train = load_train_data.load_batch()

    if LOAD_TEST_DATASET:
        test_data = "/Users/stella/Desktop/data_batch_2"
        load_test_data = LoadData(test_data)
        X_test, y_test, Y_test = load_test_data.load_batch()

    if LOAD_VALIDATION_DATASET:
        val_data = "/Users/stella/Desktop/test_batch"
        load_val_data = LoadData(val_data)
        X_val, y_val, Y_val = load_val_data.load_batch()

    if EXP0:
        net = NeuralNet(X_train, y_train, lr=0.01, lamda=0)
        net.check_gradients(X_train, Y_train, method='finite_diff')

    if EXP1:
        net = NeuralNet(X_train, y_train, lr=0.01, lamda=0)
        net.mini_batch_GD(X_train, Y_train, X_val, Y_val, al=False)

    if EXP2:
        net = NeuralNet(X_train, y_train, lr=0.1, lamda=0)
        net.mini_batch_GD(X_train, Y_train, X_val, Y_val, al=False)

    if EXP3:
        net = NeuralNet(X_train, y_train, lr=0.01, lamda=0.1)
        net.mini_batch_GD(X_train, Y_train, X_val, Y_val, al=False)

    if EXP4:
        net = NeuralNet(X_train, y_train, lr=0.01, lamda=1)
        net.mini_batch_GD(X_train, Y_train, X_val, Y_val, al=False)

    if EXP5:
        net = NeuralNet(X_train, y_train, lr=0.01, lamda=0)
        net.mini_batch_GD(X_train, Y_train, X_val, Y_val, al=True)



    print("ends")



if __name__ == "__main__":
    main()