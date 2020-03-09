from two_layer_network import *
from data_preprocess import *


def main():

    """
    First, training, test and validation sets are loaded.
    Then, various experiments are conducted so as to test the network's performance.
    The flags below enable different experiments:
    exp0: This compares the gradients computed analytically in this assignment, against a numerical gradient
    computation method: centered_diff or  finite_diff (which is faster than the first but less accurate). It then plots
    the analytical and numerical grads for visual inspection purposes.
    exp1: Tests the net's performance for 10 different λ values 
    """

    # Flags to make experiments :
    VIEW_IMAGE = False
    LOAD_TRAINING_DATASET = True
    LOAD_TEST_DATASET = True
    LOAD_VALIDATION_DATASET = True
    EXP0 = False
    EXP1 = True
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
        train_data = "data_batch_1"
        load_train_data = LoadData(train_data)
        X_train, y_train, Y_train = load_train_data.load_batch()

    if LOAD_TEST_DATASET:
        test_data = "test_batch"
        load_test_data = LoadData(test_data)
        X_test, y_test, Y_test = load_test_data.load_batch()

    if LOAD_VALIDATION_DATASET:
        val_data = "data_batch_2"
        load_val_data = LoadData(val_data)
        X_val, y_val, Y_val = load_val_data.load_batch()

    if EXP0:
        net = NeuralNet(X_train, y_train, lr=0.01, lamda=0)
        net.check_gradients(X_train, Y_train, method='finite_diff')

    if EXP1:
        lamdas = [2 * 1e-8, 4 * 1e-8, 6 * 1e-8, 8 * 1e-8, 1e-7, 2 * 1e-7, 4 * 1e-7, 6 * 1e-7, 8 * 1e-7, 2 * 1e-4,
                  4 * 1e-4, 6 * 1e-4, 8 * 1e-4, 1e-3, 2 * 1e-3, 4 * 1e-3, 6 * 1e-3, 8 * 1e-3]
        for lamda in lamdas:
            params = {
                "lamda": lamda,
                "epochs": 32
            }
            net = NeuralNet(X_train, y_train, lr=0.01, lamda=lamda)
            net.mini_batch_GD(X_train, Y_train, X_val, Y_val)

if __name__ == "__main__":
    main()