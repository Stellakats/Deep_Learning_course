import numpy as np
from matplotlib import pyplot
import numpy as np
import pickle
from one_layer_network import *
from data_preprocess import *



def main():
    """
    well, thats my main
    """

    # Flags to decide which part of  the program should be run
    VIEW_IMAGE = False
    LOAD_TRAINING_DATASET = True
    LOAD_TEST_DATASET = False
    LOAD_VALIDATION_DATASET = False

    if VIEW_IMAGE:
        path=""
        index=None
        image_viewer = ViewImage(path,index)
        image_viewer.plot_it()

    if LOAD_TRAINING_DATASET:
        train_data = "/Users/stella/Desktop/data_batch_1"
        load_train_data = LoadData(train_data)
        X_train, y_train, Y_train = load_train_data.load_batch()

    if LOAD_TEST_DATASET:
        test_data = ""
        load_test_data = LoadData(test_data)
        X_test, y_test, Y_test = load_test_data.load_batch()

    if LOAD_VALIDATION_DATASET:
        val_data = ""
        load_val_data = LoadData(val_data)
        X_val, y_val, Y_val = load_val_data.load_batch()




    lamda = 0.1
    GDparams = {'n_batch': 100, 'eta': 0.01, 'n_epochs': 20}

    #K, d, N = sizes(X_train, y_train)
    #W, b = initializer(K, d)

    #W_star, b_star = MinibatchGD(X_train, Y_train, GDparams, W, b, lamda, N)
    #p = EvaluateClassifier(X_train, W_star, b_star)
    #J = ComputeCost(p, Y_train, W, lamda, 10000)
    # print(f'for epoch:{epoch} the cost is {J}')
    #predicted = predict(p)
    #acc = accuracy(predicted, y_train)
    # print(f'for epoch:{epoch} the accuracy is {acc}\n')


    print("ends")



if __name__ == "__main__":
    main()