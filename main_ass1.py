from one_layer_network import *
from data_preprocess import *


def main():
    # Flags to conduct experiments :
    VIEW_IMAGE = False
    LOAD_TRAINING_DATASET = True
    LOAD_TEST_DATASET = False
    LOAD_VALIDATION_DATASET = True
    EXP1 = True
    EXP2 = True
    EXP3 = True
    EXP4 = True

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

    if EXP1:
        net = NeuralNet(X_train, y_train, lr=0.01, lamda=0)
        net.mini_batch_GD(X_train, Y_train, X_val, Y_val)
    
    if EXP2:
        net = NeuralNet(X_train, y_train, lr=0.1, lamda=0)
        net.mini_batch_GD(X_train, Y_train, X_val, Y_val)
        
    if EXP3:
        net = NeuralNet(X_train, y_train, lr=0.01, lamda=0.1)
        net.mini_batch_GD(X_train, Y_train, X_val, Y_val)
        
    if EXP4:
        net = NeuralNet(X_train, y_train, lr=0.01, lamda=1)
        net.mini_batch_GD(X_train, Y_train, X_val, Y_val)


    print("ends")



if __name__ == "__main__":
    main()
