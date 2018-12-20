from constants import IMG_COLS, IMG_ROWS, VALIDATION_TRAIN_RATIO
from CNN import CNN
from data_loader import DataLoader
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = DataLoader.load_data("/users/btech/sanketyd/cs771/ML_Project/dataset")

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=VALIDATION_TRAIN_RATIO)

    X_train = X_train.reshape(X_train.shape[0], IMG_ROWS, IMG_COLS, 3)
    X_val = X_val.reshape(X_val.shape[0], IMG_ROWS, IMG_COLS, 3)
    X_test = X_test.reshape(X_test.shape[0], IMG_ROWS, IMG_COLS, 3)

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    X_train = X_train/255
    X_val = X_val/255
    X_test = X_test/255

    model = CNN()
    model.train(X_train, Y_train, X_val, Y_val)
    model.test(X_test, Y_test)
