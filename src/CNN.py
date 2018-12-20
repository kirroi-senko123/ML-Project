from constants import BATCH_SIZE, EPOCHS, IMG_COLS, IMG_ROWS, INPUT_SHAPE, NUM_CLASSES
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import load_model, Sequential
from keras.optimizers import Adam

import cv2
import datetime
import numpy as np


class CNN:
    def __init__(self, input_shape=INPUT_SHAPE, total_classes=NUM_CLASSES, saved_model_path=None):
        self._input_shape = input_shape
        self._history = None
        self._model = None
        self._total_classes = total_classes
        if saved_model_path is not None:
            self._model = load_model(saved_model_path)
        else:
            self._createModel()

    def _createModel(self):
        model = Sequential()

        model.add(Conv2D(20, 3, 3, border_mode="same", activation="relu", input_shape=self._input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, 3, 3, border_mode="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(100, 3, 3, border_mode="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))

        model.add(Dense(self._total_classes, activation="softmax"))

        self._model = model
        self._model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])
        self._model.summary()

    def train(self, Xtrain, Ytrain, Xval, Yval):
        print(Xtrain.shape, Ytrain.shape, Xval.shape, Yval.shape)
        self._history = self._model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(Xval, Yval))
        self._model.save("/users/btech/sanketyd/cs771/ML_Project/model/model_{}.hd5".format(datetime.datetime.now()))

    def test(self, Xtest, Ytest):
        score = self._model.evaluate(Xtest, Ytest, verbose=1)
        print("Test Accuracy: {}".format(score[1]))

    def predict(self, img):
        num_to_symbol_map = np.load("num_to_symbol_map.npy").item()
        arr = []
        arr.append(img)
        arr = np.array(arr)
        arr = arr.reshape((1, IMG_ROWS, IMG_COLS, 1))
        arr = arr/255
        prediction = self._model.predict(arr)[0]
        print(prediction)
        print("Prediction: {}".format(num_to_symbol_map[np.argmax(prediction)]))
        return num_to_symbol_map[np.argmax(prediction)]