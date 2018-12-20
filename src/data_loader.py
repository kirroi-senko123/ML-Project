from constants import DATA_POINTS, NUM_CLASSES, TEST_RATIO
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import random


class DataLoader:
    @staticmethod
    def load_data(inp_path, max_num_points=DATA_POINTS, max_num_classes=NUM_CLASSES):
        symbol_to_num_map = np.load("/users/btech/sanketyd/cs771/ML_Project/model/symbol_to_num_map.npy").item()
        data_input = {}
        class_count = 0
        for folder in os.listdir(os.path.abspath(inp_path)):
            data_input[symbol_to_num_map[folder]] = []
            data_count = 1
            if class_count > max_num_classes:
                break
            files = os.listdir(os.path.abspath(os.path.join(os.path.abspath(inp_path), folder)))
            random.shuffle(files)

            for filename in files:
                if data_count > max_num_points:
                    break
                file_path = os.path.join(os.path.abspath(inp_path), folder, filename)
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img is not None:
                    data_input[symbol_to_num_map[folder]].append(img)
                data_count += 1

            data_input[symbol_to_num_map[folder]] += [random.choice(data_input[symbol_to_num_map[folder]]) for _ in range(max_num_points-data_count)]
            class_count += 1

        Xtrain = []
        Ytrain = []
        Xtest = []
        Ytest = []

        for key in data_input.keys():
            Ytemp = [key for _ in range(len(data_input[key]))]
            X, x, Y, y = train_test_split(data_input[key], Ytemp, test_size=TEST_RATIO)
            Xtrain += X
            Ytrain += Y
            Xtest += x
            Ytest += y

        return np.array(Xtrain), np.array(Ytrain), np.array(Xtest), np.array(Ytest)
