from constants import BATCH_SIZE, EPOCHS, IMG_COLS, IMG_ROWS, INPUT_SHAPE, NUM_CLASSES
from sklearn import svm

import numpy as np
import pickle as pkl

class SVM():
	def __init__(self, total_classes = NUM_CLASSES, saved_model_path = None):
		self._total_classes = total_classes
		if saved_model_path is not None:
			self._model = pickle.load(saved_model_path)
		else:
			self._model = svm.LinearSVC(verbose = 1)

	def train(self, Train_X, Train_Y):

		print("Training..")
		Train_Y = 
		self._model.fit(Train_X, Train_Y)

		print("Saving..")
		pickle.dump(self._model, open("SVM_Model.pkl", 'wb'))

	def test(self, Test_X, Test_Y):
		print("Accuracy is " ,self._model.score(Test_X, Test_Y))


	def predict(self, img):
		self._model.predict(img)




