import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np 
from tree import Node
from tree import Tree
from CNN import CNN
from skimage.morphology import skeletonize


def predict(fname):
	answer = ""
	def printTree(start):
		nonlocal answer
		if start is None:
			return
		answer += start.label
		print(start.label, end="")
		if start.child is not None:
			answer += '_{'
			print ('_{', end="")
			printTree(start.child)
			answer += '}'
			print ('}', end="")
		if start.top is not None:
			answer += '^{'
			print ('^{', end="")
			printTree(start.top)
			answer += '}'
			print ('}', end="")
		printTree(start.sibling)


	def first_contours(file):
		img=cv2.imread(file)
		img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		retval,img_binary=cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)
		img_binary=cv2.bitwise_and(img_gray,img_gray,img_binary)
		retval,img_binary = cv2.threshold(img_binary,0,255,cv2.THRESH_OTSU)
		img_final=img_binary
		temp,contours,hierarchy= cv2.findContours(img_final,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		# print("yo",len(contours))

		return contours

	def give_rect(contours):
		X=[]
		Y=[]
		H=[]
		W=[]
		print("no of contours=\t",len(contours))
		for i in range(len(contours)):
			[x,y,w,h]=cv2.boundingRect(contours[i])
			X.append(x)
			Y.append(y)
			W.append(w)
			H.append(h)
		# print("next=\t",len(points),len(dimensions))
		return X,Y,W,H

	def square(img, x, y, w, h):
		diff = abs(w-h)
		padding = int(diff/2)
		if(w>h):
			padded_img = cv2.copyMakeBorder(img, padding, diff - padding, 0, 0, cv2.BORDER_CONSTANT, value = (255, 255, 255))
		else:
			padded_img = cv2.copyMakeBorder(img, 0, 0, diff - padding, padding, cv2.BORDER_CONSTANT, value = (255, 255, 255))
		plt.imsave('Output', padded_img)
		final_img = cv2.resize(padded_img, (45,45))
		# cv2.imshow("Hi there",final_img)
		# cv2.waitKey()
		return final_img

	def crop(img, x, y, w, h):
		cropped_img = img[y:y+h, x:x+w]
		final_img = square(cropped_img, x, y, w, h)

		return final_img

	def check(x,y,w,h):
		final_x = []
		final_y = []
		final_w = []
		final_h = []
		for i in range(len(x)):
			flag = 0
			for j in range(len(x)):
				if (((x[i] > x[j]) and (x[i] + w[i]) < (x[j] + w[j])) and ((y[i] > y[j]) and (y[i] + h[i]) < (y[j] + h[j]))):
					print(i, j)
					flag = 1
			if(flag == 0):
				final_x.append(x[i])
				final_y.append(y[i])
				final_w.append(w[i])
				final_h.append(h[i])
		return final_x, final_y, final_w, final_h


	# Reading the file

	file=fname
	img = cv2.imread(file)

	# Padding the image with whitespace

	img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value = (255, 255, 255))

	# Eroding image to remove noisy contour detection

	kernel = np.ones((2,2),np.uint8)
	erosion = cv2.dilate(img,kernel,iterations = 1)
	erosion = cv2.erode(img, kernel, iterations = 1)


	# cv2.imshow("1",erosion)
	# cv2.waitKey()
	cv2.imwrite('messigray.png',erosion)

	newcontours = first_contours('messigray.png')

	# cv2.imshow("check" , cv2.drawContours(erosion, newcontours, -1,(0,255,0),3))
	# cv2.waitKey()

	# Get bounding rectangles for all the contours

	X,Y,W,H = give_rect(newcontours)

	# Remove the rectangles completely subsumed into larger rectangles
	subsumed = 0

	for i in range(len(X)):
		cv2.rectangle(img,(X[i],Y[i]),(X[i]+W[i],Y[i]+H[i]),(0,255,0),2)

	# cv2.imshow("rectanglfes", img)

	for i in range(len(X)):
		if(X[i] == 0 and Y[i] == 0):
			subsumed = i


	# cv2.imshow("check" + str(i), cv2.drawContours(erosion, newcontours, subsumed,(255,0,255),3))

	X.pop(subsumed)
	Y.pop(subsumed)
	W.pop(subsumed)
	H.pop(subsumed)

	x,y,w,h = check(X,Y,W,H)


	model = CNN(saved_model_path = "senior_24_128.hd5")

	# Building the Tree

	print(model._model.summary())

	labels = []

	for i in range(len(x)):
		labels.append(model.predict(crop(cv2.imread('messigray.png', cv2.IMREAD_GRAYSCALE), x[i], y[i], w[i], h[i])))

	# print(labels)

	print(x)
	print(y)
	print(labels)

	coordinate = [(x[i], y[i], w[i], h[i], labels[i]) for i in range(len(x))]

	coordinate.sort(key = lambda k : k[0])

	print(coordinate)

	X_cord = [y[0] for y in coordinate]
	Y_cord = [y[1] for y in coordinate]
	W_cord = [y[2] for y in coordinate]
	H_cord = [y[3] for y in coordinate]
	labels = [y[4] for y in coordinate]


	start = Node(index = 0, label = labels[0])
	parent_avg = (Y_cord[0] + (H_cord[0]/2))
	prev_node = start

	tree = Tree(coordinate)

	print(len(X_cord))
	for i in range(1,len(X_cord)):
		print("checkw\t" + labels[i] + "\t" + prev_node.label + "\t",i)
		curr_node = Node(index = i, label = labels[i])
		# print("afa\t" + prev_node.label)
		prev_node = tree.locate_and_label(prev_node = prev_node, curr_node = curr_node)

	printTree(start)

	return answer

if __name__ == "__main__":
	res = predict(sys.argv[1])
	with open("{}_res".format(sys.argv[1]), 'w') as f:
		f.write(res)

