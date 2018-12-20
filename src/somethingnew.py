import cv2
from matplotlib import pyplot as plt
import numpy as np 
from CNN import CNN
from skimage.morphology import skeletonize
from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import square, erosion, thin
from skimage.data import binary_blobs


def first_contours(file):
	img=cv2.imread(file)
	img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	retval,img_binary=cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)
	img_binary=cv2.bitwise_and(img_gray,img_gray,img_binary)
	retval,img_binary = cv2.threshold(img_binary,0,255,cv2.THRESH_OTSU)
	img_final=img_binary
	temp,contours,hierarchy= cv2.findContours(img_final,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	print("yo",len(contours))

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
	cv2.imshow("Hi there",final_img)
	cv2.waitKey()
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




file="delta.png"

img = cv2.imread(file)
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel = np.ones((3,3),np.uint8)

erosion = cv2.erode(img, kernel, iterations = 1)
cv2.imshow("1",erosion)
cv2.waitKey()

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gaussian_blur = gaussian(img_gray, sigma=1)
thresh_sauvola = threshold_minimum(gaussian_blur)
binary_img = gaussian_blur < thresh_sauvola

thinned_img = skeletonize(binary_img)
cv2.imshow("check" , thinned_img)
cv2.waitKey()
cv2.imwrite('messigray.png',erosion)
newcontours = first_contours('messigray.png')
cv2.imshow("check" , cv2.drawContours(erosion, newcontours, -1,(0,255,0),3))
cv2.waitKey()


X,Y,W,H = give_rect(newcontours)
khooni = 0


for i in range(len(X)):
	cv2.rectangle(img,(X[i],Y[i]),(X[i]+W[i],Y[i]+H[i]),(0,255,0),2)

cv2.imshow("rectangles", img)

for i in range(len(X)):
	if(X[i] == 0 and Y[i] == 0):
		khooni = i

X.pop(khooni)
Y.pop(khooni)
W.pop(khooni)
H.pop(khooni)

x,y,w,h = check(X,Y,W,H)

# for i in range(len(X)):
# 	cv2.rectangle(img,(X[i],Y[i]),(X[i]+W[i],Y[i]+H[i]),(0,255,0),2)




model = CNN(saved_model_path = "Final.hd5")

for i in range(len(x)):
	# cv2.imshow("check" + str(i), cv2.drawContours(erosion, newcontours, i,(255,0,255),3))
	print(model.predict(crop(cv2.imread(file), x[i], y[i], w[i], h[i])))
	# cv2.waitKey()

# for i in range(len(f)):
	# print(f[0])
