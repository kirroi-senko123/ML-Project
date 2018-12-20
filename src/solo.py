import cv2
from matplotlib import pyplot as plt
import numpy as np 
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert



file="alpha_beta.png"

img = cv2.imread(file)
cv2.imshow("rectangles0", img)
cv2.waitKey()

img = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow("rectangles1", img)
cv2.waitKey()

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("rectangles", im_bw)
cv2.waitKey()

skeleton = skeletonize(im_bw)
plt.imsave('output.png',skeleton, format="png", cmap="hot") 


cv2.waitKey()

