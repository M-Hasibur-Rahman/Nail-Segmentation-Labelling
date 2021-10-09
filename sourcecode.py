#The following code was written by M Hasibur Rahman, 295459
import cv2 as cv
import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import feature
from matplotlib import pyplot as plt

#part for showing binary segmentation and using ranges
# using a sample image from data set
img = cv.imread('images/4c48acb6-e402-11e8-97db-0242ac1c0002.jpg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# minRange for min skin color Rnage
# maxRange for maximum skin color Range
lower = np.array([10, 10, 60], dtype = "uint8") 
upper = np.array([20, 150, 255], dtype = "uint8")

mask = cv.inRange(hsv, lower,upper)

kernel = np.ones((3, 3), np.uint8)
#blur , optionally use opening
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
mask = cv.medianBlur(mask, 5)


cv.imshow("mask", mask)
cv.imshow("originalimg", img)

#-------------------------------------------------------------------------------------------------
#part for canny edge detection and calculating iou values
#image to be evaluated and set one by one
img = cv.imread('images/1eecab90-1a92-43a7-b952-0204384e1fae.jpg')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#the image given by teacher for the prediction and will be used for jaccard formula 
segimg = cv.imread('labels/1eecab90-1a92-43a7-b952-0204384e1fae.jpg')
#convert to hsv so that we can compare the masked and predicted result
seggray = cv.cvtColor(segimg, cv.COLOR_BGR2GRAY)

#using canny() from skimage with sigma value = 1
edges = feature.canny(imgray, sigma=1)
fill = ndi.binary_fill_holes(edges)
#convert image from skimage to opencv
masked = img_as_ubyte(fill)



#jaccard index (iou) - formula gotten from project description
iou = np.sum(np.logical_and(masked, seggray)) / np.sum(np.logical_or(masked, seggray))
print(iou*100)
#plt.imshow(fill)
#plt.show()

cv.imshow("ourresult", masked)
cv.imshow("prediction",seggray)
cv.waitKey(0)

