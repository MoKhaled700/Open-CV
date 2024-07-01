import cv2
import numpy as np

img = cv2.imread('Resources/lena.jpeg')
kernel = np.ones((5,5), np.uint8)


imgGrey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
imgBlure = cv2.GaussianBlur(imgGrey, (7,7), 0) #apply a gaussian filter and the args(the image, filter size, sigma"standard deviation")

# Apply Canny edg detector on the image args(image, 1st threshold, 2nd threshold)
imgCanny =cv2.Canny(img, 150, 200)

imgDialation= cv2.dilate(imgCanny, kernel, iterations=1)

imgEroded = cv2.erode(imgDialation, kernel, iterations=1)
#cv2.imshow('grey', imgGrey)
#cv2.imshow('blur', imgBlure)
#cv2.imshow('Canny', imgCanny)
cv2.imshow('dilation', imgDialation)
cv2.imshow('eroded', imgEroded)


cv2.waitKey(0)