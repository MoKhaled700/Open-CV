import cv2
import numpy as np

img = cv2.imread('Resources/lena.jpeg')

print(img.shape)

imgResized = cv2.resize(img, (500,500)) # the args(width, high)
print(imgResized.shape)

imgCropped = img[0:100, 100:200]
#cv2.imshow('img', img)
cv2.imshow('resized', imgResized)
cv2.imshow('Cropped', imgCropped)

cv2.waitKey(0)