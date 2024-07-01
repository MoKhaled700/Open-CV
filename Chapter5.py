# i don't understand this chapter

import cv2
import numpy as np
width, height = 250, 350
img = cv2.imread("Resources/playing-cards-5717969.webp")
pts1 = np.float32([[111,219],[287,188],[154,485],[352,440]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width,height))

cv2.imshow("image", img)
cv2.imshow("output", imgOutput)

cv2.waitKey(0)