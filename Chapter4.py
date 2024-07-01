import cv2
import numpy as np

img = np.zeros((512,512,3))
#img[:] = 0,0,255

# draw a line
#cv2.line(img, (0,0), (300,300), (0, 0, 255), 3) # args (image , point 1, point 2, color, thikness 'in pixels')
# if you used cv2.FILLED it will fill the rectangle or any shape

# draw a rectangle
#cv2.rectangle(img, (0,0),(200,200), (255,0,0),2) # the arguments(image, point, the corner 'diagonal' point, color,width)
#cv2.rectangle(img, (100,100),(200,200), (255,0,0),cv2.FILLED)

# Draw circle
#cv2.circle(img, (300,300), 30, (0,255,0), 2)# args(image, the center, radius, color, width)

# Text on image
cv2.putText(img, "Mohamed", (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),5) # args(image , the text, the origin we will start, the font ,scale, color)


cv2.imshow('image', img)
cv2.waitKey(0)

