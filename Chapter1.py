import cv2

# import image
# img = cv2.imread("Resources/lena.jpeg") # The args (image path)

# Display The image
# cv2.imshow('Lena', img) # The args (screen name, the image)
# cv2.waitKey(3000) # delay func with args (time in mille seconds if 0 the delay will be inf)

# import video
# cap = cv2.VideoCapture("Resources/2024-03-31 19-27-14.mkv") # the args (the video path)

# video Display
# while True:
#    success, img = cap.read()
#    cv2.imshow("our video", img)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#   Using Webcam
cap = cv2.VideoCapture(0) # args(if 0 it will use the webcam defult if there is morre than one we write the id)
cap.set(3, 640) # edit the width id number 3
cap.set(4, 480) # edit the high id number 4

cap.set(10, 100) # edit the brightness id number 10


while True:
    success, img = cap.read()
    cv2.imshow("our video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(success)
        break
