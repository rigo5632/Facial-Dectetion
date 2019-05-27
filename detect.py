import cv2
import numpy as np

# To find certain facial features. I used pre-made machine learned files. (Come Standard with CV2)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Image of person
img = cv2.imread('Sample Images\\face.jpg')
# Color
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detection
faces = faceCascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    # Marks face with rectangle
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # Detection
    eyes = eyeCascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        # Marks Eyes
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# Shows image
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
