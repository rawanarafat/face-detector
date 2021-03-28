import cv2
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')


trained_faces_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread('img.jpg')
greyscalled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coord = trained_faces_data.detectMultiScale(greyscalled_img)
print(face_coord)

for (x,y,w,h) in face_coord:
    cv2.rectangle(img , (x,y) , (x+w , y+h) ,(0,255,0),2)

cv2.imshow("Face detection" , img)
cv2.waitKey()
