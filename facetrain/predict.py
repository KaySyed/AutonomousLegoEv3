import cv2
import numpy as np
from PIL import Image

facerecog = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
facerecog.read('train2.yml')
#img = ("test_images/t1.jpg").copy()

#loading the image and converting it to gray scale
pilImage=Image.open("test_images/t1.jpg").convert('L')
#Now we are converting the PIL image into numpy array
imageNp=np.array(pilImage,'uint8')
# extract the face from the training image sample
faces=detector.detectMultiScale(imageNp)
#If a face is there then append that in the list as well as Id of it
for (x,y,w,h) in faces:
    #cv2.rectangle(img,(x-50, y-50),(x+w+50,y+h+50),(255,0,0),2)
    Id, conf = facerecog.predict(pilImage[y:y+h,x:x+w])
    print("Id = " + Id)
    break