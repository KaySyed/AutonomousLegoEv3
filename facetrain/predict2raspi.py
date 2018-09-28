from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import time

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train2.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


#cam = cv2.VideoCapture(0)
#font = cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    im = frame.array
#while True:
    #ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        # if(conf>50):
            # if(Id==1):
                # Id="Syed"
            # elif(Id==2):
                # Id="Obama"
            # elif(Id==3):
                # Id="Samu"
        # else:
            # Id="Unknown"
        #cv2.putText(im,str(Id), (x,y+h),font, 255)
        #cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im,str(Id),(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('im',im)
    rawCapture.truncate(0)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
#cam.release()
cv2.destroyAllWindows()
