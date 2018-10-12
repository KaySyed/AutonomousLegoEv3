from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time

w = 1600
h = 1200

camera = PiCamera()
camera.resolution = (w,h)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(camera.resolution))
i = 0
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port = True):
       
    image=frame.array
    cv2.imshow("frame", image)
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord("c"):
        i = i+1 
        print("Capture # %s"%i)
        cv2.imwrite('calib_images/img%s.jpg'%i, image)
    elif key == ord("q"):
        break