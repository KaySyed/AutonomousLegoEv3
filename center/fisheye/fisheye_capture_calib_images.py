from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time

w = 1280
h = 720

camera = PiCamera()
camera.resolution = (w,h)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(w, h))
i = 0
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port = True):
    i = i+1
    print("Iteration # %s"%i)
    image=frame.array
    cv2.imshow("frame", image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("c"):
        cv2.imwrite("\calib_images\img%s.jpg"%i, image)
        rawCapture.truncate(0)
    elif key == ord("q"):
        rawCapture.truncate(0)
        break