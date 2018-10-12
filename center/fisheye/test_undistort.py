from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import cv2
import time

#Initializing the camera
camera = PiCamera()
#setting the resolution of the camera
camera.resolution = (640,480)
rawCapture = PiRGBArray(camera, size=camera.resolution)

#setting the undistort matrices
DIM=(1600, 1200)
K=np.array([[783.6035751658658, 0.0, 819.349819327776], [0.0, 772.8487024908837, 564.2299105244016], [0.0, 0.0, 1.0]])
D=np.array([[-0.17818296829734215], [0.004041559932549516], [0.01083544507979581], [-0.0030635894896379866]])

#looping over every frame
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    img = frame.array
    h,w = img.shape[:2]
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        cv2.imwrite("undist.jpg", undistorted_img)
        break
    