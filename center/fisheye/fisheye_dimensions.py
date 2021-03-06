# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2

w = 1600
h = 1200
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (w, h)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=camera.resolution)

i=0
# allow the camera to warmup
time.sleep(0.1)

DIM=(1600, 1200)
K=np.array([[774.4231548805052, 0.0, 822.6167410427034], [0.0, 769.3288387349592, 565.6990482106042], [0.0, 0.0, 1.0]])
D=np.array([[-0.17875854240547795], [0.02726679508811555], [-0.010188123245693159], [0.0024264322841337192]])

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	h,w = image.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        undistorted_img = cv2.resize(undistorted_img, (720, 540))
        cv2.imshow("undistorted", undistorted_img)
        key = cv2.waitKey(1) & 0xFF
	'''
        cv2.circle(image, (int(w-(w*0.8953)), (int(h-(h*0.8864)))), 3, (255,0,0), -1)
        cv2.circle(image, (int(w-(w*0.0812)), (int(h-(h*0.8864)))), 3, (255,0,0), -1)
        cv2.circle(image, (int(w-(w*0.8953)), (int(h-(h*0.1690)))), 3, (255,0,0), -1)
        cv2.circle(image, (int(w-(w*0.0812)), (int(h-(h*0.1690)))), 3, (255,0,0), -1)
        
        #cv2.rectangle(image, (int(w-(w*0.8953)), (int(h-(h*0.8864)))), (int(w-(w*0.0812)), (int(h-(h*0.1690)))), (0,0,255), 3)
        
        #img = image[int(h-(h*0.8864)):(int(h-(h*0.8864)))+(int(h-(h*0.1690)) - int(h-(h*0.8864))), int(w-(w*0.8953)):(int(w-(w*0.8953)))+(int(w-(w*0.0812)) - int(w-(w*0.8953)))]
	# show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
'''
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	# if the `q` key was pressed, break from the loop
	if key == ord("c"):
            i = i + 1
            print("iteration %s"%i)
            cv2.imwrite("doneimg%s.jpg"%i, undistorted_img)
        elif key == ord("q"):
	    break
	