# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1280, 960)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(1280, 960))

IMAGE_H = 223
IMAGE_W = 1280

# allow the camera to warmup
time.sleep(0.1)

def func(nm):
    height, width = nm.shape[:2]
    for x in range (0, height, 20):
        hist = np.sum(nm[height-x:,:], axis=0)
        left_max = np.argmax(hist[:500])
        right_max = np.argmax(hist[500:]) +500
        
        y = height - (height-x)
        
        cv2.line(nm,(0,y),(660,y),(0,0,0),1)
        
        cv2.circle(nm, (left_max,y), 3, (0,0,0), -1)
        cv2.rectangle(nm,(left_max-20, y-10),(left_max+20, y+10),(0,255,0),1)
        
        cv2.circle(nm, (right_max, y), 3, (0,0,0), -1)
        cv2.rectangle(nm,(right_max-20, y-10),(right_max+20, y+10),(0,255,0),1)
        print(left_max)
        print(right_max)
    return nm

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	#image = frame.array
	image = cv2.imread("track120p2.png")
	bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        '''
        #bw = np.asarray(gray).copy()

        cv2.line(bw, (340,270), (950, 270), (0,0,0), 3)
        cv2.line(bw, (950,270), (1480, 660), (0,0,0), 3)
        cv2.line(bw, (1480,660), (-125, 660), (0,0,0), 3)
        cv2.line(bw, (-125,660), (340, 270), (0,0,0), 3)
        
        th = 200
        bw[bw < th] = 0    # Black
        bw[bw >= th] = 255 # White
        '''
        orig_pts = np.float32([[340, 270], [950, 270], [-125,660],[1480,660]])
        dest_pts = np.float32([[0, 0], [650, 0], [0, 1000], [660, 1000]])
        
        M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
        nm = cv2.warpPerspective(bw, M, (660,1000))
        
        nm = func(nm)
        
        cv2.imshow("track", nm)
	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
        
        
        '''
        # Pixel range is 0...255, 256/2 = 128
        th = 200
        bw[bw < th] = 0    # Black
        bw[bw >= th] = 255 # White
	'''
        rawCapture.truncate(0)
        #break