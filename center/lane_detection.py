# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1280, 960)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(1280, 960))

showimg = 0

# allow the camera to warmup
time.sleep(0.1)

def func(nm):
    
    height, width = nm.shape[:2]
    p=100
    lx = np.zeros((int((1000//p)+1), 1), dtype = "int32")
    ly= np.zeros((int((1000//p)+1), 1), dtype = "int32")
    lx = []
    ly = []
    rx = []
    ry = []
    a = []
    #a= np.zeros((int((1000//p)+1), 2), dtype = "int32")
    #b= np.zeros((int((1000//p)+1), 2), dtype = "int32")
    for val in range(p, 1000, p):
        hist = np.sum(nm[1000-val:1000-(val-p),:], axis=0)
        left_max = np.argmax(hist[:500])
        right_max = np.argmax(hist[500:]) +500

        y = 1000-(val-p)

        cv2.line(nm,(0,y),(660,y),(255,255,255),1)

        y = 1000-val

        cv2.circle(warpedorg, (left_max,y), 3, (0,0,0), -1)
        #cv2.rectangle(nm,(left_max-30, y-10),(left_max+30, y+10),(255,255,255),1)
        if len(lx) != 0:
            if left_max > (lx[-1] - 20):
                if left_max < (lx[-1] + 20):
                    lx.append(left_max)
                    ly.append(y)
        elif left_max > 7:
            lx.append(left_max)
            ly.append(y)            

        cv2.circle(warpedorg, (right_max, y), 3, (0,0,0), -1)
        #cv2.rectangle(nm,(right_max-30, y-10),(right_max+30, y+10),(255,255,255),1)
        rx.append(right_max)
        ry.append(y)
        #b[int(val/p)] = [right_max, y]

    plt.plot(lx, ly, 'o')
    #plt.imshow(nm, cmap="gray")
    lf = np.polyfit(lx, ly, 2)
    rf = np.polyfit(rx, ry, 2)
    plt.plot(lx, np.polyval(lf,lx), 'r-', linewidth = 4.0)
    plt.plot(rx, np.polyval(rf,rx), 'g-', linewidth = 4.0)
    print(lx)
    return nm

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    #image = frame.array
    image = cv2.imread("track120p.png")
    
    '''
    #bw = np.asarray(gray).copy()

    cv2.line(bw, (340,270), (950, 270), (0,0,0), 3)
    cv2.line(bw, (950,270), (1480, 660), (0,0,0), 3)
    cv2.line(bw, (1480,660), (-125, 660), (0,0,0), 3)
    cv2.line(bw, (-125,660), (340, 270), (0,0,0), 3)
    '''    

    orig_pts = np.float32([[340, 270], [950, 270], [-125,660],[1480,660]])
    dest_pts = np.float32([[0, 0], [650, 0], [0, 1000], [660, 1000]])

    M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
    warpedorg = cv2.warpPerspective(image, M, (660,1000))
    
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    warpedgray =  cv2.cvtColor(warpedorg, cv2.COLOR_BGR2GRAY)
    
    warpedbw = warpedgray.copy()
    '''
    plt.imshow(gray, cmap="gray")
    cv2.imshow("gray", gray)
    cv2.waitKey()
    plt.show()
    '''
    th = 200
    warpedbw[warpedbw < th] = 0    # Black
    warpedbw[warpedbw >= th] = 255 # White
       
    calcwarped = func(warpedbw)
    
    M = cv2.getPerspectiveTransform(dest_pts, orig_pts)
    nmuw = cv2.warpPerspective(warpedorg, M, (1280,960))
    
    #cv2.imshow("track", nm)
    #plt.show()
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
    showimg = nmuw
    break

cv2.imshow("track", warpedorg)
key = cv2.waitKey(0)
plt.imshow(cv2.cvtColor(warpedorg, cv2.COLOR_BGR2RGB))
plt.show()
