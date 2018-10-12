from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2

camera = PiCamera()
camera.resolution = (1600,1200)
rawCapture = PiRGBArray(camera, size = camera.resolution)

#setting the undistort matrices
DIM=(1600, 1200)
'''
K=np.array([[783.6035751658658, 0.0, 819.349819327776], [0.0, 772.8487024908837, 564.2299105244016], [0.0, 0.0, 1.0]])
D=np.array([[-0.17818296829734215], [0.004041559932549516], [0.01083544507979581], [-0.0030635894896379866]])
'''
K=np.array([[774.4231548805052, 0.0, 822.6167410427034], [0.0, 769.3288387349592, 565.6990482106042], [0.0, 0.0, 1.0]])
D=np.array([[-0.17875854240547795], [0.02726679508811555], [-0.010188123245693159], [0.0024264322841337192]])

dim2 = (640,480)
dim3 = (640,640)#False

for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    img = frame.array
    dim1 = img.shape[:2][::-1]
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    
    if not dim2:
        dim2 = dim1
    if not dim3:    
        dim3 = dim1
        
    scaled_K = K * dim1[0] / DIM[0]
    scaled_K[2][2] = 1
    
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=0)
        
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    #crop_img = undistorted_img[0:480, 0:640]
    resized_img = cv2.resize(undistorted_img, (720, 540)) 
    cv2.imshow("undistorted", resized_img)
    key = cv2.waitKey(1) & 0xFF
    
    rawCapture.truncate(0)
    if key == ord('q'):
        break
    elif key == ord('c'):
        cv2.imwrite("undist.jpg", undistorted_img)
        break