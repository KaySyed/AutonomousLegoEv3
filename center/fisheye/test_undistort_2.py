from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import rpyc

con = rpyc.classic.connect('ev3dev.local')
ev3 = con.modules['ev3dev2']

#ev3.sound.Sound().speak('I AM ON')

mediumMotor = ev3.motor.MediumMotor('outA');
mediumMotor.reset()
print(mediumMotor.position)

previous_diff = 0

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

dim2 = (880,660)
dim3 = (880,880)#False
def func(nm):
    
    previous_diff = 0
    i = 0
    height, width = nm.shape[:2]
    p=10

    lx = []
    ly = []
    rx = []
    ry = []
    ma = []
    my = []
    #a= np.zeros((int((1000//p)+1), 2), dtype = "int32")
    #b= np.zeros((int((1000//p)+1), 2), dtype = "int32")
    for val in range(0, height, p):
        hist = np.sum(nm[height-val:height-(val-p),:], axis=0)
        left_max = np.argmax(hist[:height//2])
        right_max = np.argmax(hist[height//2:]) + height//2

        y = height-(val-p)

        cv2.line(nm,(0,y),(width,y),(255,255,255),1)

        y = height-val

        #cv2.circle(warpedorg, (left_max,y), 3, (0,0,0), -1)
        #cv2.rectangle(warpedorg,(left_max-40, y-10),(left_max+40, y+10),(255,255,255),1)
        if len(lx) != 0:
            if left_max > (lx[-1] - 40):
                if left_max < (lx[-1] + 40):
                    cv2.circle(warpedorg, (left_max,y), 3, (0,0,0), -1)
                    lx.append(left_max)
                    ly.append(y)
        elif left_max > 7:
            lx.append(left_max)
            ly.append(y)
        if len(rx) != 0:
            if right_max > (rx[-1] - 40):
                if right_max < (rx[-1] + 40):
                    cv2.circle(warpedorg, (right_max,y), 3, (0,0,0), -1)
                    #cv2.rectangle(warpedorg,(right_max-40, y-10),(right_max+40, y+10),(255,255,255),1)
                    rx.append(right_max)
                    ry.append(y)
                elif len(rx) == 1:
                    rx.append(right_max)
                    ry.append(y)
            elif len(rx) == 1:
                rx.append(right_max)
                ry.append(y)
        elif right_max > 0:
            rx.append(right_max)
            ry.append(y)  

    for i in range (0, len(lx), 1):
        for j in range (0,len(rx), 1):
            if ly[i] == ry[j] and ly[i] > 350:                                
                m = ((rx[i] - lx[i]) //2) + lx[i]
                ma.append((m, ly[i]))
                my.append(ly[i])
                cv2.circle(warpedorg, (m, ly[i]), 3, (0,0,255), -1)
                '''
                if m >= 300-10 or m <=300+10:
                    cv2.circle(warpedorg, (m,ly[i]), 3, (0,255,255), -1)
                else:
                    cv2.circle(warpedorg, (m, ly[i]), 3, (0,0,255), -1)
        '''
    #print(len(ma))
    sum_lane_center = 0
    lenma = len(ma) -2
    for r in range (len(ma), 1, -1):
        cv2.line(warpedorg, ma[r-1], ma[r-2],(0,0,255),2)
        if r < lenma:
            sum_lane_center = (ma[r-1][0] + sum_lane_center)
            #print(ma[r-1][0])
    avg_lane_center = sum_lane_center // lenma
    diff = 250 - avg_lane_center
    #print('difference = ' + str(diff))
    print('----')
    
    diff_difference = 0

    diff_difference = previous_diff + diff
    
    
    print ('diff difference ' + str(diff_difference))
    print ('motor pos  =' + str(mediumMotor.position))
    
    if previous_diff == 0:
        if diff == 250:
            print('probably instantiation')
        elif diff > 85:
            previous_diff = 85
            print("set position to 1 " + str(diff))
            #mediumMotor.on_to_position(speed = 10, position = 85, brake = False)
        elif diff < -60:
            previous_diff = -60
            print("set position to 2 " + str(diff))
            #mediumMotor.on_to_position(speed = 10, position = -85, brake = False)
        else:
            previous_diff = diff            
            val = diff - mediumMotor.position
            if ((val> 0 and val > 5) or (val < 0 and (val * -1) > 5)):
                print("set position to 3 " + str(diff * 1.5))
                print('val = ' + str(val))
                if ((diff * 1.5) > 60):
                    diff = 60
                elif ((diff* 1.5) < -60):
                    diff = -60
                else:
                    diff = diff * 1.5
                #mediumMotor.on_to_position(speed = 10, position = int(diff) , brake = False)
    
    #print( previous_diff)
        
        #cv2.rectangle(warpedorg,(right_max-40, y-10),(right_max+40, y+10),(255,255,255),1)
#        rx.append(right_max)
#        ry.append(y)
        #b[int(val/p)] = [right_max, y]
    '''
    plt.plot(lx, ly, 'o')
    #plt.imshow(nm, cmap="gray")
    lf = np.polyfit(lx, ly, 2)
    rf = np.polyfit(rx, ry, 2)
    plt.plot(lx, np.polyval(lf,lx), 'r-', linewidth = 4.0)
    plt.plot(rx, np.polyval(rf,rx), 'g-', linewidth = 4.0)
    print(lx)
    '''
    return nm

for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    previous_diff = 0
    img = frame.array
    dim1 = img.shape[:2][::-1]
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    
    if not dim2:
        dim2 = dim1
    if not dim3:    
        dim3 = dim1
        
    scaled_K = K * dim1[0] / DIM[0]
    scaled_K[2][2] = 1
    
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=1)
        
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    #crop_img = undistorted_img[0:480, 0:640]
    resized_img = cv2.resize(undistorted_img, (1280, 960))
    
    cropped_img = resized_img[312:480, 201:830]
    resized_img = cv2.resize(cropped_img, (1200,320))
    
    orig_pts = np.float32([[497, 66], [682, 66], [43,320],[1189,320]])
    dest_pts = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])

    M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
    warpedorg = cv2.warpPerspective(resized_img, M, (600,600))
    
    warpedbw= cv2.cvtColor(warpedorg, cv2.COLOR_BGR2GRAY)
    
    th = 120
    warpedbw[warpedbw < th] = 0    # Black
    warpedbw[warpedbw >= th] = 255 # White
     
    func(warpedbw)
    #cv2.imshow("un", warpedbw)
    cv2.line(warpedorg,(300,0),(300,600),(255,0,0),2)
    cv2.imshow("undistorted", warpedorg)
    key = cv2.waitKey(1) & 0xFF
    
    rawCapture.truncate(0)
    if key == ord('q'):
        break
    elif key == ord('c'):
        cv2.imwrite("undist.jpg", warpedorg)
        break