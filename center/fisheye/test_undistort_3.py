from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import rpyc
from time import sleep
import math

#con = rpyc.classic.connect('ev3dev.local')
#motors = con.modules['ev3dev2.motor']
#sensors = con.modules['ev3dev2.sensor.lego']

#ev3.sound.Sound().speak('I AM ON')

#mediumMotor = motors.MediumMotor('outA')

#us = sensors.UltrasonicSensor()
#ts = sensors.TouchSensor()

#mediumMotor.reset()

# Instantiate OCV kalman filter
class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted


camera = PiCamera()
camera.resolution = (1600,1200)
camera.framerate = 8
rawCapture = PiRGBArray(camera, size = camera.resolution)

#setting the undistort matrices

DIM=(1600, 1200)

K=np.array([[774.4231548805052, 0.0, 822.6167410427034], [0.0, 769.3288387349592, 565.6990482106042], [0.0, 0.0, 1.0]])
D=np.array([[-0.17875854240547795], [0.02726679508811555], [-0.010188123245693159], [0.0024264322841337192]])


dim2 = (880,660)
dim3 = (880,880) #False

def run():
    lm1.run_forever(speed_sp= -150)
    lm2.run_forever(speed_sp= 150)
    sleep(1)

def stop():
    lm1.stop(stop_action="hold")
    lm2.stop(stop_action="hold")
    sleep(2)


def func(nm):    
    previous_diff = 0
    totm = 0
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
    added = 0
    roiy = 350
    
    kfObj = KalmanFilter()
    predictedCoords = np.zeros((2, 1), np.float32)

    for val in range(0, height, p):
        hist = np.sum(nm[height-val:height-(val-p),:], axis=0)
        left_max = np.argmax(hist[:height//2])
        right_max = np.argmax(hist[height//2:]) + height//2
        y = height-(val-p)
        cv2.line(warpedorg, (0, roiy), (width,roiy), (0,255,255), 1)
        if y < roiy:
            continue
        #cv2.line(nm,(0,y),(width,y),(255,255,255),1)

        y = height-val
        avgm = 0

        #cv2.circle(warpedorg, (left_max,y), 3, (0,0,0), -1)
        #cv2.rectangle(warpedorg,(left_max-60, y-10),(left_max+60, y+10),(255,255,255),1)
        writeL = False
        writeR = False
        boxth = 70
        if len(lx) != 0:
            if left_max > (lx[-1] - boxth):
                if left_max < (lx[-1] + boxth +10):
                    cv2.circle(warpedorg, (left_max,y), 3, (0,0,0), -1)
                    #cv2.rectangle(warpedorg,(left_max-60, y-10),(left_max+60, y+10),(255,255,255),1)
                    lx.append(left_max)
                    ly.append(y)
                    writeL =True
        elif left_max > 7:
            lx.append(left_max)
            ly.append(y)
            writeL =True
            
            
        if len(rx) != 0 and right_max > 310:
            if right_max > (rx[-1] - boxth):
                if right_max < (rx[-1] + boxth):
                    cv2.circle(warpedorg, (right_max,y), 3, (0,0,0), -1)
                    #cv2.rectangle(warpedorg,(right_max-60, y-10),(right_max+60, y+10),(255,255,255),1)
                    rx.append(right_max)
                    ry.append(y)
                    writeR =True
                elif len(rx) == 1:
                    rx.append(right_max)
                    ry.append(y)
                    writeR =True
            elif len(rx) == 1:
                rx.append(right_max)
                ry.append(y)
                writeR =True
        elif right_max > 0 and right_max > 310:
            rx.append(right_max)
            ry.append(y)
            writeR =True
        
        #if len(ma) > 0:
         #   avgm = (totm//len(ma))
        if writeL and writeR and y > roiy:
            m = ((right_max - left_max) //2) + left_max
            ma.append((m, y))
            cv2.circle(warpedorg, (m, y), 3, (0,0,255), -1)
            totm = (totm + m)
            predictedCoords = kfObj.Estimate(m, y)
            #avgm = totm//len(ma)
            #print(avgm)
        elif y > roiy and int(predictedCoords[0]) > 0 :
            #predictedCoords = kfObj.Estimate(0, y)
            predictedCoords = kfObj.Estimate(int(predictedCoords[0]), y)
            cv2.circle(warpedorg, (predictedCoords[0], y), 3, (0,0,255), -1)
            totm = int(totm + predictedCoords[0])
            ma.append((predictedCoords[0], y))
            #if len(ma) > 0 and y > roiy and added > 3:
                #ma.append((ma[len(ma)-1][0],y))
                #avgm = ma[len(ma)-1][0] #int(((totm // len(ma))) + (ma[len(ma)-1][0]))# * 0.5) - ma[len(ma)-1][0])
                #ma.append((avgm, y))
                #totm = ma[len(ma)-1][0] +totm
                #cv2.circle(warpedorg, (int(avgm), y), 3, (255,0,255), -1)
        if len(ma) > 1:
            cv2.line(warpedorg, ma[len(ma)-1], ma[len(ma)-2],(0,0,255),2)
    #print(len(ma))
    #sum_lane_center = 0
    #len(ma) = len(ma) - 2
    
    if len(ma) > 0:        
        tan = math.degrees((math.atan2(ma[1][1]-ma[len(ma)-1][1], ma[1][0]-ma[len(ma)-1][0])))
        tan = (90-tan)
        print('tan = ' + str(tan))
        #for r in range (len(ma), 1, -1):
        #    cv2.line(warpedorg, ma[r-1], ma[r-2],(0,0,255),2)
        #    if r < len(ma):
        #        sum_lane_center = (ma[r-1][0] + sum_lane_center)
                #print(ma[r-1][0])
        avg_lane_center = totm // len(ma)
        diff = 300 - avg_lane_center
        print('difference = ' + str(diff))
        diff = int(diff *0.7)
        #print('difference scaled = ' + str(diff))
        if diff > 0 and diff > 70:
            diff = 70
        elif diff < 0 and diff < -70:
            diff = -70
        print('adjusted diff = ' + str(diff))
        diff = diff + tan
        print('adjusted after tangent calculated = ' + str(diff) )
        #print('reached here')
        #if abs(diff - mediumMotor.position) > 8:
            #print('position is ' + str(mediumMotor.position) + '. set steering to ' + str(diff)+ '. difference is ' + str(abs(diff - mediumMotor.position)) + '. average is ' + str(avg_lane_center) + ' - needs motor moving')
            #print('abs ' + str(abs(diff - mediumMotor.position)))
            #print('setting motor pos')
            #mediumMotor.stop()
            #mediumMotor.on_to_position(speed = 8, position = int(diff) , brake = False)
            #mediumMotor.stop()
        
        print('----')
        
        #diff_difference = 0

        #diff_difference = previous_diff + diff
        
        
        #print ('diff difference ' + str(diff_difference))
        #print ('motor pos  =' + str(mediumMotor.position))

        '''
        if previous_diff == 0:
            if diff == 250:
                print('initial 250')                
            elif diff > 60:
                previous_diff = 60
                #print("set position to 1 " + str(diff))
                #mediumMotor.on_to_position(speed = 25, position = 85, brake = False)
            elif diff < -60:
                previous_diff = -60
                #print("set position to 2 " + str(diff))
                #mediumMotor.on_to_position(speed = 25, position = -85, brake = False)
            else:
                previous_diff = diff            
                val = diff - mediumMotor.position
                if ((val> 0 and val > 5) or (val < 0 and (val * -1) > 5)):
                    #print('val = ' + str(val))
                    if ((diff * 1.5) > 60):
                        diff = 60
                        #print("set position to 3 " + str(diff))
                    elif ((diff* 1.5) < -60):
                        diff = -60
                        #print("set position to 3 " + str(diff))
                    else:
                        diff = diff * 1.5
                    #mediumMotor.on_to_position(speed = 25, position = int(diff) , brake = False)
    
    #print( previous_diff)
        '''
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

#run()

for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    previous_diff = 0
    '''if ts.is_pressed:
        print('exiting')
        mediumMotor.stop()
       exit()
    '''
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
    #cv2.imshow('img', resized_img)
    cropped_img = resized_img[312:480, 201:830]
    #cv2.imshow('img', cropped_img)
    resized_img = cv2.resize(cropped_img, (1200,320))
    '''cv2.circle(resized_img, (66, 497), 3, (0,0,255), -1)
    cv2.circle(resized_img, (66, 682), 3, (0,255,0), -1)
    cv2.circle(resized_img, (320, 43), 3, (255,0,0), -1)
    cv2.circle(resized_img, (1189, 320), 3, (0,0,0), -1)
    '''
    #cv2.imshow('img', resized_img)

    orig_pts = np.float32([[497, 66], [680, 66], [43,320],[1200,320]])
    dest_pts = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])

    M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
    warpedorg = cv2.warpPerspective(resized_img, M, (600,600))
    #cv2.imshow('img', warpedorg)
    warpedbw= cv2.cvtColor(warpedorg, cv2.COLOR_BGR2GRAY)
    
    th = 180
    th = 80
    warpedbw[warpedbw < th] = 0    # Black
    warpedbw[warpedbw >= th] = 255 # White
    
    #cv2.imshow("bw", warpedbw)
        
    func(warpedbw)
    #cv2.imshow("un", warpedbw)
    
    
    cv2.line(warpedorg,(300,0),(300,600),(255,0,0),2)
    cv2.imshow("undistorted", warpedorg)
    key = cv2.waitKey(1) & 0xFF
    
    rawCapture.truncate(0)
    if key == ord('q'):
        m#ediumMotor.stop()
        break
    elif key == ord('c'):
        cv2.imwrite("undist.jpg", warpedorg)
        break
